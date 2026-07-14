# llm4rec-bias

A minimal, locally-runnable LLM4Rec pipeline on MovieLens-100K for studying
**RL shortcut / bias mitigation** in recommendation. Scaled-down version of the
RL-Shortcut-Lab execution spec (MLLMRec-R1 route): LoRA SFT → LoRA merge →
GRPO with KL constraint, with the four bias-cue dimensions controllable in the
data layer and measured by dedicated probes.

Runs on an Apple-silicon Mac (tested: M3, 16 GB) with `Qwen/Qwen2.5-0.5B-Instruct`.

## Task

Next-movie **choice**: prompt = user's recent watch history (titles) + C=10
lettered candidates (ground-truth next item among popularity-sampled
negatives); the model answers with a letter. Reward: **+1** correct, **0**
wrong-but-valid, **−0.5** unparseable.

## Bias cues → controls (spec's four dimensions)

| Cue dimension | Control | Probe / metric |
|---|---|---|
| Popularity / exposure | `--neg-sampling pop\|uniform` | `pop_lift` (chosen item's popularity quantile − candidate mean) |
| Position | `--target-pos random\|first\|last\|middle` | full-permutation probe: accuracy-by-target-position curve + `spread`; `chosen_pos_hist` |
| Text framing | `--framing neutral\|evaluative` (popularity markers) | eval same split under both framings, compare HR@1 / pop_lift |
| History recency | `--history N` | regenerate data with different N, compare |

## Pipeline

```bash
uv sync

# 1. Data (downloads ml-100k, ~5MB). Default: pop-weighted negatives, random target position
uv run python -m llm4rec.data --out data

# variant datasets for probes, e.g.:
uv run python -m llm4rec.data --out data --neg-sampling uniform --suffix _uniform
uv run python -m llm4rec.data --out data --framing evaluative --suffix _eval

# 2. Baseline eval (zero-shot base model)
uv run python -m llm4rec.eval --max-examples 300 --position-probe 30 --out runs/eval_base.json

# 3. SFT (LoRA), ~1.2s/step on M3
uv run python -m llm4rec.sft --out runs/sft

# 4. Eval SFT checkpoint
uv run python -m llm4rec.eval --adapter runs/sft/final --max-examples 300 --position-probe 30 --out runs/eval_sft.json

# 5. GRPO on top of merged SFT weights (~4-15s/step depending on batch)
uv run python -m llm4rec.grpo --sft-adapter runs/sft/final --steps 300 --out runs/grpo

# 6. Eval GRPO checkpoint
uv run python -m llm4rec.eval --adapter runs/grpo/final --max-examples 300 --position-probe 30 --out runs/eval_grpo.json
```

GRPO logs per step: `reward`, `kl`, `shortcut/invalid_rate`,
`shortcut/chosen_pos_mean`, `shortcut/pop_lift` — the spec's per-step
deliverables. Checkpoints: base (stage 0), SFT (stage 1), GRPO (stage 2), plus
`save_steps` intermediates.

## Baseline finding (zero-shot, 40 test examples)

The base 0.5B model is already a pure **position-shortcut** policy: it picks
candidate A 70% of the time (rest J). Position probe: accuracy 1.0 when the
target is at slot A, 0.0 everywhere else (spread = 1.0). HR@1 0.15 vs 0.10
chance. So even before RL there is a strong prior cue for training to inflate
or suppress — exactly the phenomenon to track through SFT → GRPO.

## Experiment plan (screening → causal → mitigation)

1. **Screening**: train default pipeline; track `shortcut/*` per step. Compare
   `pop_lift` when trained on pop-sampled vs uniform negatives; position probe
   spread at each stage.
2. **Causal check**: hold content fixed, permute one cue (position probe does
   this; framing A/B does it for text). A cue is causal if choice tracks the
   cue, not the content.
3. **Mitigation** (equal budget comparisons):
   - KL strength sweep (`--beta`) — the generic baseline
   - Cue-randomized training data (position/framing randomization at data level)
   - Reward-side fixes: penalize choices that track the cue (edit `reward.py`)
   - Early stopping at KL/probe thresholds

## Layout

```
src/llm4rec/
  prompts.py   templates + framing cue + answer parser
  data.py      ml-100k download, leave-one-out split, cue-controlled examples
  reward.py    GRPO reward + shortcut telemetry
  sft.py       stage 1: LoRA SFT (assistant-only loss)
  grpo.py      stage 2: merge SFT LoRA, GRPO with KL constraint
  eval.py      letter-logprob ranking: HR@1/NDCG@5, pop_lift, position probe
```

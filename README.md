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

## Python interface

The CLI entrypoints are thin wrappers; everything is importable for custom
experiments (`uv sync` installs `llm4rec` editable).

### Prompts and answer parsing (`llm4rec.prompts`)

```python
from llm4rec.prompts import build_prompt, parse_choice

messages = build_prompt(
    history=["Fargo (1996)", "Groundhog Day (1993)"],   # oldest -> newest
    candidates=["Titanic (1997)", "Vertigo (1958)"],     # letters A, B, ...
    pop_quantiles=[0.98, 0.61],                          # per-candidate popularity in [0,1]
    framing="neutral",                                   # or "evaluative" -> popularity markers
)  # -> [{"role": "system", ...}, {"role": "user", ...}]

parse_choice("Answer: B", num_candidates=2)   # -> 1
parse_choice("Based on history, A", 2)        # -> None (invalid, not a bare letter)
```

### Dataset rows (`data/*.jsonl`)

```python
import json

row = json.loads(open("data/train.jsonl").readline())
row["prompt"]         # chat messages (list of dicts)
row["target"]         # ground-truth candidate index (int)
row["answer"]         # same as a letter, e.g. "B" (SFT label)
row["candidates"]     # candidate titles, index-aligned with the letters
row["item_ids"]       # MovieLens item ids, index-aligned
row["pop_quantiles"]  # popularity quantile per candidate, index-aligned
```

### Custom rewards for GRPO (`llm4rec.reward`)

`GRPOTrainer` calls the reward function with the completions plus every extra
dataset column as a keyword list. To try a mitigation idea, write a new reward
with the same signature and pass it in `grpo.py`:

```python
from llm4rec.prompts import parse_choice
from llm4rec.reward import choice_reward

def depop_reward(prompts, completions, target=None, pop_quantiles=None, **kw):
    """Example mitigation: subtract a popularity-tracking penalty."""
    base = choice_reward(prompts, completions, target=target,
                         pop_quantiles=pop_quantiles, **kw)
    out = []
    for r, comp, quants in zip(base, completions, pop_quantiles):
        text = comp if isinstance(comp, str) else comp[-1]["content"]
        c = parse_choice(text, len(quants))
        lift = 0.0 if c is None else quants[c] - sum(quants) / len(quants)
        out.append(r - 0.5 * max(lift, 0.0))
    return out

# in grpo.py: GRPOTrainer(model=..., reward_funcs=depop_reward, ...)
```

### Programmatic evaluation (`llm4rec.eval`)

```python
import json
from llm4rec.eval import load_model, evaluate, score_letters

tok, model = load_model("Qwen/Qwen2.5-0.5B-Instruct",
                        adapter="runs/sft/final", device="mps")
rows = [json.loads(l) for l in open("data/test.jsonl")][:200]

report = evaluate(tok, model, "mps", rows, position_probe_n=20)
report["hr@1"], report["pop_lift"], report["position_probe"]["spread"]

# or score one prompt directly: log-prob of each candidate letter
scores = score_letters(tok, model, "mps", rows[0]["prompt"],
                       n=len(rows[0]["candidates"]))   # np.ndarray, argmax = choice
```

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

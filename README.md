# llm4rec-bias

A minimal, locally-runnable LLM4Rec pipeline on MovieLens-100K for studying
**RL shortcut / bias mitigation** in recommendation. Scaled-down version of the
RL-Shortcut-Lab execution spec (MLLMRec-R1 route): LoRA SFT → LoRA merge →
GRPO with KL constraint, with the four bias-cue dimensions controllable in the
data layer and measured by dedicated probes.

Runs on an Apple-silicon Mac (tested: M3, 16 GB) with `Qwen/Qwen2.5-0.5B-Instruct`.

Two task routes are implemented:

| | Route 1: letter choice | Route 2: semantic-ID generative retrieval (recommended) |
|---|---|---|
| Task | pick among 10 lettered candidates | generate the next item's semantic ID; scored against the **full catalog** |
| Spec analogue | MLLMRec-R1 (discriminative form) | MiniOneRec |
| Item identity | letters A–J per prompt | global `<s0_i><s1_j><s2_k><s3_c>` codes; similar movies share prefixes |
| Files | `data / sft / grpo / eval / reward` | `semid / sid_data / sid_sft / sid_grpo / sid_eval / sid_reward` |

## Route 1: letter choice

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

## Route 2: semantic-ID generative retrieval

Fixes the weakness of per-prompt letter mapping: item identity lives in
**global semantic-ID tokens**, and retrieval runs against the whole catalog
instead of 10 candidates.

**Semantic IDs** (`semid.py`): each item's text ("Title (year). Genres: ...")
is embedded with frozen MiniLM, then residual k-means (3 levels × 64 codes)
quantizes the embeddings; a 4th level breaks collisions. Similar movies share
leading codes (e.g. *Toy Story* and *A Goofy Movie* share a 2-level prefix).
The 196 code tokens are added to the tokenizer; only those embedding rows are
trained (peft `trainable_token_indices`) alongside the usual LoRA adapters.

**Task**: history (titles + sids) → generate the next item's sid.
**Reward** (`sid_reward.py`): 1.0 exact item; else `0.1 ×` matching leading
levels (semantic-closeness credit — itself a researchable shortcut, disable
with `--prefix-credit 0`); −0.5 invalid ID. Telemetry per GRPO step:
`shortcut/invalid_rate`, `shortcut/pop_lift`, `shortcut/prefix_depth`.
**Eval** (`sid_eval.py`): constrained beam search over the trie of valid IDs →
top-K catalog ranking → HR@1/HR@10/NDCG@10, `pop_lift@1`, plus unconstrained
generation validity.

```bash
uv run python -m llm4rec.semid    --out data          # build semantic_ids.json
uv run python -m llm4rec.sid_data --out data          # sid_{train,val,test}.jsonl + item_meta.json
uv run python -m llm4rec.sid_sft  --out runs/sid_sft  # stage 1 (~40 min on M3)
uv run python -m llm4rec.sid_eval --adapter runs/sid_sft/final --max-examples 300 --out runs/eval_sid_sft.json
uv run python -m llm4rec.sid_grpo --sft-adapter runs/sid_sft/final --steps 300 --out runs/sid_grpo
uv run python -m llm4rec.sid_eval --adapter runs/sid_grpo/final --max-examples 300 --out runs/eval_sid_grpo.json
```

Bias-cue notes for this route: the position cue disappears (no candidate
list); popularity bias is measured on *generated* items vs the catalog mean;
the semantic-prior cue becomes first-class — `shortcut/prefix_depth` tracks
whether GRPO learns to farm prefix credit (right neighborhood, wrong movie)
instead of exact retrieval.

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

## Multimodal extension (planned): poster input via Gemma 4

Not implemented yet — this documents the prompt formats for extending the lab
to multimodal input (the spec's MLLMRec-R1 route). Target model:
`google/gemma-4-E2B-it` (multimodal, fits 16 GB Apple silicon; HF
license-gated). trl ≥ 1.8 supports VLMs in both `SFTTrainer` and `GRPOTrainer`
natively via an `"images"` dataset column. MovieLens-100K ships no images;
posters come from community item-id → TMDb mappings.

Example user (real test row; comedy-leaning mid-90s watcher, ground truth
**F. The Birdcage (1996)**).

### Route 0 — text-only (current)

```
Movies this user watched recently (oldest to newest):
- Phenomenon (1996)
- That Thing You Do! (1996)
...
Candidates:
A. Jane Eyre (1996)
...
F. Birdcage, The (1996)
...
Which candidate will the user watch next? Answer with only the letter.
```

### Route A — offline image-to-text (spec-faithful, recommended first)

One-time captioning pass per poster (any local VLM, cached):

```
[user]  <poster image: The Birdcage (1996)>
        Describe this movie poster in one sentence: visual style, tone,
        and what genre it signals. Do not name the movie.

[assistant]  Bright pink-and-white poster with two smiling middle-aged men
             in a tropical art-deco setting, signaling a lighthearted
             mainstream comedy.
```

The recommendation prompt stays text-only (same policy model as now), with
captions interleaved:

```
Candidates (with poster descriptions):
A. Jane Eyre (1996) — muted period portrait of a woman in Victorian
   dress, somber romantic drama tone
B. Tales from the Crypt Presents: Bordello of Blood (1996) — lurid
   red horror-comedy art with a leering vampire figure
...
F. The Birdcage (1996) — bright pink-and-white art-deco comedy
   poster with two smiling middle-aged men
...
Which candidate will the user watch next? Answer with only the letter.
```

### Route B — end-to-end pixels into Gemma 4

The dataset row carries actual images; chat messages use structured content
parts (the format trl's VLM path consumes):

```python
{
  "images": [PIL.Image, ...],                     # 10 posters, order = A..J
  "prompt": [
    {"role": "system", "content": "You are a movie recommender. ..."},
    {"role": "user", "content": [
        {"type": "text",  "text": "Movies this user watched recently:\n- Phenomenon (1996)\n...\n\nCandidates:"},
        {"type": "text",  "text": "A. Jane Eyre (1996)"},
        {"type": "image"},                        # poster A
        {"type": "text",  "text": "B. Tales from the Crypt Presents: Bordello of Blood (1996)"},
        {"type": "image"},                        # poster B
        # ... C through J ...
        {"type": "text",  "text": "Which candidate will the user watch next? Answer with only the letter."}
    ]}
  ]
}
```

Cost note: each image expands to ~256 tokens, so this prompt is ~3K tokens vs
~350 for Route A — the compute price of true multimodality (slower GRPO
rollouts on MPS).

### The probe this buys: visual salience (fifth cue dimension)

Hold every title fixed and swap only the posters between two candidates:

```
A. Jane Eyre (1996)        + [poster of The Birdcage]     <- swapped
...
F. The Birdcage (1996)     + [poster of Jane Eyre]        <- swapped
```

A content-driven model still picks F; a model shortcutting on visual
attractiveness follows the flashy poster to A. Same permutation logic as the
position probe, applied to pixels. In Route A the analogous test swaps caption
lines — cheaper, and it isolates whether the shortcut lives in visual features
or merely in the evaluative language describing them.

## Layout

```
src/llm4rec/
  # Route 1: letter choice
  prompts.py    templates + framing cue + answer parser
  data.py       ml-100k download, leave-one-out split, cue-controlled examples
  reward.py     GRPO reward + shortcut telemetry
  sft.py        stage 1: LoRA SFT (assistant-only loss)
  grpo.py       stage 2: merge SFT LoRA, GRPO with KL constraint
  eval.py       letter-logprob ranking: HR@1/NDCG@5, pop_lift, position probe
  # Route 2: semantic-ID generative retrieval
  semid.py      MiniLM embeddings -> residual k-means -> sid tokens + trie
  sid_model.py  tokenizer/model setup with sid tokens (mean-init rows)
  sid_data.py   history -> next-sid dataset + item_meta.json
  sid_sft.py    stage 1: LoRA + trainable sid token rows
  sid_reward.py exact/prefix-credit/invalid reward + telemetry
  sid_grpo.py   stage 2: GRPO on merged SFT weights
  sid_eval.py   constrained beam search: HR@K/NDCG@K over full catalog
```

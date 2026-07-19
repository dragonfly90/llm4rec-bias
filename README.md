# llm4rec-bias

A minimal, locally-runnable LLM4Rec pipeline on MovieLens-100K for studying
**RL shortcut / bias mitigation** in recommendation. Scaled-down version of the
[RL-Shortcut-Lab execution spec](https://rl-shortcut-lab.myflorey111.chatgpt.site/zh/literature)
(MLLMRec-R1 route): LoRA SFT → LoRA merge →
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

**Vocabulary sizes** (measured):

| what | size |
|---|---|
| Qwen2.5-0.5B tokenizer (base) | 151,665 tokens |
| semantic-ID code tokens added | 196 (= 3 levels × 64 codes + 4 collision-breakers) |
| tokenizer after adding | 151,861 |
| model embedding rows | 151,936 (Qwen ships slack, so **no resize needed** — sid tokens occupy reserved rows) |
| item catalog | 1,682 movies, each exactly 4 sid tokens (64³ = 262,144 addressable IDs) |
| distinct tokens actually used in the sid dataset (prompts + answers) | 2,584 |

The effective *output* vocabulary of the task is just the 196 code tokens (plus
EOS): every answer is 4 codes, and constrained decoding walks a trie whose
root branches over at most 64 level-0 tokens.

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

**Results — SFT vs GRPO** (300 test users, full-catalog retrieval over 1,682
items; chance HR@10 = 0.6%. GRPO: 300 steps, prefix-credit reward, β=0.04):

| metric | SFT (2 ep) | GRPO (300 steps) | reading |
|---|---|---|---|
| HR@1 | 1.3% | **1.7%** | RL improved exactly what the reward pays: top-1 exact match (+30% rel.) |
| HR@10 | **7.7%** | 6.7% | ranking quality *below* rank 1 slipped — the reward never sees it |
| NDCG@10 | 0.039 | 0.036 | same story |
| free-gen validity | 94% | **100%** | the −0.5 invalid penalty worked completely |
| pop_lift@1 | +0.48 | +0.48 | KL held the popularity profile frozen: RL neither amplified nor mitigated; the **+0.21 excess lift persists** |

The GRPO row is a mild but clean **proxy-narrowing** result: training reward
rose (−0.19 → −0.07) while held-out HR@10 fell — a positive *hacking gap* in
the refined-metrics sense. The reward pays top-1 exact match and validity;
the policy delivered precisely those two and nothing else.

**Mitigation run #1 — and the invalidity escape hatch.** 300 GRPO steps with
`--reward minionerec --pop-weight 0.5` (rank-aware penalty + popularity
penalty, invalid at the original −0.5):

| metric | SFT | GRPO (prefix) | GRPO (minionerec + pop, v1) |
|---|---|---|---|
| HR@1 | 1.3% | 1.7% | **0.3%** |
| HR@10 | 7.7% | 6.7% | 6.7% |
| pop_lift@1 | +0.48 | +0.48 | **+0.42** |
| free-gen validity | 94% | 100% | **64%** |

The popularity penalty *did* bite (+0.48 → +0.42, removing ~⅓ of the excess
lift) — but the run mostly discovered a **new shortcut**: with rank penalties
reaching −1.0 and the pop penalty stacked on top, a *wrong valid* answer cost
up to ≈ −1.25 while an *invalid* one cost only −0.5, so the policy learned to
hide in garbage output (train invalid_rate 0.40 → 0.70; KL 3× the vanilla
run). A mitigation reward changed the action ordering and the policy exited
through the cheapest door — the exact dynamic this lab exists to observe.
MiniOneRec never faces this because constrained-beam rollouts make invalid
output impossible; free-sampling variants must keep invalid **strictly
dominated** (fixed: invalid_penalty now defaults to −1.5).

**Mitigation run #2 — hatch closed, policy pays the tax.** Same reward
(`minionerec + pop-weight 0.5`) with invalid at −1.5. Training-side (300
steps): the fix works — invalid_rate now *falls* 0.36 → 0.21 (v1: rose to
0.70), reward climbs −0.93 → −0.75, KL moderate (0.08, vs 0.28 in v1). But
the popularity pressure went nowhere: rollout `pop_lift` held at ~0.43 and
the per-step pop penalty (`penalty/pop_mean`) *drifted up* 0.26 → 0.34. With
the invalidity door sealed, the policy chose to **pay the popularity tax
rather than diversify** — popular guesses still earn enough exact-hit reward
to be worth −0.5 × lift. Eval side confirms it — the four-run ladder:

| metric | SFT | GRPO (prefix) | +pop v1 (open hatch) | +pop v2 (w=0.5) | +pop w=1.0 |
|---|---|---|---|---|---|
| HR@1 | 1.3% | **1.7%** | 0.3% | 1.0% | **1.7%** |
| HR@10 | **7.7%** | 6.7% | 6.7% | 6.7% | 7.3% |
| NDCG@10 | 0.039 | 0.036 | 0.029 | 0.033 | 0.038 |
| pop_lift@1 | +0.48 | +0.48 | +0.42 | +0.48 | **+0.46** |
| free-gen validity | 94% | 100% | 64% | 100% | 100% |

Sweep reading:
- **w=0.5 repriced but did not reroute** — greedy-decode popularity identical
  to baseline (+0.479); the policy absorbed the tax and kept the
  popular-guess strategy (v1's apparent lift reduction was purchased with the
  validity collapse, not diversification).
- **w=1.0 is the best RL checkpoint overall and the first genuine (small)
  reroute**: ties the best HR@1, best GRPO-stage HR@10 (7.3% — the rank-aware
  penalty defending ranking depth, as MiniOneRec intends), 100% validity, and
  a real −0.02 lift reduction (+0.48 → +0.46, removing ~12% of the +0.21
  excess) at **zero accuracy cost** — a Pareto improvement over w=0.5.
- The dose–response is nonlinear: 0.5 buys nothing, 1.0 starts to bite.
  Next arms: `--pop-weight 2.0`, and the *wrong-only* penalty (tax popular
  guesses only when they miss), which breaks the tax-vs-hit-rate tradeoff
  instead of shifting it.

**What `pop_lift@1` means.** Every movie gets a popularity quantile in [0,1]
(ranked by training interaction count: 0 = least-watched, 1 = most-watched,
0.5 = median). `pop_lift@1` is the mean quantile of the model's rank-1
retrievals minus the catalog mean (≈ 0.5). Scale: −0.5 = only retrieves the
most obscure items, **0 = popularity-neutral**, +0.5 = only retrieves the most
popular. Our +0.48 means top-1 picks average ~0.98 quantile — the model almost
exclusively retrieves the top few percent most-popular movies. Two caveats:
(1) some lift is legitimate — popular movies genuinely are watched next more
often, and the held-out targets themselves average above 0.5, so the research
question is how much lift *exceeds* what held-out data justifies and whether
RL inflates it (that's why the same quantity is logged per GRPO step as
`shortcut/pop_lift`); (2) in the letter route the analogous metric subtracts
the *candidate-set* mean instead of the catalog mean, since the model can only
choose among the 10 shown items.

Bias-cue notes for this route: the position cue disappears (no candidate
list); popularity bias is measured on *generated* items vs the catalog mean;
the semantic-prior cue becomes first-class — `shortcut/prefix_depth` tracks
whether GRPO learns to farm prefix credit (right neighborhood, wrong movie)
instead of exact retrieval.

## Metrics reference

Every metric in the project, where it is computed, and what it tells you.

**Retrieval / task quality** (evaluation scripts):

| metric | where | definition |
|---|---|---|
| HR@1 / HR@10 | `sid_eval` (constrained beam over full catalog), `eval` (letter log-probs over 10 candidates) | target in top-1 / top-K |
| NDCG@5 / NDCG@10 | same | 1/log2(rank+2) if target ranked, else 0 |
| free-gen validity | `sid_eval` | unconstrained greedy generation emits a real catalog ID |
| eval loss / token accuracy | SFT logs | per-token quality on held-out answers |

**Bias / shortcut — implemented**:

| metric | where | cue | definition |
|---|---|---|---|
| `pop_lift@1` | `sid_eval` | popularity | popularity quantile of top-1 retrieval − catalog mean (0.50); justified level from held-out targets = +0.27 |
| `pop_lift` | `eval` (letter) | popularity | chosen item's quantile − candidate-set mean (exposure-matched, so justified ≈ 0) |
| `shortcut/pop_lift` | GRPO logs, per step | popularity | same quantity on training rollouts |
| position-probe curve + `spread` | `eval --position-probe` | position | accuracy with the target re-placed at every slot, content fixed; spread = max − min (0 = position-blind, 1 = pure position policy) |
| `chosen_pos_hist`, `shortcut/chosen_pos_mean` | `eval` / GRPO logs | position | marginal distribution of chosen slots |
| `shortcut/invalid_rate` | GRPO logs, per step | format | fraction of rollouts that parse to no valid ID (the metric that exposed both RL bugs) |
| `shortcut/prefix_depth` | GRPO logs, per step | semantic prior | matching leading code levels between generation and target (chance = 0.025); rising depth with stalling hits = prefix-credit farming |
| `penalty/pop_mean`, `reward/rank_penalty_mean` | GRPO logs | — | magnitudes of the active reward components |
| `kl`, `reward`, `frac_reward_zero_std` | GRPO logs | — | policy drift from reference; training reward; fraction of zero-gradient groups (pinned at 1.0 = no learning) |
| hacking gap | computed from logs + checkpoint evals | proxy–true divergence | Δ(training reward) − Δ(held-out HR@10) per phase; measured: +0.12 reward vs −1.0pp HR@10 on vanilla GRPO |

**Bias / shortcut — planned** (documented in the refinement table below):
ΔGAP (user-anchored pop lift) · IPS-corrected HR/NDCG · per-tier HR
(head/mid/tail) · exposure Gini + aggregate diversity · feedback-loop
amplification curve · reward–cue correlation · primacy–recency asymmetry ·
framing gap (paired neutral/evaluative eval) · history reversal gap ·
permutation flip rate · representation probes R1–R6 (linear probing, CKA
drift, activation intervention).

### Metric → paper provenance

Where each metric comes from — the [RL-Shortcut-Lab spec](https://rl-shortcut-lab.myflorey111.chatgpt.site/zh/literature),
the [curated paper list](https://docs.google.com/document/d/1ovjbt635409rSpyq3FBChWpblxwLujOLdM3YtXMXT1w/) (#N = its numbering),
or the method papers:

| metric | source |
|---|---|
| HR@K, NDCG@K | standard IR/rec metrics; used as preference targets by the lab spec and MiniOneRec ([arXiv:2510.24431](https://arxiv.org/abs/2510.24431)) |
| popularity lift (`pop_lift`, `pop_lift@1`) | lab spec popularity cues; rooted in #3 *A Study of Popularity Bias* ([arXiv:2406.01285](https://arxiv.org/abs/2406.01285), ARP/GAP family) |
| ΔGAP (user-anchored lift) | #3 *A Study of Popularity Bias* ([arXiv:2406.01285](https://arxiv.org/abs/2406.01285)) — GAP compares recommendation popularity to the user's own profile |
| head/mid/tail share, per-tier HR | #8 *Revealing Potential Biases … Cold Start* ([arXiv:2508.20401](https://arxiv.org/abs/2508.20401), segment-wise evaluation) |
| IPS-corrected HR/NDCG | #6 *Mitigating Propensity Bias of LLMs for RecSys* ([arXiv:2409.20052](https://arxiv.org/abs/2409.20052)); #12 *ReCRec* ([ACM TOIS](https://doi.org/10.1145/3672275), exposure-aware debiased evaluation) |
| exposure Gini, aggregate diversity / coverage | #11 *Modeling and Counteracting Exposure Bias* ([arXiv:2001.04832](https://arxiv.org/abs/2001.04832)); #10 *Feedback Loop and Bias Amplification* ([arXiv:2007.13019](https://arxiv.org/abs/2007.13019)) |
| feedback-loop amplification curve | #13 *Echoes in the Loop* ([arXiv:2602.07442](https://arxiv.org/abs/2602.07442), LLM rec loops); #10 ([arXiv:2007.13019](https://arxiv.org/abs/2007.13019), simulation methodology) |
| position-probe curve, `spread`, position-conditioned selection | lab spec position cues (permutation swaps, position-conditioned selection rate) |
| permutation flip rate, Kendall/Spearman consistency | lab spec position metrics (flip rate selected, consistency dropped) |
| primacy–recency asymmetry | #9 *Cognitive Biases in LLMs for News Recommendation* ([arXiv:2410.02897](https://arxiv.org/abs/2410.02897)) |
| framing gap (neutral vs evaluative), paraphrase consistency | lab spec textual-framing metrics; #9 ([arXiv:2410.02897](https://arxiv.org/abs/2410.02897)) for the cognitive-bias framing |
| history reversal gap, recent-window concentration | lab spec recency metrics |
| `invalid_rate` | lab spec per-step deliverable; MiniOneRec ([arXiv:2510.24431](https://arxiv.org/abs/2510.24431)) motivates the constrained-decoding contrast |
| `prefix_depth` (semantic-neighborhood tracking) | this repo, instantiating the lab's semantic-prior cue on TIGER/MiniOneRec-style hierarchical SIDs; semantic-bias framing per #4 ([arXiv:2601.09478](https://arxiv.org/abs/2601.09478)), #7 *LLM-RecG* ([arXiv:2501.19232](https://arxiv.org/abs/2501.19232)) |
| rank-aware penalty (`reward/rank_penalty_mean`) | MiniOneRec hybrid reward ([arXiv:2510.24431](https://arxiv.org/abs/2510.24431)) |
| popularity penalty (`penalty/pop_mean`) | reward-side mitigation per #5 *SPLiT* ([OpenReview](https://openreview.net/forum?id=M36IXztHLF), no arXiv); #6 ([arXiv:2409.20052](https://arxiv.org/abs/2409.20052), propensity correction as training signal) |
| hacking gap | #15 *Correlated Proxies* ([arXiv:2403.03185](https://arxiv.org/abs/2403.03185), hacking = proxy–true divergence); #14 *ODIN* ([arXiv:2402.07319](https://arxiv.org/abs/2402.07319)) |
| reward–cue correlation | #14 *ODIN* ([arXiv:2402.07319](https://arxiv.org/abs/2402.07319), reward vs length proxy, transplanted to popularity/prefix cues); #15 ([arXiv:2403.03185](https://arxiv.org/abs/2403.03185)) |
| `kl`, group-normalized reward, `frac_reward_zero_std` | GRPO method (DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) as packaged by trl; KL-as-mitigation per the lab spec |
| representation probes R1–R6 (probing, CKA drift, subspace estimation, activation intervention) | lab spec representation section; #1, #2 (attention-hacking / shortcut rectification in reward models) motivate the representation-level diagnosis |

#1 and #2 are otherwise out of scope: this lab's rewards are rule-based, so
there is no learned reward model to hack — the analogous surface here is
reward *parsing* (see the skip_special_tokens incident).

## Dataset-side cue baselines (measured)

Each probe compares a model metric against what the *data* justifies. These
are the measured baselines (ml-100k, default generation settings):

| Cue | Probe / metric | Dataset-side value | Interpretation |
|---|---|---|---|
| Popularity (sid route) | `pop_lift` vs catalog mean (0.50) | held-out targets average quantile **0.77** → justified lift **+0.27** | SFT model's +0.48 ⇒ **+0.21 excess lift** beyond user behavior — the quantified popularity bias |
| Popularity (letter route) | `pop_lift` vs candidate-set mean | pop-sampled negatives average **0.83** ≈ targets (0.77–0.84) → justified lift **−0.05 ≈ 0** | candidate sets are exposure-matched, so any positive lift is pure shortcut — a clean detector |
| Position (letter route) | target-position histogram; probe `spread` | placement near-uniform: counts 319–399 across slots A–J (max/min 1.25) | data carries **no position signal**; the base model's spread = 1.0 is 100% model prior |
| Text framing (letter route) | neutral vs evaluative A/B | evaluative would mark **73.7%** of candidates "(popular hit)", 1.4% "(rarely watched)" | with pop-sampled negatives the marker is nearly non-discriminative — use `--neg-sampling uniform` for a sharp framing experiment |
| History recency | `--history N` variants | histories saturate the cap (~8–9 shown, min 5) | recency experiments need regenerated datasets (e.g. N=2 vs N=8), not post-hoc analysis |
| Semantic prior (sid route) | `shortcut/prefix_depth` on wrong answers | random item pair shares **0.025** levels on average | wrong-answer depth ≫ 0.03 = right-neighborhood learning; rising depth with stalling exact hits = prefix-credit farming (the route's signature reward hack) |
| Invalid rate | `shortcut/invalid_rate` | n/a (all training answers valid by construction) | model-side references: 94% valid free-gen after SFT; 100% under constrained decoding |

## Selected metrics per bias (following the RL-Shortcut-Lab representation section)

Metric selection based on the lab's five bias families and representation
methods R1–R6 ([RL-Shortcut-Lab literature: representation](https://rl-shortcut-lab.myflorey111.chatgpt.site/zh/literature#representation)).
Each bias gets one cheap screening metric plus one representation probe for
the causal stage:

| Bias | Selected behavioral metric | Why this one | Selected representation method | Status |
|---|---|---|---|---|
| Popularity | popularity lift (quantile form) + head/mid/tail share + long-tail coverage@10 | lift is the headline number (+0.48 vs +0.27 justified); share/coverage catch tail collapse that lift can hide | R1 probing: linear probe decoding item popularity from the hidden state at the answer position, across base→SFT→GRPO checkpoints | lift ✅; share/coverage ➕ planned in `sid_eval` |
| Position (letter route) | permutation flip rate + position-probe `spread` | flip rate (does the chosen *item* change when candidates are shuffled?) is the cleanest causal signal; Kendall-τ adds little beyond it for K=10 lists | R6 activation intervention: project out the position-decodable direction, re-measure spread | spread ✅; flip rate ➕ planned in `eval` |
| Repetition / exposure | exposure calibration: KL between popularity histogram of top-1 retrievals and of held-out targets | data dedupes consecutive repeats, so repeat-count lift is structurally absent; calibration subsumes excess lift into a distribution-level check | R3 shortcut-subspace: variance of answer logits explained by a popularity direction | ➕ planned in `sid_eval` |
| Recency | history reversal gap: ΔHR@10 + prediction flip rate under reversed history order | content-identical, order-only manipulation → causal reading; recent-window concentration comes free via sid prefix overlap with last-k vs earlier history | R4 geometry: prefix depth of prediction vs history position | ➕ planned eval flag, no retraining |
| Textual framing | neutral-vs-evaluative gap (paired A/B on identical examples) | the direct instrument; note the measured caveat — markers only discriminate on `--neg-sampling uniform` data (73.7% marker saturation otherwise) | R1 probing: framing-marker decodability from candidate representations | framing flag ✅; paired eval ➕ |

Cross-cutting for causal → mitigation: **R2 (CKA drift)** across the four
spec checkpoints (base / SFT / GRPO-mid / GRPO-final) screens *where* RL moved
representations; **R6 (scale/project the identified subspace)** is the
mechanism-guided mitigation benchmarked against the generic KL-strength sweep
under equal budget — the spec's core comparison.

Deliberately not selected: ARP (redundant with quantile lift),
Kendall/Spearman consistency (subsumed by flip rate at K=10), repeat-count
lift (absent from deduplicated data), temporal calibration (ml-100k too small
for clean timestamped eval windows).

### Metric refinements from the reward-hacking / bias paper list

Refinements to the selection above, drawn from the
[curated paper list](https://docs.google.com/document/d/1ovjbt635409rSpyq3FBChWpblxwLujOLdM3YtXMXT1w/)
(15 papers: popularity/propensity/semantic bias in LLM recommenders, feedback
loops, exposure bias, RLHF reward hacking — incl. *A Study of Popularity
Bias*, *SPLIT*, *Mitigating Propensity Bias*, *ReCRec*, *Echoes in the Loop*,
*ODIN*, *Correlated Proxies*):

| Refinement | Definition | Replaces / augments | Source idea | Status |
|---|---|---|---|---|
| **User-anchored popularity lift (ΔGAP)** | pop(top-1 retrieval) − mean pop(that user's own history), averaged over users | catalog-mean `pop_lift` — ΔGAP separates "model over-popularizes" from "this user genuinely likes popular items"; a per-user justified baseline instead of one global +0.27 | *LLMs as Recommender Systems: A Study of Popularity Bias* (GAP metrics) | ➕ needs `history_items` column in `sid_data` |
| **IPS-corrected HR@K / NDCG@K** | weight each test hit by inverse propensity ∝ 1/pop(target)^γ (self-normalized) | raw HR/NDCG, which reward popular-guessing because test targets are themselves popular (0.77 mean quantile) — IPS makes tail hits count more, so the metric can't be farmed by popularity | *Mitigating Propensity Bias of LLMs for RecSys*; *ReCRec* | ➕ easy add to `sid_eval` |
| **Per-tier HR (head/mid/tail)** | HR@10 computed separately for targets in top/mid/bottom popularity tiers | single aggregate HR — a model can score 7.7% overall with literally 0% on tail targets; the tier split exposes it | cold-start bias paper (segment-wise evaluation) | ➕ easy add to `sid_eval` |
| **Exposure Gini + aggregate diversity** | Gini coefficient of item exposure counts across all users' top-K, plus % of catalog ever retrieved | long-tail coverage@10 alone — Gini captures *concentration* among the items that do get exposed | *Modeling and Counteracting Exposure Bias*; *Feedback Loop and Bias Amplification* | ➕ easy add to `sid_eval` |
| **Feedback-loop amplification curve** | simulate T loop iterations (append top-1 retrieval to history, re-retrieve); plot pop_lift / Gini vs T | all static metrics — bias that looks mild in one shot can compound in the loop; LLM rec loops shown to collapse diversity | *Echoes in the Loop*; *Feedback Loop and Bias Amplification* | ➕ planned (new script, no retraining) |
| **Hacking gap** | Δ(training reward) − Δ(held-out HR@10), per checkpoint segment | eyeballing reward vs HR curves — makes "reward up, utility flat" a single reportable number per training phase | *Correlated Proxies* (hacking = proxy–true divergence); ODIN | ✅ **measured**: vanilla GRPO reward +0.12 while HR@10 −1.0pp (positive gap = proxy narrowing); v1 pop run reward flat while validity −36pp (gap via new shortcut) |
| **Reward–cue correlation** | per-step Pearson r between sample reward and cue value (popularity of generated item; prefix depth) | threshold-watching on `shortcut/*` — rising r(reward, cue) is the early-warning signal that the policy is monetizing the cue, before HR moves | ODIN (disentangling reward from length proxy, transplanted to popularity/prefix proxies) | ➕ add inside reward funcs via `log_metric`; the v1/v2 runs show why it's needed — `pop_lift` alone couldn't distinguish repricing from rerouting |
| **Primacy–recency asymmetry** (letter route) | acc(first ⅓ of slots) − acc(last ⅓) from the position-probe curve | scalar `spread` — the base model showed A-and-J concentration, i.e. *both* primacy and recency effects; the asymmetry says which dominates | *Cognitive Biases in LLMs for News Recommendation* | ➕ trivial add to `eval` position probe |

**Primacy–recency asymmetry, expanded.** The serial-position effect from
cognitive psychology, transplanted to LLM candidate lists: *primacy* =
over-selecting early slots (anchoring on the first items read), *list-recency*
= over-selecting late slots (closest to the generation position; a distinct
mechanism — attention sinks vs context proximity — so mitigations may fix one
end and not the other). Signed: positive → primacy dominates, negative →
recency dominates. It complements `spread` rather than replacing it; the two
together classify the curve's shape:

| `spread` | asymmetry | reading |
|---|---|---|
| ~0 | ~0 | position-blind (the goal) |
| high | strongly + | pure primacy policy ("always pick A") |
| high | strongly − | pure list-recency policy ("always pick the last item") |
| high | ~0 | U-shaped: both ends favored, middle ignored |

Our zero-shot baseline is the instructive case: choices split ~70% slot A /
~30% slot J — spread 1.0, asymmetry ~+0.4 (positive but far from ceiling).
A pure-primacy story would be wrong; the model exhibits the full **U-shaped
serial-position curve** of the psychology literature. Letter route only — the
sid route has no candidate list.

Reward-model-side papers on the list (attention hacking, shortcut
rectification in preference-based reward learning) are noted but out of scope:
this lab uses rule-based rewards, so there is no learned RM to hack — the
analogous failure surface here is the *reward-parsing* path (see the
skip_special_tokens incident above).

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

### Setting the reward function

**Interface.** `GRPOTrainer` accepts any callable via `reward_funcs`. Per
batch of rollouts it calls it with the completions plus **every extra column
of the training JSONL as a batch-aligned keyword list** — that's how rewards
receive ground truth (`target_item` here; `target`/`pop_quantiles` in the
letter route). Return one float per completion. trl also injects
`log_metric(name, value)`: anything you log lands in the per-step training
logs next to reward and KL (all `shortcut/*` telemetry works this way).
GRPO normalizes rewards within each group of `num_generations` samples of the
same prompt, so no value network is involved.

```python
def my_reward(prompts, completions, target_item=None, log_metric=None, **kw):
    ...                    # completions[k] is a str (or chat-message list)
    return rewards         # list[float], one per completion
```

**Design of the built-in rewards.** Sid route (`sid_reward.py`):

| outcome | reward | role |
|---|---|---|
| exact target item | 1.0 | the actual objective |
| wrong item, k matching leading codes | 0.1 × k (≤ 0.3) | shaping: gradient signal while exact hits are rare (sparse 0/1 over 1,682 items leaves most groups all-zero) |
| unparseable / unknown ID | −0.5 | prices format collapse |

Two constraints set the numbers: shaping must stay well below the exact
reward or the policy optimizes the proxy, and the penalty must be modest or
the policy collapses to low-entropy conservative output. The letter route
(`reward.py`) is the degenerate version: +1 / 0 / −0.5, no shaping (chance is
already 10%).

**MiniOneRec reward + popularity tuning.** `sid_reward.py` also implements
the MiniOneRec hybrid reward ([arXiv:2510.24431](https://arxiv.org/html/2510.24431v1))
and a popularity penalty, selected from the `sid_grpo` CLI:

```bash
# MiniOneRec hybrid: binary rule reward + rank-aware hard-negative penalty
uv run python -m llm4rec.sid_grpo --reward minionerec ...

# combine with popularity tuning (second reward function, weighted 0.5)
uv run python -m llm4rec.sid_grpo --reward minionerec --pop-weight 0.5 ...
```

- `make_minionerec_reward`: exact hit → 1.0; wrong valid item →
  `-mag/Σmag` with `mag = 1/log(rank+1)`, penalties summing to −1 per GRPO
  group. The paper ranks wrong items by constrained-beam position; with trl's
  sampled rollouts we rank by **frequency within the group** — a Monte-Carlo
  confidence estimate, so the most *confidently* wrong item is punished
  hardest. Invalid → −0.5 (the paper has no invalid case since it decodes
  with constrained beams; we sample freely and keep invalid rate measurable).
- `make_pop_penalty`: `-max(pop_lift, 0)` per completion — penalizes
  retrieving above-catalog-mean-popularity items even when correct,
  repricing the popular-guess strategy. Added as a second `reward_funcs`
  entry with its own `reward_weights` coefficient; sweep `--pop-weight` to
  trade HR@10 against `pop_lift` (the mitigation experiment for the +0.21
  excess lift measured after SFT). Logged separately as `penalty/pop_mean`.

**Changing it.** `--prefix-credit 0.05` scales the shaping, `0` disables it —
a planned experiment, since the credit is itself a shortcut incentive
(`shortcut/prefix_depth` rising while exact hits stall = neighborhood
farming). For custom rewards, write the same signature and swap it into
`GRPOTrainer(reward_funcs=...)` in `sid_grpo.py` (see `depop_reward` above
for a mitigation example). trl also accepts a *list* of reward functions and
sums them (`reward_weights` in `GRPOConfig`) — cleaner than wrapping when you
want accuracy and a bias penalty logged as separate curves.

**Pitfalls (paid for in this repo).**
- The reward sees decoded text, and **trl decodes rollouts with
  `skip_special_tokens=True`** — our first GRPO run burned 3 h at reward −0.5
  because special-flagged sid tokens were stripped before parsing. Verify
  your parser on actual rollout decodings, not constructed strings.
- Use one parser everywhere: reward and eval share `parse_sid`/`parse_choice`
  so "valid" can't diverge between training and evaluation.
- Watch `frac_reward_zero_std`: a group with identical rewards contributes
  zero gradient; pinned at 1.0 = no learning (the canary that caught the bug).
- Whatever you reward is what you get: the reward checks only the final
  answer, so popularity/prefix shortcuts stay open — that's the object of
  study, and the telemetry exists to catch it.

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

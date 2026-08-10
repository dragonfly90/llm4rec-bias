"""Rewards for GRPO on the semantic-ID route.

make_sid_reward       prefix-credit shaping: 1.0 exact / 0.1 per matching
                      leading code level / -0.5 invalid. The prefix credit is
                      itself a researchable cue (neighborhood farming);
                      disable with prefix_credit=0.
make_minionerec_reward  MiniOneRec hybrid (https://arxiv.org/abs/2510.24431): binary rule
                      reward + rank-aware hard-negative penalty. The paper
                      ranks by beam position; with trl's sampled rollouts we
                      rank wrong items by their frequency within the GRPO
                      group (a Monte-Carlo confidence estimate): the most
                      confidently-wrong item gets the steepest penalty
                      -1/log(rank+1), normalized to sum to -1 per group.
make_pop_penalty      popularity tuning: -max(pop_lift, 0) per completion,
                      meant as a second entry in reward_funcs with its own
                      weight (GRPOConfig.reward_weights).
make_sda_reward       Semantic Distribution Alignment: reward = log P*(i) -
                      log Q_theta(i), the log of the spec's importance ratio,
                      where P* comes from the transition model in
                      sid_transition.py. Not a hand-mixed heuristic — it is the
                      policy-gradient reward of D_KL(P* || Q_theta).

Telemetry: invalid rate, popularity lift of generated items vs catalog mean,
mean matched-prefix depth (all logged per step).
"""

import json
import math
from collections import Counter

import numpy as np

from .semid import SidTable, parse_sid


def _text(completion) -> str:
    return completion if isinstance(completion, str) else completion[-1]["content"]


def _log_shortcuts(log_metric, items, target_item, table, meta, catalog_pop_mean):
    if log_metric is None:
        return
    invalid = sum(i is None for i in items)
    log_metric("shortcut/invalid_rate", invalid / len(items))
    lifts = [meta[i]["pop_quantile"] - catalog_pop_mean for i in items if i is not None]
    if lifts:
        log_metric("shortcut/pop_lift", float(np.mean(lifts)))
    depths = []
    for i, t in zip(items, target_item):
        if i is None:
            continue
        d = 0
        for a, b in zip(table.codes[i], table.codes[t]):
            if a != b:
                break
            d += 1
        depths.append(d)
    if depths:
        log_metric("shortcut/prefix_depth", float(np.mean(depths)))


def make_sid_reward(sid_table_path: str, item_meta_path: str,
                    prefix_credit: float = 0.1, invalid_penalty: float = -0.5):
    table = SidTable(sid_table_path)
    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
    catalog_pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))

    def sid_reward(prompts, completions, target_item=None, log_metric=None, **kwargs):
        items = [table.parse(_text(c)) for c in completions]
        rewards = []
        for k, item in enumerate(items):
            if item is None:
                rewards.append(invalid_penalty)
            elif item == target_item[k]:
                rewards.append(1.0)
            else:
                depth = 0
                for a, b in zip(table.codes[item], table.codes[target_item[k]]):
                    if a != b:
                        break
                    depth += 1
                rewards.append(prefix_credit * depth)
        _log_shortcuts(log_metric, items, target_item, table, meta, catalog_pop_mean)
        return rewards

    return sid_reward


def make_minionerec_reward(sid_table_path: str, item_meta_path: str,
                           num_generations: int, invalid_penalty: float = -1.5):
    """MiniOneRec hybrid reward: R = R_rule + R_rank (frequency-rank variant).

    Paper: https://arxiv.org/abs/2510.24431

    Within each GRPO group (num_generations completions of one prompt):
      exact hit          -> 1.0
      wrong valid item   -> -mag / sum(mags) over the group's wrong items,
                            mag = 1/log(rank+1), rank 1 = the wrong item
                            generated most often in the group (most confident)
      invalid            -> invalid_penalty (paper has no invalid case: it
                            decodes with constrained beams; we sample freely)

    invalid_penalty defaults to -1.5: it must be strictly worse than the worst
    valid outcome (rank penalty bottoms at -1.0, plus any weighted pop penalty
    ~-0.25). At -0.5 the policy learns to hide in invalidity — measured: a
    300-step run drove invalid_rate from 0.40 to 0.70 because garbage was
    cheaper than being confidently wrong.
    """
    table = SidTable(sid_table_path)
    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
    catalog_pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))

    def minionerec_reward(prompts, completions, target_item=None, log_metric=None, **kwargs):
        n = len(completions)
        items = [table.parse(_text(c)) for c in completions]
        rewards = [0.0] * n
        rank_pens = []
        for g in range(0, n, num_generations):
            idx = list(range(g, min(g + num_generations, n)))
            wrong = [items[i] for i in idx
                     if items[i] is not None and items[i] != target_item[i]]
            order = [it for it, _ in Counter(wrong).most_common()]
            mag = {it: 1.0 / math.log(r + 2) for r, it in enumerate(order)}  # rank 1 -> 1/log2
            denom = sum(mag.values())
            for i in idx:
                item = items[i]
                if item is None:
                    rewards[i] = invalid_penalty
                elif item == target_item[i]:
                    rewards[i] = 1.0
                else:
                    rewards[i] = -mag[item] / denom
                    rank_pens.append(rewards[i])
        if log_metric is not None and rank_pens:
            log_metric("reward/rank_penalty_mean", float(np.mean(rank_pens)))
        _log_shortcuts(log_metric, items, target_item, table, meta, catalog_pop_mean)
        return rewards

    return minionerec_reward


def make_pop_penalty(sid_table_path: str, item_meta_path: str,
                     anchor: str = "catalog", wrong_only: bool = False):
    """Popularity tuning: -max(q(item) - baseline, 0) per completion.

    anchor="catalog": baseline = catalog-mean popularity (~0.5), a uniform tax.
      Measured to reprice but not reroute — it taxes popular recommendations
      equally for every user, so it fights the exact-hit reward on the ~77% of
      users whose true next item is genuinely popular.
    anchor="user": baseline = this user's own history-popularity mean
      (hist_pop_mean, a dataset column). This is a direct gradient on the ΔGAP
      metric (https://arxiv.org/abs/2406.01285): recommending a blockbuster to a
      blockbuster-lover costs ~0, while an over-popular pick for a niche user is
      penalized hard. Concentrates the pressure where the bias actually lives
      without raising the global weight.
    wrong_only=True: apply the penalty only when the generated item != target.
      A correct retrieval is, by definition, the right popularity for that user,
      so it is never taxed — breaking the tax-vs-hit-rate tradeoff instead of
      shifting it.

    Use as a second entry in reward_funcs with its own GRPOConfig.reward_weights.
    """
    table = SidTable(sid_table_path)
    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
    catalog_pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))

    def pop_penalty(prompts, completions, target_item=None, hist_pop_mean=None,
                    log_metric=None, **kwargs):
        if anchor == "user" and hist_pop_mean is None:
            raise ValueError("anchor='user' needs the hist_pop_mean dataset column "
                             "(regenerate data with the updated sid_data.py)")
        rewards = []
        for k, c in enumerate(completions):
            item = table.parse(_text(c))
            if item is None:
                rewards.append(0.0)  # invalidity is priced by the main reward
                continue
            if wrong_only and item == target_item[k]:
                rewards.append(0.0)
                continue
            baseline = hist_pop_mean[k] if anchor == "user" else catalog_pop_mean
            rewards.append(-max(meta[item]["pop_quantile"] - baseline, 0.0))
        if log_metric is not None:
            log_metric("penalty/pop_mean", float(-np.mean(rewards)))
        return rewards

    return pop_penalty


class PolicyScorer:
    """log Q_theta(item | history) under the *live* GRPO policy.

    The SDA reward needs the policy's own probability for the sampled semantic
    ID. trl hands reward functions only text, so we score it ourselves: append
    the item's canonical SID tokens to the prompt and read the teacher-forced
    log-probabilities off one extra forward pass (batch of 16 short sequences —
    a few percent on top of a ~12 s GRPO step).

    Scoring the *canonical* tokens rather than the sampled string makes Q an
    item-level distribution, directly comparable with the item-level P*, and
    independent of any junk the policy emitted around the ID.

    The model reference is set after GRPOTrainer builds the PEFT policy, so
    Q_theta always tracks the current parameters (on-policy, as the ratio
    P*/Q_theta in the spec assumes).
    """

    def __init__(self, tok, table, device="mps"):
        self.tok, self.table, self.device = tok, table, device
        self.model = None
        self.levels = table.levels
        self.level1_ids = [tok.convert_tokens_to_ids(f"<s0_{c}>") for c in range(table.K)]
        self._prompt_cache = {}

    def set_model(self, model):
        self.model = model

    def _prompt_ids(self, messages):
        key = messages[-1]["content"]
        if key not in self._prompt_cache:
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=True)
            # transformers 5 returns a BatchEncoding (possibly batched) here,
            # older versions a bare list of ids
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            self._prompt_cache[key] = list(ids)
        return self._prompt_cache[key]

    def __call__(self, prompts, items):
        """-> (log Q per row, per-level breakdown (B, levels), Q(C1) per row).

        Rows whose item is None are nan / None (the reward prices invalidity on
        its own).
        """
        import torch

        n = len(items)
        logq = np.full(n, np.nan)
        levels = np.full((n, self.levels), np.nan)
        q1 = [None] * n
        valid = [k for k, it in enumerate(items) if it is not None]
        if not valid or self.model is None:
            return logq, levels, q1

        seqs = []
        for k in valid:
            sid_ids = [self.tok.convert_tokens_to_ids(f"<s{l}_{c}>")
                       for l, c in enumerate(self.table.codes[items[k]])]
            seqs.append(self._prompt_ids(prompts[k]) + sid_ids)
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        # left padding keeps the scored SID tokens at fixed offsets from the end
        input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs],
                                 device=self.device)
        attn = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs],
                            device=self.device)
        L = self.levels
        with torch.no_grad():
            # logits_to_keep is not an optimization detail: the full-sequence
            # logits are batch x ~360 x 152k (~1.7 GB in bf16) and thrash a 16 GB
            # Mac, dragging GRPO step time from 11 s to 29 s. We need L+1
            # positions: the generation prompt plus the first L-1 SID tokens.
            logits = self.model(input_ids=input_ids, attention_mask=attn,
                                logits_to_keep=L + 1).logits
            lp = torch.log_softmax(logits[:, -(L + 1):-1, :].float(), dim=-1)
            tgt = input_ids[:, -L:]
            per_level = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)   # (V, levels)
            # Q(C1): the level-1 marginal, renormalized over the 64 code tokens
            first = torch.softmax(logits[:, -(L + 1), :].float(), dim=-1)[:, self.level1_ids]
            first = first / first.sum(-1, keepdim=True).clamp(min=1e-12)
        per_level = per_level.cpu().numpy()
        first = first.cpu().numpy()
        for j, k in enumerate(valid):
            levels[k] = per_level[j]
            logq[k] = per_level[j].sum()
            q1[k] = first[j]
        return logq, levels, q1


def make_sda_reward(sid_table_path: str, item_meta_path: str, transition,
                    policy_scorer: PolicyScorer, clip: float = 4.0,
                    invalid_penalty: float | None = None, hit_weight: float = 0.0):
    """Semantic Distribution Alignment reward (llm4rec-bias-Integrated issue #2).

        R_SDA(s) = P*(s) / Q_theta(s | H_t)

    is the importance ratio whose policy gradient is exactly -grad of
    D_KL(P* || Q_theta): rollouts the policy *under*-produces relative to the
    user's next-step target distribution get R > 1, over-produced ones R < 1. We
    reward the **log** ratio, clipped to +-clip:

        R(s) = clip(log P*(s) - log Q_theta(s), -clip, +clip)

    Two reasons. (1) GRPO standardizes rewards inside each group, so only the
    ordering and spread survive anyway — and the raw ratio's variance is the
    spec's own listed limitation (small Q_theta blows it up). (2) A bounded
    reward is what lets invalid output stay *strictly dominated*: this repo
    previously measured a policy escaping into garbage when a penalty stack made
    wrong-but-valid cost more than unparseable. invalid_penalty therefore
    defaults to -(clip + 1).

    Unlike every other reward here, this one never looks at target_item: the
    supervision is a distribution, not the single held-out answer. That is the
    bias-resistance claim — the policy is pulled toward the user's whole
    next-step semantic neighborhood instead of toward one popular point. Set
    hit_weight > 0 for a hybrid that adds back an explicit exact-hit bonus.

    Telemetry: sda/log_ratio_mean, sda/logp_mean, sda/logq_mean,
    sda/frac_clipped, sda/D1 (exact KL(P*(C1) || Q(C1)), the coarse-grained
    mismatch of spec section 10), and sda/gap_l{1..} (per-level chain-rule
    mismatch at the sampled codes).
    """
    table = SidTable(sid_table_path)
    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
    catalog_pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))
    inv = -(clip + 1.0) if invalid_penalty is None else invalid_penalty

    def sda_reward(prompts, completions, target_item=None, history_items=None,
                   log_metric=None, **kwargs):
        if history_items is None:
            raise ValueError("SDA needs the history_items dataset column "
                             "(regenerate data with the updated sid_data.py)")
        items = [table.parse(_text(c)) for c in completions]
        log_p, p_levels = transition.log_p_items(history_items, items, per_level=True)
        log_q, q_levels, q1 = policy_scorer(prompts, items)

        rewards, clipped, ratios = [], 0, []
        for k, item in enumerate(items):
            if item is None or not np.isfinite(log_q[k]):
                rewards.append(inv)
                continue
            ratio = float(log_p[k] - log_q[k])
            ratios.append(ratio)
            r = float(np.clip(ratio, -clip, clip))
            clipped += abs(ratio) > clip
            if hit_weight and item == target_item[k]:
                r += hit_weight
            rewards.append(r)

        if log_metric is not None:
            if ratios:
                log_metric("sda/log_ratio_mean", float(np.mean(ratios)))
                log_metric("sda/log_ratio_std", float(np.std(ratios)))
                log_metric("sda/frac_clipped", clipped / len(ratios))
                log_metric("sda/logp_mean", float(np.nanmean(log_p)))
                log_metric("sda/logq_mean", float(np.nanmean(log_q)))
                for l in range(p_levels.shape[1] - 1):
                    gap = p_levels[:, l] - q_levels[:, l]
                    if np.isfinite(gap).any():
                        log_metric(f"sda/gap_l{l + 1}", float(np.nanmean(gap)))
            # D1 = KL(P*(C1) || Q(C1)), computed once per distinct prompt
            seen, d1 = set(), []
            for k, q in enumerate(q1):
                key = tuple(history_items[k])
                if q is None or key in seen:
                    continue
                seen.add(key)
                p = transition.level1_probs([history_items[k]])[0].cpu().numpy()
                d1.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
            if d1:
                log_metric("sda/D1", float(np.mean(d1)))
        _log_shortcuts(log_metric, items, target_item, table, meta, catalog_pop_mean)
        return rewards

    return sda_reward


def make_rare_hit_bonus(sid_table_path: str, item_meta_path: str, gamma: float = 0.5):
    """Propensity-weighted exact-hit bonus: + 1/max(count,1)^gamma when correct.

    The reward-side mirror of the IPS-corrected HR metric. The main reward pays a
    flat +1.0 for any exact hit, so the policy maximizes it by farming popular
    targets (measured: hr_by_tier head 10.8% / mid 0% / tail 0%). This bonus adds
    almost nothing for a correct blockbuster (count~490 -> +0.045) but a large
    amount for a correct rare item (count 1 -> +1.0), so within a GRPO group the
    advantage points toward retrieving rare targets correctly — lifting mid/tail
    HR and hr_ips@K. Wrong / invalid completions get 0 (their cost stays with the
    main reward). gamma matches the IPS metric's propensity exponent; 0.5 (1/√count)
    is a gentler default than the metric's 1.0 to limit reward variance.

    Use as a second entry in reward_funcs with its own GRPOConfig.reward_weights.
    https://arxiv.org/abs/2409.20052  https://arxiv.org/abs/2508.20401
    """
    table = SidTable(sid_table_path)
    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}

    def rare_hit_bonus(prompts, completions, target_item=None, log_metric=None, **kwargs):
        rewards = []
        for k, c in enumerate(completions):
            item = table.parse(_text(c))
            if item is not None and item == target_item[k]:
                rewards.append(1.0 / max(meta[item].get("count", 1), 1) ** gamma)
            else:
                rewards.append(0.0)
        if log_metric is not None:
            log_metric("bonus/rare_hit_mean", float(np.mean(rewards)))
        return rewards

    return rare_hit_bonus

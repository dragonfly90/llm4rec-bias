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

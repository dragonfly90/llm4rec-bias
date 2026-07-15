"""Reward for GRPO on the semantic-ID route.

Reward: 1.0 exact item; otherwise 0.1 per matching leading code level
(semantic closeness credit, capped below exact); -0.5 unparseable/unknown ID.
The prefix credit is itself a researchable cue: it pays the policy for landing
in the right semantic neighborhood, which can become a shortcut. Disable with
prefix_credit=0.

Telemetry: invalid rate, popularity lift of generated items vs catalog mean,
mean matched-prefix depth.
"""

import json

import numpy as np

from .semid import SidTable, parse_sid


def make_sid_reward(sid_table_path: str, item_meta_path: str,
                    prefix_credit: float = 0.1, invalid_penalty: float = -0.5):
    table = SidTable(sid_table_path)
    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
    catalog_pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))

    def sid_reward(prompts, completions, target_item=None, log_metric=None, **kwargs):
        rewards, pop_lifts, prefix_depths, invalid = [], [], [], 0
        for k, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            codes = parse_sid(text, table.levels)
            item = table.item_of.get(codes) if codes else None
            if item is None:
                rewards.append(invalid_penalty)
                invalid += 1
                continue
            if item == target_item[k]:
                rewards.append(1.0)
                prefix_depths.append(table.levels)
            else:
                tgt = table.codes[target_item[k]]
                depth = 0
                for a, b in zip(codes, tgt):
                    if a != b:
                        break
                    depth += 1
                rewards.append(prefix_credit * depth)
                prefix_depths.append(depth)
            pop_lifts.append(meta[item]["pop_quantile"] - catalog_pop_mean)

        if log_metric is not None:
            log_metric("shortcut/invalid_rate", invalid / len(completions))
            if pop_lifts:
                log_metric("shortcut/pop_lift", float(np.mean(pop_lifts)))
            if prefix_depths:
                log_metric("shortcut/prefix_depth", float(np.mean(prefix_depths)))
        return rewards

    return sid_reward

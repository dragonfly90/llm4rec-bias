"""Reward function and shortcut-metric hooks for GRPO."""

import numpy as np

from .prompts import parse_choice

REWARD_HIT = 1.0
REWARD_MISS = 0.0
REWARD_INVALID = -0.5


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    # chat format: [{"role": "assistant", "content": ...}]
    return completion[-1]["content"]


def choice_reward(prompts, completions, target=None, pop_quantiles=None,
                  log_metric=None, **kwargs):
    """+1 correct candidate, 0 wrong-but-valid, -0.5 unparseable.

    Also logs shortcut telemetry: invalid rate, mean chosen position (normalized),
    popularity lift of chosen items vs candidate-set mean.
    """
    rewards, positions, pop_lifts, invalid = [], [], [], 0
    for k, completion in enumerate(completions):
        text = _completion_text(completion)
        quants = pop_quantiles[k]
        n = len(quants)
        choice = parse_choice(text, n)
        if choice is None:
            rewards.append(REWARD_INVALID)
            invalid += 1
            continue
        rewards.append(REWARD_HIT if choice == target[k] else REWARD_MISS)
        positions.append(choice / max(n - 1, 1))
        pop_lifts.append(quants[choice] - float(np.mean(quants)))

    if log_metric is not None:
        log_metric("shortcut/invalid_rate", invalid / len(completions))
        if positions:
            log_metric("shortcut/chosen_pos_mean", float(np.mean(positions)))
        if pop_lifts:
            log_metric("shortcut/pop_lift", float(np.mean(pop_lifts)))
    return rewards

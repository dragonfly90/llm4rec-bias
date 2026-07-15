"""Generative-retrieval dataset: history -> semantic ID of the next item.

No candidate list: the model must generate the target's semantic ID and is
scored against the full catalog (constrained beam search at eval time).

Row: {prompt (chat messages), answer (sid string), target_item, pop_quantile}
Also writes data/item_meta.json (title + popularity quantile per item) used by
the reward and eval telemetry.
"""

import argparse
import json
import random
from pathlib import Path

from .data import download_ml100k, load_interactions, popularity_stats
from .semid import SidTable

SYSTEM_PROMPT = (
    "You are a movie recommender. Every movie has a semantic ID made of codes "
    "like <s0_12><s1_45><s2_7><s3_0>; similar movies share leading codes. "
    "Given a user's watch history, respond with only the semantic ID of the "
    "movie they will watch next."
)


def build_example(user, hist_items, target, titles, table, popq,
                  history_len, with_titles):
    hist = [i for i in hist_items if i in titles][-history_len:]
    if len(hist) < 2 or target not in titles:
        return None
    if with_titles:
        lines = [f"- {titles[i]} {table.sid(i)}" for i in hist]
    else:
        lines = [f"- {table.sid(i)}" for i in hist]
    user_msg = (
        "Movies this user watched recently (oldest to newest):\n"
        + "\n".join(lines)
        + "\n\nSemantic ID of the next movie:"
    )
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_msg}],
        "answer": table.sid(target),
        "target_item": target,
        "pop_quantile": popq.get(target, 0.5),
        "user": user,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--train-per-user", type=int, default=4)
    ap.add_argument("--sid-only", action="store_true",
                    help="drop titles from history (pure semantic-ID prompts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    raw = download_ml100k(out)
    seqs, titles = load_interactions(raw)
    counts, popq = popularity_stats(seqs)
    table = SidTable(args.sid_table)

    json.dump({int(i): {"title": titles.get(i, ""), "pop_quantile": popq.get(i, 0.5)}
               for i in table.codes},
              open(out / "item_meta.json", "w"))

    splits = {"train": [], "val": [], "test": []}
    for u, s in seqs.items():
        pos_choices = list(range(2, len(s) - 2))
        for t in rng.sample(pos_choices, min(args.train_per_user, len(pos_choices))):
            ex = build_example(u, s[:t], s[t], titles, table, popq,
                               args.history, not args.sid_only)
            if ex:
                splits["train"].append(ex)
        for name, (h, tgt) in {"val": (s[:-2], s[-2]), "test": (s[:-1], s[-1])}.items():
            ex = build_example(u, h, tgt, titles, table, popq,
                               args.history, not args.sid_only)
            if ex:
                splits[name].append(ex)

    for name, rows in splits.items():
        rng.shuffle(rows)
        p = out / f"sid_{name}{args.suffix}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{p}: {len(rows)} examples")


if __name__ == "__main__":
    main()

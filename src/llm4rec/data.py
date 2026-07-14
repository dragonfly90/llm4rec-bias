"""MovieLens-100K -> prompt/choice examples with controllable bias cues.

Leave-one-out split per user: last interaction = test, second-to-last = val,
the rest generate training examples.

Cue controls:
  --neg-sampling pop|uniform   popularity-weighted negatives simulate exposure bias
  --target-pos random|first|last|middle   where the ground truth sits in the list
  --framing neutral|evaluative            popularity markers in candidate text
  --history N                              history length shown (recency cue)

Each JSONL row: {prompt (chat messages), target (index), candidates (titles),
                 item_ids, pop_quantiles, user, framing}
"""

import argparse
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .prompts import build_prompt, LETTERS

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_ml100k(root: Path) -> Path:
    raw = root / "ml-100k"
    if (raw / "u.data").exists():
        return raw
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "ml-100k.zip"
    if not zpath.exists():
        print(f"downloading {ML100K_URL} ...")
        urllib.request.urlretrieve(ML100K_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(root)
    return raw


def load_interactions(raw: Path, min_rating: int = 4):
    """Positive interactions (rating >= min_rating), per-user chronological order."""
    titles = {}
    with open(raw / "u.item", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            titles[int(parts[0])] = parts[1]

    events = []
    with open(raw / "u.data") as f:
        for line in f:
            u, i, r, t = line.split("\t")
            if int(r) >= min_rating:
                events.append((int(u), int(i), int(t)))
    events.sort(key=lambda e: (e[0], e[2]))

    seqs = defaultdict(list)
    for u, i, _ in events:
        if not seqs[u] or seqs[u][-1] != i:
            seqs[u].append(i)
    seqs = {u: s for u, s in seqs.items() if len(s) >= 5}
    return seqs, titles


def popularity_stats(seqs: dict, train_cut: int = 2):
    """Item interaction counts on the training region only (avoid test leakage)."""
    counts = defaultdict(int)
    for s in seqs.values():
        for i in s[:-train_cut]:
            counts[i] += 1
    items = sorted(counts)
    ranks = {i: r for r, i in enumerate(sorted(items, key=lambda x: counts[x]))}
    n = len(items)
    quantile = {i: ranks[i] / max(n - 1, 1) for i in items}
    return counts, quantile


class ExampleBuilder:
    def __init__(self, titles, counts, quantile, num_candidates, history_len,
                 neg_sampling, target_pos, framing, rng):
        self.titles = titles
        self.counts = counts
        self.quantile = quantile
        self.C = num_candidates
        self.H = history_len
        self.neg_sampling = neg_sampling
        self.target_pos = target_pos
        self.framing = framing
        self.rng = rng
        self.pool = [i for i in counts if i in titles]
        w = np.array([counts[i] for i in self.pool], dtype=np.float64)
        self.pop_w = w / w.sum()

    def sample_negatives(self, exclude: set, k: int) -> list[int]:
        negs, seen = [], set(exclude)
        while len(negs) < k:
            if self.neg_sampling == "pop":
                batch = list(np.random.default_rng(self.rng.randrange(2**31)).choice(
                    self.pool, size=k * 2, p=self.pop_w))
            else:
                batch = self.rng.sample(self.pool, min(k * 2, len(self.pool)))
            for i in batch:
                i = int(i)
                if i not in seen:
                    seen.add(i)
                    negs.append(i)
                    if len(negs) == k:
                        break
        return negs

    def place_target(self, target: int, negs: list[int]) -> tuple[list[int], int]:
        if self.target_pos == "first":
            pos = 0
        elif self.target_pos == "last":
            pos = self.C - 1
        elif self.target_pos == "middle":
            pos = self.C // 2
        else:
            pos = self.rng.randrange(self.C)
        cands = negs[:pos] + [target] + negs[pos:]
        return cands, pos

    def build(self, user: int, hist_items: list[int], target: int) -> dict | None:
        hist_items = [i for i in hist_items if i in self.titles][-self.H:]
        if len(hist_items) < 2 or target not in self.titles:
            return None
        negs = self.sample_negatives(set(hist_items) | {target}, self.C - 1)
        cands, pos = self.place_target(target, negs)
        titles = [self.titles[i] for i in cands]
        quants = [self.quantile.get(i, 0.5) for i in cands]
        prompt = build_prompt([self.titles[i] for i in hist_items], titles, quants, self.framing)
        return {
            "prompt": prompt,
            "target": pos,
            "answer": LETTERS[pos],
            "candidates": titles,
            "item_ids": cands,
            "pop_quantiles": quants,
            "user": user,
            "framing": self.framing,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--num-candidates", type=int, default=10)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--neg-sampling", choices=["pop", "uniform"], default="pop")
    ap.add_argument("--target-pos", choices=["random", "first", "last", "middle"], default="random")
    ap.add_argument("--framing", choices=["neutral", "evaluative"], default="neutral")
    ap.add_argument("--train-per-user", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="tag for output filenames, e.g. '_uniform'")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    raw = download_ml100k(out)
    seqs, titles = load_interactions(raw)
    counts, quantile = popularity_stats(seqs)
    print(f"{len(seqs)} users, {len(counts)} items with training interactions")

    b = ExampleBuilder(titles, counts, quantile, args.num_candidates, args.history,
                       args.neg_sampling, args.target_pos, args.framing, rng)

    splits = {"train": [], "val": [], "test": []}
    for u, s in seqs.items():
        # training positions: random cut points strictly before the val item
        pos_choices = list(range(2, len(s) - 2))
        for t in rng.sample(pos_choices, min(args.train_per_user, len(pos_choices))):
            ex = b.build(u, s[:t], s[t])
            if ex:
                splits["train"].append(ex)
        val_ex = b.build(u, s[:-2], s[-2])
        test_ex = b.build(u, s[:-1], s[-1])
        if val_ex:
            splits["val"].append(val_ex)
        if test_ex:
            splits["test"].append(test_ex)

    for name, rows in splits.items():
        rng.shuffle(rows)
        p = out / f"{name}{args.suffix}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{p}: {len(rows)} examples")


if __name__ == "__main__":
    main()

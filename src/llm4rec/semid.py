"""Semantic IDs for generative retrieval (MiniOneRec route).

Pipeline: item text (title + genres) -> frozen MiniLM embedding -> residual
k-means (L levels x K codes) -> collision-breaking final level. Each item
becomes a token sequence like <s0_12><s1_45><s2_7><s3_0>: similar movies share
code prefixes, so the ID itself carries semantics.

Also provides the token/trie utilities used by SFT, GRPO and constrained
beam-search eval.

CLI: uv run python -m llm4rec.semid --out data   -> data/semantic_ids.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import download_ml100k, load_interactions, popularity_stats

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

SID_RE = re.compile(r"<s(\d+)_(\d+)>")


# ---------------- embedding ----------------

def item_texts(raw: Path) -> dict[int, str]:
    texts = {}
    with open(raw / "u.item", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            flags = [GENRES[i] for i, v in enumerate(p[5:24]) if v == "1"]
            texts[int(p[0])] = f"{p[1]}. Genres: {', '.join(flags) or 'unknown'}."
    return texts


@torch.no_grad()
def embed(texts: list[str], device: str, batch: int = 64) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (h * mask).sum(1) / mask.sum(1)  # mean pooling
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out)


# ---------------- residual k-means ----------------

def kmeans(X: np.ndarray, K: int, iters: int, rng) -> tuple[np.ndarray, np.ndarray]:
    C = X[rng.choice(len(X), size=K, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        for k in range(K):
            m = assign == k
            C[k] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
    return C, assign


def residual_quantize(X: np.ndarray, levels: int, K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    res = X.copy()
    codes = np.zeros((len(X), levels), dtype=int)
    for level in range(levels):
        C, a = kmeans(res, K, iters=50, rng=rng)
        codes[:, level] = a
        res = res - C[a]
    return codes


def break_collisions(codes: np.ndarray) -> np.ndarray:
    """Append one disambiguation level so every item's ID is unique."""
    groups = defaultdict(list)
    for i, c in enumerate(map(tuple, codes)):
        groups[c].append(i)
    extra = np.zeros(len(codes), dtype=int)
    for members in groups.values():
        for j, i in enumerate(members):
            extra[i] = j
    return np.concatenate([codes, extra[:, None]], axis=1)


# ---------------- tokens / parsing / trie ----------------

def sid_string(codes) -> str:
    return "".join(f"<s{l}_{c}>" for l, c in enumerate(codes))


def all_sid_tokens(levels: int, K: int, collision_K: int) -> list[str]:
    toks = [f"<s{l}_{c}>" for l in range(levels) for c in range(K)]
    toks += [f"<s{levels}_{c}>" for c in range(collision_K)]
    return toks


def parse_sid(text: str, num_levels: int) -> tuple | None:
    """First num_levels well-ordered <sL_C> tokens in text -> codes tuple."""
    hits = SID_RE.findall(text)
    codes = []
    for level, code in hits:
        if int(level) == len(codes):
            codes.append(int(code))
            if len(codes) == num_levels:
                return tuple(codes)
        else:
            break
    return None


class SidTable:
    """item_id <-> semantic ID, plus trie for constrained decoding."""

    def __init__(self, path: str):
        d = json.load(open(path))
        self.levels = d["levels"] + 1  # + collision level
        self.K = d["K"]
        self.collision_K = d["collision_K"]
        self.codes = {int(i): tuple(c) for i, c in d["items"].items()}
        self.item_of = {c: i for i, c in self.codes.items()}

    def sid(self, item_id: int) -> str:
        return sid_string(self.codes[item_id])

    def tokens(self) -> list[str]:
        return all_sid_tokens(self.levels - 1, self.K, self.collision_K)

    def parse(self, text: str) -> int | None:
        codes = parse_sid(text, self.levels)
        return self.item_of.get(codes) if codes else None

    def trie(self, tokenizer, eos_id: int) -> dict:
        """Token-id trie over all valid IDs, terminated by eos."""
        root = {}
        for codes in self.codes.values():
            ids = [tokenizer.convert_tokens_to_ids(f"<s{l}_{c}>")
                   for l, c in enumerate(codes)]
            node = root
            for t in ids:
                node = node.setdefault(t, {})
            node[eos_id] = {}
        return root

    def prefix_fn(self, tokenizer, prompt_len: int, eos_id: int):
        root = self.trie(tokenizer, eos_id)
        def fn(batch_id, input_ids):
            node = root
            for t in input_ids[prompt_len:].tolist():
                node = node.get(t)
                if node is None:
                    return [eos_id]
            return list(node.keys()) or [eos_id]
        return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--codes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw = download_ml100k(Path(args.out))
    texts = item_texts(raw)
    ids = sorted(texts)
    print(f"embedding {len(ids)} items with {ENCODER} ...")
    X = embed([texts[i] for i in ids], device)

    codes = residual_quantize(X, args.levels, args.codes, args.seed)
    codes = break_collisions(codes)
    n_coll = int(codes[:, -1].max()) + 1
    uniq = len({tuple(c) for c in codes})
    assert uniq == len(ids), "IDs not unique after collision breaking"
    print(f"levels={args.levels}+1, K={args.codes}, max collision group={n_coll}")

    out = Path(args.out) / "semantic_ids.json"
    json.dump({"levels": args.levels, "K": args.codes, "collision_K": n_coll,
               "encoder": ENCODER,
               "items": {int(i): [int(x) for x in c] for i, c in zip(ids, codes)}},
              open(out, "w"))
    print(f"wrote {out}")

    # sanity: nearest neighbors share prefixes?
    seqs, titles = load_interactions(raw)
    counts, _ = popularity_stats(seqs)
    ex = ids[0]
    same = [i for i, c in zip(ids, codes) if tuple(c[:2]) == tuple(codes[0][:2]) and i != ex]
    print(f"shares 2-level prefix with '{texts[ex][:40]}':",
          [texts[i][:40] for i in same[:3]])


if __name__ == "__main__":
    main()

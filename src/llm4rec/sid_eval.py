"""Full-catalog evaluation for the semantic-ID route.

Constrained beam search over the trie of all valid semantic IDs gives a
top-K item ranking against the entire catalog (~1.4K items) -> HR@K, NDCG@K.
Also reports unconstrained top-1 validity (does free generation produce a
real ID?) and several popularity/exposure bias metrics:

  pop_lift@1   q(top-1) - catalog mean (global baseline)
  delta_gap    q(top-1) - the user's own history-popularity mean, averaged
               over users (ΔGAP; per-user baseline, arXiv:2406.01285). Needs
               the hist_pop_mean dataset column.
  exposure_gini  Gini of item exposure counts across all users' top-K, over
                 the whole catalog incl. never-retrieved items (0 = uniform
                 exposure, 1 = all exposure on one item; arXiv:2001.04832,
                 2007.13019).
  coverage@K   fraction of the catalog that appears in at least one top-K.
  hr_ips@K,    HR/NDCG re-weighted by inverse target propensity
  ndcg_ips@K   (w = 1/max(count,1)^gamma, self-normalized) so tail hits count
               more — a popularity-farming policy can't inflate them
               (arXiv:2409.20052, 3672275).
  hr_by_tier   HR@K split by the target's popularity tier (head/mid/tail =
               top/mid/bottom third of the quantile range; arXiv:2508.20401).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from .semid import SidTable
from .sid_model import prepare


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative exposure vector (zeros included)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    # efficient form of the double-sum definition; i = 1..n over ascending x
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1) / n)


@torch.no_grad()
def beam_retrieve(tok, model, device, table, messages, k: int):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = (enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc).to(device)
    eos = tok.eos_token_id
    out = model.generate(
        ids,
        max_new_tokens=table.levels + 1,
        num_beams=k,
        num_return_sequences=k,
        prefix_allowed_tokens_fn=table.prefix_fn(tok, ids.shape[1], eos),
        early_stopping=True,
        do_sample=False,
        pad_token_id=eos,
    )
    items = []
    for seq in out:
        text = tok.decode(seq[ids.shape[1]:], skip_special_tokens=False)
        item = table.parse(text)
        if item is not None and item not in items:
            items.append(item)
    return items


@torch.no_grad()
def free_top1_valid(tok, model, device, table, messages) -> bool:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = (enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc).to(device)
    out = model.generate(ids, max_new_tokens=table.levels + 2, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    return table.parse(text) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="", help="LoRA adapter ('' = base + fresh sid tokens)")
    ap.add_argument("--sft-adapter", default="",
                    help="adapter to merge BEFORE --adapter; required for GRPO "
                         "checkpoints, which are trained on merged-SFT weights")
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--item-meta", default="data/item_meta.json")
    ap.add_argument("--data", default="data/sid_test.jsonl")
    ap.add_argument("--max-examples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--free-gen-n", type=int, default=50,
                    help="check unconstrained validity on N examples")
    ap.add_argument("--ips-gamma", type=float, default=1.0,
                    help="propensity exponent for IPS-corrected HR/NDCG: "
                         "weight = 1/max(count,1)^gamma (0 = uncorrected)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    table = SidTable(args.sid_table)
    tok, model, _ = prepare(args.model, table)
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()

    meta = {int(i): v for i, v in json.load(open(args.item_meta)).items()}
    pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))
    rows = [json.loads(l) for l in open(args.data)][:args.max_examples]

    from collections import Counter
    hr1, hrk, ndcg, lifts, gaps = [], [], [], [], []
    exposure = Counter()  # item -> times it appears in any user's top-K
    # per-tier HR: bucket by target popularity quantile (bottom/mid/top third)
    tier_hits = {"head": [], "mid": [], "tail": []}
    # IPS-corrected HR/NDCG: inverse-propensity weights over the target
    ips_w, ips_hit_w, ips_ndcg_w = [], [], []
    for r in rows:
        items = beam_retrieve(tok, model, device, table, r["prompt"], args.topk)
        tgt = r["target_item"]
        rank = items.index(tgt) if tgt in items else None
        hit = rank is not None
        dcg = 1.0 / np.log2(rank + 2) if hit else 0.0
        hr1.append(rank == 0)
        hrk.append(hit)
        ndcg.append(dcg)
        exposure.update(items)
        if items:
            q_top1 = meta[items[0]]["pop_quantile"]
            lifts.append(q_top1 - pop_mean)
            if r.get("hist_pop_mean") is not None:  # ΔGAP: per-user baseline
                gaps.append(q_top1 - r["hist_pop_mean"])

        # per-tier HR by the TARGET's popularity (thirds of the quantile range)
        q_tgt = meta[tgt]["pop_quantile"]
        tier = "tail" if q_tgt < 1 / 3 else ("mid" if q_tgt < 2 / 3 else "head")
        tier_hits[tier].append(hit)

        # IPS: rarer targets get higher weight, w = 1/max(count,1)^gamma (self-normalized)
        w = 1.0 / max(meta[tgt].get("count", 1), 1) ** args.ips_gamma
        ips_w.append(w)
        ips_hit_w.append(w * hit)
        ips_ndcg_w.append(w * dcg)

    # exposure over the FULL catalog (never-retrieved items count as 0)
    exposure_vec = np.array([exposure.get(i, 0) for i in table.codes])
    coverage = float((exposure_vec > 0).sum() / len(exposure_vec))
    ips_denom = sum(ips_w)

    valid = [free_top1_valid(tok, model, device, table, r["prompt"])
             for r in rows[:args.free_gen_n]]

    result = {
        "n": len(rows),
        f"hr@1": float(np.mean(hr1)),
        f"hr@{args.topk}": float(np.mean(hrk)),
        f"ndcg@{args.topk}": float(np.mean(ndcg)),
        f"hr_ips@{args.topk}": float(sum(ips_hit_w) / ips_denom) if ips_denom else None,
        f"ndcg_ips@{args.topk}": float(sum(ips_ndcg_w) / ips_denom) if ips_denom else None,
        "hr_by_tier": {t: (float(np.mean(h)) if h else None) for t, h in tier_hits.items()},
        "tier_n": {t: len(h) for t, h in tier_hits.items()},
        "pop_lift@1": float(np.mean(lifts)) if lifts else None,
        "delta_gap": float(np.mean(gaps)) if gaps else None,
        "exposure_gini": gini(exposure_vec),
        f"coverage@{args.topk}": coverage,
        "free_gen_valid_rate": float(np.mean(valid)) if valid else None,
        "ips_gamma": args.ips_gamma,
        "model": args.model, "adapter": args.adapter, "data": args.data,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

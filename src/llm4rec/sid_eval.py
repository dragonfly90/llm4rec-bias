"""Full-catalog evaluation for the semantic-ID route.

Constrained beam search over the trie of all valid semantic IDs gives a
top-K item ranking against the entire catalog (~1.7K items). We report standard
accuracy plus a battery of popularity/exposure bias metrics; each is designed
to reveal something the aggregate HR@K hides. Every "SFT:" value below is the
measured SFT-checkpoint reading on 300 test users, kept here as a sanity anchor.

Accuracy
  hr@1, hr@K    target in the top-1 / top-K beam. Catalog is ~1.7K, so
                chance HR@10 ≈ 0.6%.                                  SFT: 1.3% / 7.7%
  ndcg@K        1/log2(rank+2) if the target ranks in top-K, else 0. SFT: 0.039
  free-gen      unconstrained greedy decode emits a real catalog ID — measures
                whether the ID grammar was actually learned, not just enforced
                by the beam constraint.                              SFT: 94%

Popularity (all use the item popularity quantile q ∈ [0,1]; 0.5 = catalog median)
  pop_lift@1    q(top-1) − catalog mean (≈0.5). Global "how popular are the
                recommendations" signal. +0.5 = only blockbusters, 0 = neutral.
                Confound: real next-watches are popular too, so a positive value
                is partly justified — that's why ΔGAP exists.        SFT: +0.48
  delta_gap     q(top-1) − THIS user's own history-popularity mean, averaged
                over users (ΔGAP). Per-user baseline strips out "the user simply
                likes popular items", leaving the genuinely unjustified excess.
                Needs the hist_pop_mean dataset column.              SFT: +0.19
                https://arxiv.org/abs/2406.01285

Exposure / diversity (over ALL users' top-K pooled together)
  exposure_gini Gini of per-item exposure counts across the whole catalog,
                zeros included (0 = every item shown equally, 1 = all exposure
                on one item). Catches concentration that pop_lift/ΔGAP miss —
                "same few blockbusters for everyone".                SFT: 0.97
                https://arxiv.org/abs/2001.04832 https://arxiv.org/abs/2007.13019
  coverage@K    fraction of the catalog that appears in ≥1 user's top-K.
                The blunt long-tail-reach number.                    SFT: 7.4%

Accuracy-under-debiasing (do the hits survive when you stop rewarding popularity?)
  hr_ips@K,     HR/NDCG re-weighted by inverse target propensity
  ndcg_ips@K    (w = 1/max(count,1)^gamma, self-normalized), so a hit on a rare
                target counts far more than a hit on a blockbuster. A big gap
                below raw HR = accuracy is popularity-farmed.        SFT: 0.6% (vs 7.7%)
                https://arxiv.org/abs/2409.20052 https://doi.org/10.1145/3672275
  hr_by_tier    HR@K computed separately for targets in the top/mid/bottom third
                of the popularity range (head/mid/tail). Exposes tail collapse an
                aggregate HR averages away.        SFT: head 10.8% / mid 0% / tail 0%
                https://arxiv.org/abs/2508.20401
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
    """Gini coefficient of a non-negative exposure vector (zeros included).

    The zeros matter: passing only the exposed items understates concentration,
    since the never-recommended tail is exactly what makes exposure unequal.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:  # empty, or nothing was ever retrieved
        return 0.0
    # O(n log n) equivalent of the mean-absolute-difference definition
    # G = (Σ_i Σ_j |x_i - x_j|) / (2 n Σ x); the sorted form below avoids the n² sum.
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1) / n)


@torch.no_grad()
def beam_retrieve(tok, model, device, table, messages, k: int):
    """Constrained beam search -> up to k distinct catalog items, best-first.

    prefix_allowed_tokens_fn masks the logits to the SID trie at every step, so
    every beam spells a real item; this is what lets the sid route rank against
    the *full* catalog (~1.7K items) instead of a shortlist. Contrast
    free_top1_valid, which does NOT constrain and thus measures whether the
    model learned validity on its own.
    """
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt")
    # transformers 5 returns a BatchEncoding here, not a bare tensor
    ids = (enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc).to(device)
    eos = tok.eos_token_id
    out = model.generate(
        ids,
        max_new_tokens=table.levels + 1,  # L code tokens + EOS
        num_beams=k,
        num_return_sequences=k,           # return the whole beam as the ranked list
        prefix_allowed_tokens_fn=table.prefix_fn(tok, ids.shape[1], eos),
        early_stopping=True,
        do_sample=False,                  # deterministic: eval must be reproducible
        pad_token_id=eos,
    )
    items = []
    for seq in out:                       # beams come back already sorted best-first
        text = tok.decode(seq[ids.shape[1]:], skip_special_tokens=False)  # completion only
        item = table.parse(text)
        if item is not None and item not in items:  # dedup: two beams may map to one item
            items.append(item)
    return items


@torch.no_grad()
def free_top1_valid(tok, model, device, table, messages) -> bool:
    """Unconstrained greedy decode: did the model emit a real ID on its own?

    No trie mask here — this is the validity telemetry (the spec's invalid-rate
    deliverable, viewed at eval time). A high rate means SFT/GRPO taught the ID
    grammar; a low rate means retrieval only works because beam search forces it.
    """
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
    tok, model, _ = prepare(args.model, table)  # also adds the SID tokens to the tokenizer
    # Merge order matters: a GRPO adapter is trained relative to merged-SFT
    # weights, so SFT must be merged FIRST. Loading GRPO onto the raw base
    # silently drops the SFT deltas (incl. the trained SID embeddings) -> garbage.
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()

    meta = {int(i): v for i, v in json.load(open(args.item_meta)).items()}
    pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))  # ~0.5, the pop_lift baseline
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
        rank = items.index(tgt) if tgt in items else None  # None = target not in top-K
        hit = rank is not None
        dcg = 1.0 / np.log2(rank + 2) if hit else 0.0
        hr1.append(rank == 0)
        hrk.append(hit)
        ndcg.append(dcg)
        exposure.update(items)  # tally this user's whole top-K toward global exposure
        if items:  # constrained beam can occasionally return nothing (all beams hit EOS early)
            # popularity of the *rank-1* item drives both lift metrics:
            q_top1 = meta[items[0]]["pop_quantile"]
            lifts.append(q_top1 - pop_mean)                 # pop_lift@1: vs catalog mean
            if r.get("hist_pop_mean") is not None:          # ΔGAP: vs THIS user's history mean
                gaps.append(q_top1 - r["hist_pop_mean"])    # (per-user baseline; needs the column)

        # per-tier HR keys off the TARGET's popularity, not the prediction's:
        # it asks "does the model work for users whose next item is obscure?"
        q_tgt = meta[tgt]["pop_quantile"]
        tier = "tail" if q_tgt < 1 / 3 else ("mid" if q_tgt < 2 / 3 else "head")
        tier_hits[tier].append(hit)

        # IPS: down-weight hits on popular targets so accuracy can't be farmed by
        # popular-guessing. max(count,1) guards div-by-zero for cold (test-only) items;
        # weights are self-normalized below, so any common scale factor cancels.
        w = 1.0 / max(meta[tgt].get("count", 1), 1) ** args.ips_gamma
        ips_w.append(w)
        ips_hit_w.append(w * hit)
        ips_ndcg_w.append(w * dcg)

    # exposure vector spans the FULL catalog in SID-table order; never-retrieved
    # items contribute 0 (essential for Gini — see gini() docstring)
    exposure_vec = np.array([exposure.get(i, 0) for i in table.codes])
    coverage = float((exposure_vec > 0).sum() / len(exposure_vec))
    ips_denom = sum(ips_w)  # self-normalizing denominator (SNIPS)

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

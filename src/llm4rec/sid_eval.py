"""Full-catalog evaluation for the semantic-ID route.

Constrained beam search over the trie of all valid semantic IDs gives a
top-K item ranking against the entire catalog (~1.4K items) -> HR@K, NDCG@K.
Also reports unconstrained top-1 validity (does free generation produce a
real ID?) and popularity lift of the top-1 retrieved item.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from .semid import SidTable
from .sid_model import prepare


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
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--item-meta", default="data/item_meta.json")
    ap.add_argument("--data", default="data/sid_test.jsonl")
    ap.add_argument("--max-examples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--free-gen-n", type=int, default=50,
                    help="check unconstrained validity on N examples")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    table = SidTable(args.sid_table)
    tok, model, _ = prepare(args.model, table)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()

    meta = {int(i): v for i, v in json.load(open(args.item_meta)).items()}
    pop_mean = float(np.mean([m["pop_quantile"] for m in meta.values()]))
    rows = [json.loads(l) for l in open(args.data)][:args.max_examples]

    hr1, hrk, ndcg, lifts = [], [], [], []
    for r in rows:
        items = beam_retrieve(tok, model, device, table, r["prompt"], args.topk)
        tgt = r["target_item"]
        rank = items.index(tgt) if tgt in items else None
        hr1.append(rank == 0)
        hrk.append(rank is not None)
        ndcg.append(1.0 / np.log2(rank + 2) if rank is not None else 0.0)
        if items:
            lifts.append(meta[items[0]]["pop_quantile"] - pop_mean)

    valid = [free_top1_valid(tok, model, device, table, r["prompt"])
             for r in rows[:args.free_gen_n]]

    result = {
        "n": len(rows),
        f"hr@1": float(np.mean(hr1)),
        f"hr@{args.topk}": float(np.mean(hrk)),
        f"ndcg@{args.topk}": float(np.mean(ndcg)),
        "pop_lift@1": float(np.mean(lifts)) if lifts else None,
        "free_gen_valid_rate": float(np.mean(valid)) if valid else None,
        "model": args.model, "adapter": args.adapter, "data": args.data,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

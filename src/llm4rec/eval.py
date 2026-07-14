"""Evaluate a checkpoint on the choice task + bias probes.

Scoring: length-normalized log-likelihood of each candidate letter as the
assistant reply -> full ranking per example. This gives HR@1, NDCG, and lets
the probes re-score identical content under controlled cue changes.

Probes (RL-Shortcut-Lab cue dimensions):
  position   re-place the SAME target+negatives at every list position;
             accuracy-by-position curve + choice-position marginal
  popularity lift of chosen item's popularity quantile over candidate mean
  framing    run with --framing-file to compare neutral vs evaluative variants
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import LETTERS, build_prompt


def load_model(base: str, adapter: str | None, device: str):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return tok, model.to(device).eval()


@torch.no_grad()
def score_letters(tok, model, device, messages: list[dict], n: int) -> np.ndarray:
    """Log-prob of each letter A..<n> as the first assistant token(s)."""
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    out = model(ids.to(device))
    logits = out.logits[0, -1]  # next-token distribution
    logp = torch.log_softmax(logits.float(), dim=-1)
    scores = []
    for i in range(n):
        ids = tok.encode(LETTERS[i], add_special_tokens=False)
        scores.append(logp[ids[0]].item())
    return np.array(scores)


def ndcg_at(rank: int, k: int = 5) -> float:
    return 1.0 / np.log2(rank + 2) if rank < k else 0.0


def evaluate(tok, model, device, rows, position_probe_n=0):
    hits, ndcgs, pop_lifts, chosen_pos = [], [], [], []
    probe_acc = None
    for r in rows:
        scores = score_letters(tok, model, device, r["prompt"], len(r["candidates"]))
        order = np.argsort(-scores)
        rank = int(np.where(order == r["target"])[0][0])
        choice = int(order[0])
        hits.append(choice == r["target"])
        ndcgs.append(ndcg_at(rank))
        q = r["pop_quantiles"]
        pop_lifts.append(q[choice] - float(np.mean(q)))
        chosen_pos.append(choice)

    result = {
        "n": len(rows),
        "hr@1": float(np.mean(hits)),
        "ndcg@5": float(np.mean(ndcgs)),
        "pop_lift": float(np.mean(pop_lifts)),
        "chosen_pos_hist": dict(Counter(chosen_pos)),
    }

    if position_probe_n:
        sub = rows[:position_probe_n]
        C = len(sub[0]["candidates"])
        acc_by_pos = np.zeros(C)
        for r in sub:
            titles = r["candidates"]
            quants = r["pop_quantiles"]
            t = r["target"]
            tgt_title, tgt_q = titles[t], quants[t]
            neg_t = [x for j, x in enumerate(titles) if j != t]
            neg_q = [x for j, x in enumerate(quants) if j != t]
            hist = [line[2:] for line in
                    r["prompt"][1]["content"].split("\n\nCandidates:")[0]
                    .split("recently (oldest to newest):\n")[1].split("\n")]
            for pos in range(C):
                cand = neg_t[:pos] + [tgt_title] + neg_t[pos:]
                qq = neg_q[:pos] + [tgt_q] + neg_q[pos:]
                msgs = build_prompt(hist, cand, qq, r.get("framing", "neutral"))
                scores = score_letters(tok, model, device, msgs, C)
                acc_by_pos[pos] += float(np.argmax(scores) == pos)
        acc_by_pos /= len(sub)
        result["position_probe"] = {
            "n": len(sub),
            "acc_by_target_pos": acc_by_pos.round(3).tolist(),
            "spread": float(acc_by_pos.max() - acc_by_pos.min()),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="", help="LoRA adapter path ('' = base model)")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--max-examples", type=int, default=300)
    ap.add_argument("--position-probe", type=int, default=0,
                    help="run full-permutation position probe on N examples")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(args.model, args.adapter or None, device)

    rows = [json.loads(l) for l in open(args.data)][:args.max_examples]
    result = evaluate(tok, model, device, rows, args.position_probe)
    result["model"] = args.model
    result["adapter"] = args.adapter
    result["data"] = args.data

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

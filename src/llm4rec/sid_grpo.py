"""Stage 2 (semantic-ID route): GRPO with KL constraint on generative retrieval.

Sampling is unconstrained (invalid IDs are penalized and tracked — the spec's
invalid-rate deliverable); constrained beam search is applied at eval time.
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

from .semid import SidTable
from .sid_model import prepare
from .sid_reward import (make_minionerec_reward, make_pop_penalty,
                         make_rare_hit_bonus, make_sid_reward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sft-adapter", default="runs/sid_sft/final")
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--item-meta", default="data/item_meta.json")
    ap.add_argument("--train", default="data/sid_train.jsonl")
    ap.add_argument("--out", default="runs/sid_grpo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--prefix-credit", type=float, default=0.1)
    ap.add_argument("--reward", choices=["prefix", "minionerec"], default="prefix",
                    help="prefix: exact + prefix-credit shaping; "
                         "minionerec: binary rule + rank-aware hard-negative penalty")
    ap.add_argument("--pop-weight", type=float, default=0.0,
                    help="weight of the popularity penalty added as a second "
                         "reward function (0 = off)")
    ap.add_argument("--pop-anchor", choices=["catalog", "user"], default="catalog",
                    help="catalog: tax vs catalog-mean popularity (uniform); "
                         "user: tax vs the user's own history popularity (ΔGAP-aligned)")
    ap.add_argument("--pop-wrong-only", action="store_true",
                    help="apply the popularity penalty only to incorrect retrievals")
    ap.add_argument("--rare-hit-weight", type=float, default=0.0,
                    help="weight of the propensity-weighted exact-hit bonus "
                         "(+1/count^gamma when correct; 0 = off) — targets tail/IPS HR")
    ap.add_argument("--rare-hit-gamma", type=float, default=0.5,
                    help="propensity exponent for the rare-hit bonus")
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.train)["train"]
    ds = ds.remove_columns([c for c in ("answer",) if c in ds.column_names])

    table = SidTable(args.sid_table)
    tok, model, _ = prepare(args.model, table)
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    print(f"merged SFT adapter {args.sft_adapter}")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.reward == "minionerec":
        main_reward = make_minionerec_reward(args.sid_table, args.item_meta,
                                             num_generations=args.num_generations)
    else:
        main_reward = make_sid_reward(args.sid_table, args.item_meta,
                                      prefix_credit=args.prefix_credit)
    reward_funcs = [main_reward]
    reward_weights = [1.0]
    if args.pop_weight > 0:
        reward_funcs.append(make_pop_penalty(args.sid_table, args.item_meta,
                                             anchor=args.pop_anchor,
                                             wrong_only=args.pop_wrong_only))
        reward_weights.append(args.pop_weight)
    if args.rare_hit_weight > 0:
        reward_funcs.append(make_rare_hit_bonus(args.sid_table, args.item_meta,
                                                gamma=args.rare_hit_gamma))
        reward_weights.append(args.rare_hit_weight)

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=args.prompts_per_step * args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=8,
        temperature=args.temperature,
        beta=args.beta,
        learning_rate=args.lr,
        reward_weights=reward_weights,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print(f"saved GRPO adapter to {args.out}/final")


if __name__ == "__main__":
    main()

"""Stage 2: GRPO policy updates with KL constraint on the choice task.

Reward = choice_reward (hit / miss / invalid). Shortcut telemetry
(invalid rate, chosen-position mean, popularity lift) is logged per step
alongside trl's reward and KL metrics.
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from .reward import choice_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sft-adapter", default="runs/sft/final",
                    help="LoRA adapter from SFT to merge before RL ('' = start from base)")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.train)["train"]
    ds = ds.remove_columns([c for c in ("answer",) if c in ds.column_names])

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()  # spec: LoRA merge, then RL on top
        print(f"merged SFT adapter {args.sft_adapter}")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=args.prompts_per_step * args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=8,
        temperature=args.temperature,
        beta=args.beta,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=choice_reward,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    print(f"saved GRPO adapter to {args.out}/final")


if __name__ == "__main__":
    main()

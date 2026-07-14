"""Stage 1: LoRA supervised fine-tuning on the choice task."""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from .prompts import LETTERS


def to_chat(example):
    return {"messages": example["prompt"] + [{"role": "assistant", "content": example["answer"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=0, help="cap train examples (0 = all)")
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    ds = ds.map(to_chat, remove_columns=ds["train"].column_names)
    if args.max_train:
        ds["train"] = ds["train"].select(range(min(args.max_train, len(ds["train"]))))

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        max_length=1024,
        assistant_only_loss=True,
        bf16=torch.backends.mps.is_available() or torch.cuda.is_available(),
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16},
    )
    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"].select(range(min(200, len(ds["val"])))),
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    print(f"saved LoRA adapter to {args.out}/final")


if __name__ == "__main__":
    main()

"""Stage 1 (semantic-ID route): LoRA SFT on generative retrieval.

New sid token embeddings are trained via peft's trainable_token_indices
(only the new rows get gradients; handles Qwen's tied lm_head).
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from .semid import SidTable
from .sid_model import prepare


def to_chat(example):
    return {"messages": example["prompt"] + [{"role": "assistant", "content": example["answer"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--train", default="data/sid_train.jsonl")
    ap.add_argument("--val", default="data/sid_val.jsonl")
    ap.add_argument("--out", default="runs/sid_sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=0)
    args = ap.parse_args()

    table = SidTable(args.sid_table)
    tok, model, new_ids = prepare(args.model, table)

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    ds = ds.map(to_chat, remove_columns=ds["train"].column_names)
    if args.max_train:
        ds["train"] = ds["train"].select(range(min(args.max_train, len(ds["train"]))))

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        trainable_token_indices={"embed_tokens": new_ids},
    )
    model = get_peft_model(model, peft_cfg)
    head = model.get_output_embeddings()
    if not hasattr(head, "bias"):  # peft TrainableTokensWrapper lacks .bias; trl's loss reads it
        head.bias = None
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
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"].select(range(min(200, len(ds["val"])))),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print(f"saved LoRA adapter + tokenizer to {args.out}/final")


if __name__ == "__main__":
    main()

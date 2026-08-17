"""Stage 2, alternative to GRPO: distill the SDA target distribution directly.

Why this exists. SDA (see sid_reward.make_sda_reward) minimizes
D_KL(P* || Q_theta) via the reward P*/Q_theta. That reward is the *policy-gradient*
estimator of the KL, and on this task it is a bad one, for two measured reasons:

  1. GRPO normalizes advantages by each group's own std, so the magnitude of
     P*/Q_theta -- which is the entire mass-covering / anti-concentration signal
     of a forward KL -- is divided away before it reaches the gradient.
  2. The estimator samples from Q_theta, but the terms that dominate a forward
     KL are exactly those with Q_theta -> 0. At exposure Gini 0.97 the policy
     draws ~100 of 1682 items, so the coverage term is absent from every
     gradient estimate. "You can't reward what never happens in the rollouts."

But the same gradient has a second, better estimator:

    -grad D_KL(P* || Q) = E_{s ~ Q}[ (P*(s)/Q(s)) grad log Q(s) ]
                        = E_{s ~ P*}[ grad log Q(s) ]

Sampling from the *target* needs no importance ratio, no clipping and no
advantage normalization, and it puts tail items into training by construction
rather than hoping the policy stumbles onto them. A KL to a known distribution
over an enumerable output space (1682 items) is a distillation problem; casting
it as a reward buys nothing and costs the mechanism.

Loss (hard-label + soft-target distillation + an ensemble term):

    L = alpha * CE(i*)  +  (1 - alpha) * mean_m CE(i_m)  +  lambda * KL(Q_bar || P_bar)

with i_m ~ P~*, the (optionally propensity-debiased) target — sampled, not
top-M, see DistillRows. The label term defends HR, which pure distillation would
drag down toward the teacher's own accuracy; the soft term supplies the spread a
single label can never carry.

The third term is the one nothing in this repo has had. Every reward tried here
is per-example, but exposure Gini and coverage@K are properties of the
*ensemble* of recommendations across users, so a per-example loss can only
influence them indirectly — which is exactly the weak coupling that was
measured. KL(Q_bar || P_bar) compares the batch-mean predicted code
distribution against the target's own mean, making concentration a term in the
objective rather than a hoped-for side effect.

CLI:
  uv run python -m llm4rec.sid_distill --sft-adapter runs/sid_sft/final \
      --transition runs/transition/transition.pt --out runs/sid_distill
"""

import argparse
import json

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset

from .semid import SidTable
from .sid_model import prepare
from .sid_transition import Transition


class DistillRows(Dataset):
    """Prompts + M items **sampled from** the target, and its level-1 marginal.

    Targets are computed per batch rather than up front: a 300-step run at
    batch 4 touches ~1200 of the 3724 prompts, so precomputing the whole split
    would score 6.3M items to use a third of them. Per row it is one small-MLP
    pass over the catalog — milliseconds.

    **Sampled, not top-M.** The estimator being implemented is
    E_{s ~ P*}[grad log Q(s)], which needs draws from the target. Taking the
    top-M instead is mode-seeking: it trains on the most probable items, which
    even after debiasing are that distribution's head, so truncation silently
    re-concentrates the very spread this objective exists to transfer. Sampling
    keeps the estimate unbiased and puts tail items in the gradient by
    construction; the weights are then uniform, since the draw already carries
    the probability.
    """

    def __init__(self, path: str, transition: Transition, top_m: int, seed: int = 0):
        self.rows = [json.loads(l) for l in open(path)]
        self.transition = transition
        self.top_m = top_m
        self.catalog = np.array(transition.items)
        self.rng = np.random.default_rng(seed)
        # item -> level-1 code, for marginalizing the target down to 64 codes
        self.c1 = np.array([transition.table.codes[i][0] for i in transition.items])
        self.K = transition.table.K

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        logp = self.transition.log_p_target([row["history_items"]])[0]
        p = np.exp(logp - logp.max())
        p /= p.sum()
        idx = self.rng.choice(len(p), size=self.top_m, p=p)
        marg = np.bincount(self.c1, weights=p, minlength=self.K)  # P*(C1)
        return row, self.catalog[idx], np.full(self.top_m, 1.0 / self.top_m), marg


def make_collate(tok, table):
    """One sequence per (prompt, candidate item); labels mask everything but the ID."""
    def collate(batch):
        seqs, w, gold, margs = [], [], [], []
        for row, items, weights, marg in batch:
            prompt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                             tokenize=True)
            prompt = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
            if prompt and isinstance(prompt[0], (list, tuple)):
                prompt = prompt[0]
            prompt = list(prompt)
            margs.append(marg)
            cand = list(items) + [row["target_item"]]      # M sampled + 1 hard
            for j, it in enumerate(cand):
                ids = [tok.convert_tokens_to_ids(f"<s{l}_{c}>")
                       for l, c in enumerate(table.codes[int(it)])]
                seqs.append(prompt + ids)
                w.append(float(weights[j]) if j < len(items) else 0.0)
                gold.append(j == len(items))
        width = max(len(s) for s in seqs)
        pad = tok.pad_token_id or 0
        input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs])
        attn = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
        return (input_ids, attn, len(table.codes[int(cand[0])]), torch.tensor(w),
                torch.tensor(gold), torch.tensor(np.array(margs), dtype=torch.float32))
    return collate


def sequence_nll(model, input_ids, attn, n_label_tokens: int):
    """-log Q_theta(item | prompt) per row, plus the logits that predict code 1.

    Teacher-forced on the ID tokens. The first kept position is the one the
    exposure term reads: its distribution over the 64 level-1 code tokens is
    Q_theta(C1 | H), available for free from the same pass.
    """
    L = int(n_label_tokens)
    out = model(input_ids=input_ids, attention_mask=attn, logits_to_keep=L + 1)
    logits = out.logits[:, -(L + 1):, :].float()
    lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, -L:]
    nll = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).sum(-1)
    return nll, logits[:, 0, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sft-adapter", default="runs/sid_sft/final")
    ap.add_argument("--transition", default="runs/transition/transition.pt")
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--item-meta", default="data/item_meta.json")
    ap.add_argument("--train", default="data/sid_train.jsonl")
    ap.add_argument("--out", default="runs/sid_distill")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2, help="prompts per step")
    ap.add_argument("--top-m", type=int, default=4,
                    help="how many items to SAMPLE from the target per prompt")
    ap.add_argument("--lambda-exp", type=float, default=1.0,
                    help="weight on the batch-level exposure term (0 = off); the "
                         "only term that sees the ensemble Gini/coverage measure")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on the ground-truth label; 0 = pure distillation "
                         "(accuracy converges toward the teacher's own)")
    ap.add_argument("--pop-gamma", type=float, default=0.3,
                    help="propensity-debias the target: P~* ∝ P*/count^γ")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    table = SidTable(args.sid_table)
    tok, model, _ = prepare(args.model, table)
    model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
    print(f"merged SFT adapter {args.sft_adapter}")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    # Each prompt contributes top_m + 1 sequences of ~355 tokens, so the batch
    # is several times an SFT batch and the activations OOM a 16 GB Mac without
    # this. enable_input_require_grads is required for checkpointing under PEFT,
    # where the frozen embedding output would otherwise not need grad.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(device).train()

    transition = Transition(args.transition, args.sid_table, "cpu",
                            pop_gamma=args.pop_gamma, item_meta_path=args.item_meta)
    ds = DistillRows(args.train, transition, args.top_m, seed=args.seed)
    print(f"{len(ds)} prompts; {args.top_m} sampled from target, "
          f"pop_gamma={args.pop_gamma}, alpha={args.alpha}, "
          f"lambda_exp={args.lambda_exp}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    collate_fn=make_collate(tok, table))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, args.steps)
    level1_ids = torch.tensor(
        [tok.convert_tokens_to_ids(f"<s0_{c}>") for c in range(table.K)], device=device)

    step = 0
    while step < args.steps:
        for input_ids, attn, nlab, w, gold, marg in dl:
            input_ids, attn = input_ids.to(device), attn.to(device)
            w, gold, marg = w.to(device), gold.to(device), marg.to(device)
            nll, first_logits = sequence_nll(model, input_ids, attn, nlab)
            # soft term: mean over sampled items; hard term: CE(i*). Both are
            # means over prompts, so the two coefficients stay comparable.
            n_prompts = gold.sum().clamp(min=1)
            soft = (nll * w).sum() / n_prompts
            hard = (nll * gold.float()).sum() / n_prompts

            # Exposure term. Gini and coverage are properties of the *ensemble*
            # of recommendations across users, which no per-example loss can
            # address; this is the one term that sees the aggregate. Q_bar is
            # the batch-mean level-1 code distribution, matched to the target's
            # own mean marginal (not to uniform — the goal is the debiased
            # target's spread, not maximal entropy). Level-1 codes stand in for
            # the catalog: 64-way, free from this forward pass, and semantically
            # clustered so concentration tracks. Estimated from `batch` prompts,
            # so it is noisy per step and only unbiased in expectation.
            exp_loss = torch.zeros((), device=device)
            if args.lambda_exp > 0:
                q1 = torch.softmax(first_logits[gold][:, level1_ids], dim=-1)
                q_bar = q1.mean(0).clamp(min=1e-9)
                p_bar = marg.mean(0).clamp(min=1e-9)
                p_bar = p_bar / p_bar.sum()
                exp_loss = (q_bar * (q_bar.log() - p_bar.log())).sum()

            loss = args.alpha * hard + (1 - args.alpha) * soft + args.lambda_exp * exp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{args.steps}  loss {loss.item():.4f}  "
                      f"hard {hard.item():.4f}  soft {soft.item():.4f}  "
                      f"exp_kl {exp_loss.item():.4f}")
            if step >= args.steps:
                break

    model.save_pretrained(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print(f"saved distilled adapter to {args.out}/final")


if __name__ == "__main__":
    main()

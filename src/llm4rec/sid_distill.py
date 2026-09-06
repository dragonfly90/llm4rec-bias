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

Loss (default, --loss exact), following GKD (https://arxiv.org/abs/2306.13649):

    L = (1/L) sum_n JSD_beta( p_T(.|path_<n) || p_S(.|path_<n) )
        + alpha * CE(i*)  +  lambda_exp * KL(Q_bar || P_bar)

Three deliberate choices, each from a measurement in this repo:

  * **Exact, not sampled.** At each of the 3 SID positions the vocabulary is 64
    code tokens, so the two distributions are compared in full. The older
    --loss sampled draws M items and takes cross-entropy, which is noisy and
    makes alpha depend on the teacher's entropy -- swapping teachers then
    silently changes the effective hyperparameter (measured: ~1 nat shift at
    identical alpha, which is what made the v1/v2 teacher comparison
    uninterpretable).
  * **beta interpolates forward KL (0) to reverse KL (1).** Final policy
    entropy correlates +0.97 with HR@10 across every checkpoint here, so how
    peaked the student ends up is the dominant factor in retrieval quality;
    beta is the principled control over it. Mass-covering spreads the student
    (better coverage, worse HR@10), mode-seeking keeps it sharp.
  * **--on-policy scores a path sampled from the student.** GKD's central
    claim: training only on teacher/ground-truth sequences leaves a
    train-inference mismatch. Cheap here -- completions are 3 code tokens.

alpha is ADDITIVE, not a convex mixture: the divergence is ~0.07 nats and the
label cross-entropy ~5.6, so mixing them convexly at 0.5 would reduce the
divergence to a rounding error.

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
import math

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


def exact_divergence(model, transition, prompt_ids, attn, path_codes, lvl_ids,
                     histories, beta: float):
    """Per-position JSD between teacher and student over the full 64-code vocab.

    The sampled-item loss it replaces is a Monte-Carlo estimate from M draws,
    which is noisy and -- worse -- makes the alpha coefficient depend on the
    teacher's entropy, so swapping teachers silently changes the effective
    hyperparameter (measured: v1 vs v2 soft loss differed by ~1 nat at identical
    alpha). Here nothing is sampled: at each of the 3 SID positions the
    vocabulary is 64 code tokens, small enough to compare the two distributions
    exactly. Requires a code-head teacher, which factors the way the student
    generates.
    """
    L = lvl_ids.size(0)
    ids = torch.cat([prompt_ids,
                     torch.stack([lvl_ids[l][path_codes[:, l]] for l in range(L)], 1)], 1)
    mask = torch.cat([attn, torch.ones_like(path_codes)], 1)
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=L + 1).logits
    h = transition._encode(histories)                       # teacher state, no grad
    pc = path_codes.to(h.device)                            # teacher runs on CPU
    total = 0.0
    for l in range(L):
        # student: renormalize over this level's 64 code tokens
        s = torch.log_softmax(logits[:, l, lvl_ids[l]].float(), dim=-1)
        # teacher: same conditioning prefix, exact 64-way conditional
        with torch.no_grad():
            t = torch.log_softmax(
                transition.model.level_logits(h, [pc[:, j] for j in range(l)]), dim=-1)
        total = total + jsd(t.to(s.device), s, beta)
    return (total / L).mean()


def make_collate(tok, table):
    """One sequence per (prompt, candidate item); labels mask everything but the ID."""
    def collate(batch):
        seqs, w, gold, margs, prompts, golds, hists = [], [], [], [], [], [], []
        for row, items, weights, marg in batch:
            prompt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                             tokenize=True)
            prompt = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
            if prompt and isinstance(prompt[0], (list, tuple)):
                prompt = prompt[0]
            prompt = list(prompt)
            prompts.append(prompt); golds.append(row["target_item"])
            hists.append(row["history_items"])
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
        # prompt-only tensors + gold code path, for the exact-divergence loss
        pw = max(len(p) for p in prompts)
        p_ids = torch.tensor([[pad] * (pw - len(p)) + p for p in prompts])
        p_attn = torch.tensor([[0] * (pw - len(p)) + [1] * len(p) for p in prompts])
        gold_path = torch.tensor([list(table.codes[int(g)][:table.levels - 1]) for g in golds])
        return (input_ids, attn, len(table.codes[int(cand[0])]), torch.tensor(w),
                torch.tensor(gold), torch.tensor(np.array(margs), dtype=torch.float32),
                p_ids, p_attn, gold_path, hists)
    return collate


def jsd(logp_t: torch.Tensor, logp_s: torch.Tensor, beta: float) -> torch.Tensor:
    """Generalized Jensen-Shannon between two log-distributions (GKD, eq. 2).

    beta -> 0 recovers forward KL (mass-covering: the student must cover
    everything the teacher considers plausible); beta -> 1 recovers reverse KL
    (mode-seeking: the student may concentrate on one mode). That axis is
    exactly the sharpness knob this repo measured the hard way -- final policy
    entropy correlates +0.97 with HR@10, so how peaked the student ends up is
    the single best predictor of retrieval quality here.
    https://arxiv.org/abs/2306.13649
    """
    if beta <= 0:
        return (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    if beta >= 1:
        return (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    logm = torch.logaddexp(logp_t + math.log(beta), logp_s + math.log1p(-beta))
    return (beta * (logp_t.exp() * (logp_t - logm)).sum(-1)
            + (1 - beta) * (logp_s.exp() * (logp_s - logm)).sum(-1))


def level_token_ids(tok, table, device):
    """(levels, K) token ids: row l holds the ids of <sl_0> .. <sl_K-1>."""
    return torch.tensor(
        [[tok.convert_tokens_to_ids(f"<s{l}_{c}>") for c in range(table.K)]
         for l in range(table.levels - 1)], device=device)


@torch.no_grad()
def sample_path(model, prompt_ids, attn, lvl_ids, temperature: float = 1.0):
    """Autoregressively sample a SID prefix from the *student* (on-policy).

    GKD's central claim is that training on the student's own generations, not
    on teacher/dataset sequences, is what removes the train-inference
    distribution mismatch. Completions here are 3 code tokens, so this costs 3
    short forward passes -- the manoeuvre that is expensive for a 50K-vocab
    summarizer is nearly free at 64 codes.
    """
    ids, mask = prompt_ids, attn
    path = []
    for l in range(lvl_ids.size(0)):
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1]
        p = torch.softmax(logits[:, lvl_ids[l]].float() / temperature, dim=-1)
        code = torch.multinomial(p, 1)                       # (B, 1) index into K
        path.append(code)
        ids = torch.cat([ids, lvl_ids[l][code.squeeze(-1)].unsqueeze(-1)], dim=1)
        mask = torch.cat([mask, torch.ones_like(code)], dim=1)
    return torch.cat(path, dim=1), ids, mask


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
    ap.add_argument("--loss", choices=["sampled", "exact"], default="exact",
                    help="sampled: CE on M draws from the target (noisy, and "
                         "makes alpha depend on teacher entropy); exact: "
                         "per-position JSD over the full 64-code vocabulary")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="[exact] generalized JSD: 0 = forward KL (mass-covering), "
                         "1 = reverse KL (mode-seeking). Controls how peaked the "
                         "student ends up, which predicts HR@10 here")
    ap.add_argument("--on-policy", type=float, default=0.5,
                    help="[exact] fraction of steps scored on a path sampled from "
                         "the STUDENT rather than the ground-truth path (GKD lambda)")
    ap.add_argument("--soft-weight", type=float, default=1.0,
                    help="weight on the teacher term. 0 disables the teacher "
                         "entirely, leaving plain continued SFT -- the control "
                         "that decides whether the teacher contributes anything")
    ap.add_argument("--lambda-exp", type=float, default=1.0,
                    help="weight on the batch-level exposure term (0 = off); the "
                         "only term that sees the ensemble Gini/coverage measure")
    ap.add_argument("--alpha", type=float, default=0.02,
                    help="ADDITIVE weight on the ground-truth label CE (not a "
                         "convex mixture: the label term is ~80x the divergence "
                         "in magnitude). 0 = pure distillation, which GKD does "
                         "by construction since ground truth enters via the path")
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
    lvl_ids = level_token_ids(tok, table, device)
    level1_ids = lvl_ids[0]
    if args.loss == "exact" and transition.model.head != "code":
        raise SystemExit("--loss exact needs a code-head teacher (train T_phi "
                         "without --head item): the per-position divergence "
                         "requires the teacher to factor as the student generates")

    step = 0
    while step < args.steps:
        for (input_ids, attn, nlab, w, gold, marg,
             p_ids, p_attn, gold_path, hists) in dl:
            input_ids, attn = input_ids.to(device), attn.to(device)
            w, gold, marg = w.to(device), gold.to(device), marg.to(device)
            p_ids, p_attn = p_ids.to(device), p_attn.to(device)
            gold_path = gold_path.to(device)
            nll, first_logits = sequence_nll(model, input_ids, attn, nlab)
            # soft term: mean over sampled items; hard term: CE(i*). Both are
            # means over prompts, so the two coefficients stay comparable.
            n_prompts = gold.sum().clamp(min=1)
            hard = (nll * gold.float()).sum() / n_prompts
            if args.loss == "exact":
                if np.random.rand() < args.on_policy:
                    path, _, _ = sample_path(model, p_ids, p_attn, lvl_ids)
                else:
                    path = gold_path
                soft = exact_divergence(model, transition, p_ids, p_attn, path,
                                        lvl_ids, hists, args.beta)
            else:
                soft = (nll * w).sum() / n_prompts

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

            # NOT a convex combination: with --loss exact the two terms carry
            # different units. The divergence is a JSD between 64-way code
            # distributions (~0.07 nats); the label term is a cross-entropy over
            # a 4-token sequence (~5.6 nats), ~80x larger. Convex mixing at
            # alpha=0.5 would silently reduce the divergence to a rounding
            # error, which is the same class of bug that made alpha
            # teacher-dependent under --loss sampled.
            loss = (args.soft_weight * soft + args.alpha * hard
                    + args.lambda_exp * exp_loss)
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

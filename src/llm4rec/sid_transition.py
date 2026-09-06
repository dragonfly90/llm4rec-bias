"""Next-step semantic transition model T_phi for SDA (Semantic Distribution
Alignment).

Design spec: "Semantic Distribution Alignment for Bias-Resistant RL in
MiniOneRec" (llm4rec-bias-Integrated issue #2). The idea: instead of hand-mixing
heuristic reward terms, define a *target distribution* over the next semantic ID
and align the policy to it. This module builds that target.

    P*(s) = T_phi(s | P_t, H_t),     s = (c1, c2, c3)

factored along the SID hierarchy, exactly as the policy factors it:

    P*(C1, C2, C3) = P*(C1) · P*(C2 | C1) · P*(C3 | C1, C2)

so the SDA loss D_KL(P* || Q_theta) decomposes by KL chain rule into
D1 + D_{2|1} + D_{3|12} without any hand-set per-level weights.

**User SID preference state P_t.** The spec defines P_t^(l) as the histogram of
the history's level-l SID *prefixes*. Full prefix histograms are 64, 64^2, 64^3
wide; we feed the per-level marginal histograms (3 x 64, the l=1 case exact, the
deeper ones marginalized) *plus* pooled code embeddings of the history, which
carry the joint information the marginals drop. Cheap and sufficient at this
scale (1682 items).

**Training.** Supervised on real next-step interactions, L_T = -log T_phi(
SID(i_{t+1}) | P_t, H_t), summed over the three levels (= the joint NLL).
Windows are drawn from the *training region* of each user sequence only
(s[:-2]); the val/test targets are never a training label, so the RL reward
built on top of this model cannot leak the held-out answer.

**Collision level.** This repo's SIDs carry a 4th, semantics-free
collision-breaking code. T_phi models the three semantic levels; item-level
probability spreads the 3-prefix mass uniformly over the items sharing it:

    P*(item i) = P*(c1, c2, c3) / |{j : prefix3(j) = prefix3(i)}|

CLI: uv run python -m llm4rec.sid_transition --out runs/transition
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import download_ml100k, load_interactions, popularity_stats
from .semid import SidTable

PAD = -1


class TransitionModel(nn.Module):
    """History of SID codes -> hierarchical distribution over the next SID.

    Encoder: per-level code embeddings, recency-weighted mean pooling, the last
    item, and the per-level preference histograms P_t. Heads: one softmax per
    level, each conditioned on the codes already chosen at the levels above —
    the same autoregressive factorization the LLM policy uses.
    """

    def __init__(self, levels: int = 3, K: int = 64, d: int = 128,
                 hidden: int = 256, decay: float = 0.9, dropout: float = 0.3,
                 n_items: int = 0, item_emb: bool = False, head: str = "code"):
        super().__init__()
        self.levels, self.K, self.decay = levels, K, decay
        self.head, self.n_items = head, n_items
        self.code_emb = nn.ModuleList([nn.Embedding(K, d) for _ in range(levels)])
        # Item-identity embeddings. Without these the model sees only SID codes,
        # which are a coarse quantization of *text* (title + genres) embeddings —
        # so it can represent semantic neighbourhoods but not collaborative
        # co-occurrence, which is exactly what a last-item Markov baseline
        # exploits to match it. Input-side only, so the output factorization the
        # SDA KL relies on is untouched.
        self.item_emb = nn.Embedding(n_items, d) if item_emb else None
        self.enc = nn.Sequential(
            nn.Linear(2 * d + levels * K, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        if head == "item":
            # Direct softmax over the catalog. Gives up the hierarchical chain
            # rule (and its per-level telemetry) for the ability to separate two
            # items that share all three codes.
            self.item_head = nn.Linear(hidden, n_items)
        else:
            # head l sees the encoder state plus the embeddings of codes 1..l-1
            self.heads = nn.ModuleList(
                [nn.Linear(hidden + l * d, K) for l in range(levels)])

    def encode(self, codes: torch.Tensor, items: torch.Tensor | None = None) -> torch.Tensor:
        """codes: (B, T, levels) long, PAD-filled -> (B, hidden)."""
        mask = (codes[:, :, 0] != PAD).float()                    # (B, T)
        safe = codes.clamp(min=0)
        item = sum(self.code_emb[l](safe[:, :, l]) for l in range(self.levels))
        if self.item_emb is not None and items is not None:
            item = item + self.item_emb(items.clamp(min=0))
        # recency weighting: the newest shown item counts most
        pos = mask.cumsum(1)                                      # 1..n over real items
        n = mask.sum(1, keepdim=True).clamp(min=1)
        w = (self.decay ** (n - pos)) * mask
        pooled = (item * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-6)
        last_idx = (n.long() - 1).clamp(min=0)
        last = item.gather(1, last_idx.unsqueeze(-1).expand(-1, 1, item.size(-1))).squeeze(1)
        # P_t: per-level code histograms of the history (the spec's preference state)
        hists = []
        for l in range(self.levels):
            oh = F.one_hot(safe[:, :, l], self.K).float() * mask.unsqueeze(-1)
            hists.append(oh.sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-6))
        return self.enc(torch.cat([pooled, last] + hists, dim=-1))

    def level_logits(self, h: torch.Tensor, prev_codes: list) -> torch.Tensor:
        """Logits for level len(prev_codes), conditioned on the codes above it."""
        l = len(prev_codes)
        ctx = [h] + [self.code_emb[j](prev_codes[j]) for j in range(l)]
        return self.heads[l](torch.cat(ctx, dim=-1))

    def forward(self, codes: torch.Tensor, target: torch.Tensor,
                label_smoothing: float = 0.0, items: torch.Tensor | None = None,
                target_item: torch.Tensor | None = None) -> torch.Tensor:
        """Joint NLL of the target: summed over SID levels, or over the catalog."""
        h = self.encode(codes, items)
        if self.head == "item":
            return F.cross_entropy(self.item_head(h), target_item,
                                   label_smoothing=label_smoothing)
        loss = 0.0
        for l in range(self.levels):
            logits = self.level_logits(h, [target[:, j] for j in range(l)])
            loss = loss + F.cross_entropy(logits, target[:, l],
                                          label_smoothing=label_smoothing)
        return loss


# ---------------- data ----------------

def build_windows(seqs: dict, table: SidTable, history_len: int, holdout: int = 2):
    """(history, next item) pairs from the training region of every sequence.

    holdout=2 drops the last two interactions of each user (the val and test
    targets), so no supervision here can leak the held-out answer.
    """
    pairs = []
    for s in seqs.values():
        s = [i for i in s if i in table.codes]
        for t in range(2, max(len(s) - holdout, 2)):
            pairs.append((s[max(0, t - history_len):t], s[t]))
    return pairs


def encode_batch(histories: list, table: SidTable, levels: int, history_len: int,
                 device, item_pos: dict | None = None):
    """item-id lists -> (B, T, levels) codes, PAD-filled, and (B, T) item rows."""
    B = len(histories)
    out = np.full((B, history_len, levels), PAD, dtype=np.int64)
    idx = np.full((B, history_len), PAD, dtype=np.int64)
    for b, h in enumerate(histories):
        h = [i for i in h if i in table.codes][-history_len:]
        for t, i in enumerate(h):
            out[b, t] = table.codes[i][:levels]
            if item_pos is not None:
                idx[b, t] = item_pos[i]
    return torch.from_numpy(out).to(device), torch.from_numpy(idx).to(device)


# ---------------- inference wrapper ----------------

class Transition:
    """Loaded T_phi + catalog bookkeeping: gives log P*(item | history).

    Used by the SDA reward (single items, batch of rollouts) and by this
    module's own validation (full catalog ranking).
    """

    def __init__(self, ckpt: str, sid_table_path: str, device: str = "cpu",
                 smooth: float | None = None, pop_gamma: float = 0.0,
                 item_meta_path: str | None = None):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        self.model = TransitionModel(
            **{k: cfg[k] for k in ("levels", "K", "d", "hidden", "decay")},
            dropout=cfg.get("dropout", 0.0), n_items=cfg.get("n_items", 0),
            item_emb=cfg.get("item_emb", False), head=cfg.get("head", "code"))
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.levels = cfg["levels"]
        self.history_len = cfg["history_len"]
        # uniform smoothing: P* <- (1-e)P_model + e/|catalog|. Guarantees full
        # support (the KL in the spec needs it), bounds log P* from below so the
        # reward can't explode on an item the model considers impossible, and
        # keeps the smoothing *bias-neutral* — mixing toward a popularity prior
        # would inject the very bias SDA is meant to resist.
        self.smooth = cfg.get("smooth", 0.05) if smooth is None else smooth
        self.table = SidTable(sid_table_path)
        # items sharing a 3-level semantic prefix: the collision level is not
        # semantic, so P*(prefix) is split evenly across its members
        groups = defaultdict(list)
        for item, codes in self.table.codes.items():
            groups[tuple(codes[:self.levels])].append(item)
        self.group = groups
        self.items = sorted(self.table.codes)
        self.item_codes = torch.tensor(
            [self.table.codes[i][:self.levels] for i in self.items],
            dtype=torch.long, device=device)
        self.log_group_size = torch.tensor(
            [math.log(len(groups[tuple(self.table.codes[i][:self.levels])]))
             for i in self.items], dtype=torch.float32, device=device)
        self.item_pos = {i: k for k, i in enumerate(self.items)}

        # Propensity-debiased target: P~*(i) ∝ P*(i) / count(i)^gamma.
        # The measured failure of plain SDA was that the policy inherits the
        # teacher's popularity profile, so the fix belongs in the *target*, not
        # in a separate penalty fighting the alignment objective. This keeps one
        # KL objective — the policy is aligned to the next-step distribution the
        # user would have under uniform exposure — instead of two terms pulling
        # against each other. Unlike the (inert) rare-hit bonus, this reweights
        # every valid rollout, not only correct retrievals of rare items.
        self.pop_gamma = pop_gamma
        self.log_bias = None
        if pop_gamma:
            if item_meta_path is None:
                raise ValueError("pop_gamma > 0 needs item_meta_path for counts")
            meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
            counts = np.array([max(meta[i].get("count", 1), 1) for i in self.items],
                              dtype=np.float64)
            self.log_bias = -pop_gamma * np.log(counts)

    def _mix(self, logp):
        """(1-e)·P_model + e/|catalog|, in log space."""
        if not self.smooth:
            return logp
        floor = math.log(self.smooth / len(self.items))
        return np.logaddexp(math.log1p(-self.smooth) + logp, floor)

    def _debias(self, logp, rows=None):
        """Apply -gamma·log count and renormalize over the catalog.

        logp is (B, |catalog|) when rows is None, else (B,) with `rows` giving
        each entry's catalog index; the normalizer needs the full row either
        way, so callers with single items pass the full matrix separately.
        """
        if self.log_bias is None:
            return logp
        return logp + (self.log_bias if rows is None else self.log_bias[rows])

    def _log_norm(self, full_logp):
        """logsumexp of the debiased catalog row — a per-prompt constant.

        GRPO standardizes within a group and every rollout in a group shares the
        history, so this cancels in the advantage; we subtract it anyway so that
        log P~* stays a genuine log-probability and the logged telemetry means
        what it says.
        """
        if self.log_bias is None:
            return np.zeros(len(full_logp))
        x = full_logp + self.log_bias
        m = x.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(x - m).sum(axis=1)))

    @torch.no_grad()
    def _encode(self, histories):
        codes, idx = encode_batch(histories, self.table, self.levels,
                                  self.history_len, self.device, self.item_pos)
        return self.model.encode(codes, idx)

    @torch.no_grad()
    def log_p_items(self, histories: list, items: list, per_level: bool = False):
        """log P*(item_b | history_b), one per row. items[b] may be None -> nan.

        per_level also returns the (B, levels+1) breakdown by SID level — the
        chain-rule terms log P*(c1), log P*(c2|c1), log P*(c3|c1,c2) plus the
        collision-level split — which the reward logs as per-level mismatch
        telemetry (spec section 10). Note the breakdown is pre-smoothing, so its
        row sums differ slightly from the mixed total.
        """
        if self.model.head == "item":
            full = self.log_p_all(histories)
            out = np.full(len(items), np.nan)
            lv = np.full((len(items), self.levels + 1), np.nan)
            for b, it in enumerate(items):
                if it is not None:
                    out[b] = full[b, self.item_pos[it]]
            return (out, lv) if per_level else out
        h = self._encode(histories)
        valid = [b for b, it in enumerate(items) if it is not None]
        out = np.full(len(items), np.nan, dtype=np.float64)
        levels_out = np.full((len(items), self.levels + 1), np.nan, dtype=np.float64)
        if not valid:
            return (out, levels_out) if per_level else out
        hv = h[valid]
        codes = torch.tensor([self.table.codes[items[b]][:self.levels] for b in valid],
                             dtype=torch.long, device=self.device)
        lp = torch.zeros(len(valid), device=self.device)
        for l in range(self.levels):
            logits = self.model.level_logits(hv, [codes[:, j] for j in range(l)])
            step = F.log_softmax(logits, dim=-1).gather(1, codes[:, l:l + 1]).squeeze(1)
            lp = lp + step
            levels_out[valid, l] = step.cpu().numpy()
        gsz = self.log_group_size[[self.item_pos[items[b]] for b in valid]]
        levels_out[valid, self.levels] = -gsz.cpu().numpy()
        out[valid] = self._mix((lp - gsz).cpu().numpy())
        if self.log_bias is not None:
            # the debiased target needs a catalog-wide normalizer, so score the
            # whole row (a small-MLP pass over 1682 items — milliseconds)
            full = self.log_p_all(histories)
            rows = np.array([self.item_pos[items[b]] for b in valid])
            out[valid] = (out[valid] + self.log_bias[rows]
                          - self._log_norm(full)[valid])
        return (out, levels_out) if per_level else out

    @torch.no_grad()
    def level1_probs(self, histories: list) -> torch.Tensor:
        """P*(C1) per row — the coarse-grained target, for the D1 telemetry.

        With an item-level head there is no C1 factor to read off, so it is
        recovered by summing item probabilities within each level-1 code group.
        """
        if self.model.head == "item":
            p = torch.tensor(np.exp(self.log_p_all(histories)), device=self.device)
            c1 = torch.tensor([self.table.codes[i][0] for i in self.items],
                              device=self.device)
            agg = torch.zeros(p.size(0), self.table.K, device=self.device)
            agg.index_add_(1, c1, p)
            return agg / agg.sum(-1, keepdim=True).clamp(min=1e-12)
        return F.softmax(self.model.level_logits(self._encode(histories), []), dim=-1)

    @torch.no_grad()
    def log_p_all(self, histories: list, chunk: int = 32) -> np.ndarray:
        """(B, |catalog|) matrix of log P*(item | history). Validation only."""
        out = []
        for s in range(0, len(histories), chunk):
            h = self._encode(histories[s:s + chunk])
            if self.model.head == "item":
                lp = F.log_softmax(self.model.item_head(h), dim=-1).cpu().numpy()
                out.append(self._mix(lp))
                continue
            B, n = h.size(0), len(self.items)
            hh = h.unsqueeze(1).expand(B, n, h.size(-1)).reshape(B * n, -1)
            codes = self.item_codes.unsqueeze(0).expand(B, n, self.levels).reshape(B * n, -1)
            lp = torch.zeros(B * n, device=self.device)
            for l in range(self.levels):
                logits = self.model.level_logits(hh, [codes[:, j] for j in range(l)])
                lp = lp + F.log_softmax(logits, dim=-1).gather(1, codes[:, l:l + 1]).squeeze(1)
            out.append(self._mix((lp.view(B, n) - self.log_group_size).cpu().numpy()))
        return np.concatenate(out)

    def log_p_target(self, histories: list) -> np.ndarray:
        """(B, |catalog|) log P~*: the actual alignment target, debiased and
        renormalized. Identical to log_p_all when pop_gamma = 0."""
        full = self.log_p_all(histories)
        if self.log_bias is None:
            return full
        return full + self.log_bias - self._log_norm(full)[:, None]


# ---------------- evaluation ----------------

def gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * x)) / (n * total) - (n + 1) / n)


def evaluate(tr: Transition, rows: list, item_counts: dict, item_meta_path: str) -> dict:
    """Two questions about the SDA target distribution.

    1. Is T_phi a good enough teacher? It must beat *uniform* (or it carries no
       information) and *popularity* (or SDA is a bias amplifier wearing a
       distribution-alignment costume). Calibration (joint NLL) matters as much
       as ranking, because the reward consumes log P* values, not just order.
    2. What bias does the teacher itself carry? Whatever P* looks like on
       pop_lift / ΔGAP / Gini is the fixed point SDA pulls the policy toward —
       i.e. the ceiling on what alignment alone can fix. The spec flags this as
       a limitation ("P_t itself may contain popularity bias"); these numbers
       are that limitation, measured.
    """
    hist = [r["history_items"] for r in rows]
    logp = tr.log_p_target(hist)
    pos = np.array([tr.item_pos[r["target_item"]] for r in rows])
    tgt_lp = logp[np.arange(len(pos)), pos]
    rank = (logp > tgt_lp[:, None]).sum(1)

    counts = np.array([item_counts.get(i, 0) for i in tr.items], dtype=np.float64)
    pop = np.log((counts + 1) / (counts + 1).sum())
    pop_rank = (pop[None, :] > pop[pos][:, None]).sum(1)

    meta = {int(k): v for k, v in json.load(open(item_meta_path)).items()}
    q = np.array([meta[i]["pop_quantile"] for i in tr.items])
    top1 = logp.argmax(1)
    top10 = np.argsort(-logp, axis=1)[:, :10]
    exposure = np.bincount(top10.ravel(), minlength=len(tr.items))
    hist_pop = np.array([r["hist_pop_mean"] for r in rows])

    return {
        "n": len(pos),
        "val_item_nll": float(-tgt_lp.mean()),
        "uniform_nll": float(math.log(len(tr.items))),
        "popularity_nll": float(-pop[pos].mean()),
        "hr@1": float((rank < 1).mean()),
        "hr@10": float((rank < 10).mean()),
        "ndcg@10": float(np.mean(np.where(rank < 10, 1 / np.log2(rank + 2), 0.0))),
        "popularity_hr@10": float((pop_rank < 10).mean()),
        # the target distribution's own bias profile = SDA's fixed point
        "pop_lift@1": float(q[top1].mean() - q.mean()),
        "delta_gap": float(np.mean(q[top1] - hist_pop)),
        "exposure_gini": gini(exposure),
        "coverage@10": float(np.mean(exposure > 0)),
        "smooth": tr.smooth,
    }


# ---------------- training ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/transition")
    ap.add_argument("--data", default="data")
    ap.add_argument("--sid-table", default="data/semantic_ids.json")
    ap.add_argument("--val", default="data/sid_val.jsonl")
    ap.add_argument("--item-meta", default="data/item_meta.json")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; report on the checkpoint already in --out")
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.05,
                    help="uniform mixing weight on P*: (1-e)P_model + e/|catalog|")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--decay", type=float, default=0.9,
                    help="recency weight per step back in the history")
    ap.add_argument("--item-emb", action="store_true",
                    help="add per-item identity embeddings to the encoder input; "
                         "without them the model sees only SID codes, a coarse "
                         "quantization of item TEXT, so it cannot represent "
                         "collaborative co-occurrence at all")
    ap.add_argument("--head", choices=["code", "item"], default="code",
                    help="code: hierarchical P(C1)P(C2|C1)P(C3|C1,C2), keeps the "
                         "KL chain rule; item: direct softmax over the catalog")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    table = SidTable(args.sid_table)
    levels = table.levels - 1  # drop the collision level: not semantic

    raw = download_ml100k(Path(args.data))
    seqs, _ = load_interactions(raw)
    item_counts, _ = popularity_stats(seqs)

    if args.eval_only:
        out = Path(args.out)
        tr = Transition(str(out / "transition.pt"), args.sid_table, device)
        rows = [json.loads(l) for l in open(args.val)]
        report = evaluate(tr, rows, item_counts, args.item_meta)
        report["data"] = args.val
        name = "transition_eval" + ("" if "val" in Path(args.val).stem else "_test")
        json.dump(report, open(out / f"{name}.json", "w"), indent=2)
        print(json.dumps(report, indent=2))
        return

    pairs = build_windows(seqs, table, args.history)
    print(f"{len(pairs)} next-step training windows from {len(seqs)} users "
          f"(last 2 interactions per user held out)")

    item_pos = {i: k for k, i in enumerate(sorted(table.codes))}
    hist, hist_idx = encode_batch([h for h, _ in pairs], table, levels,
                                  args.history, device, item_pos)
    tgt = torch.tensor([table.codes[t][:levels] for _, t in pairs],
                       dtype=torch.long, device=device)
    tgt_item = torch.tensor([item_pos[t] for _, t in pairs],
                            dtype=torch.long, device=device)

    val_rows = [json.loads(l) for l in open(args.val)]
    val_hist = [r["history_items"] for r in val_rows]
    val_tgt = [r["target_item"] for r in val_rows]
    val_codes_h, val_idx = encode_batch(val_hist, table, levels, args.history,
                                        device, item_pos)
    val_codes_t = torch.tensor([table.codes[t][:levels] for t in val_tgt],
                               dtype=torch.long, device=device)
    val_tgt_item = torch.tensor([item_pos[t] for t in val_tgt],
                                dtype=torch.long, device=device)

    model = TransitionModel(levels=levels, K=table.K, d=args.d, hidden=args.hidden,
                            decay=args.decay, dropout=args.dropout,
                            n_items=len(item_pos), item_emb=args.item_emb,
                            head=args.head).to(device)
    print(f"T_phi: {sum(p.numel() for p in model.parameters()):,} params "
          f"(item_emb={args.item_emb}, head={args.head})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    n = len(pairs)
    # The SDA reward consumes log P* values, not just the ranking, so the model
    # is selected on held-out *calibration* (val joint NLL), not on val HR.
    best, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            loss = model(hist[idx], tgt[idx], label_smoothing=args.label_smoothing,
                         items=hist_idx[idx], target_item=tgt_item[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = float(model(val_codes_h, val_codes_t, items=val_idx,
                                target_item=val_tgt_item))
        flag = ""
        if vloss < best:
            best, best_state = vloss, {k: v.detach().clone()
                                       for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"epoch {ep + 1}/{args.epochs}  train joint NLL {tot / n:.4f}  "
              f"val joint NLL {vloss:.4f}{flag}")

    model.load_state_dict(best_state)
    print(f"restored best checkpoint (val joint NLL {best:.4f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(levels=levels, K=table.K, d=args.d, hidden=args.hidden,
               decay=args.decay, dropout=args.dropout,
               history_len=args.history, smooth=args.smooth,
               n_items=len(item_pos), item_emb=args.item_emb, head=args.head)
    torch.save({"config": cfg, "state_dict": model.state_dict()}, out / "transition.pt")
    print(f"saved {out}/transition.pt")

    tr = Transition(str(out / "transition.pt"), args.sid_table, device)
    report = evaluate(tr, val_rows, item_counts, args.item_meta)
    report["train_windows"] = len(pairs)
    json.dump(report, open(out / "transition_eval.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

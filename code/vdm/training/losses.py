"""Training objectives (plan 7).

    L = L_dec + lambda_a * L_ans + lambda_r * L_rank + lambda_c * L_cons

L_dec   cross entropy on the decision, over the valid option slots only, and
        masked out entirely when the observation no longer supports an answer.
        Forcing the original label onto a degraded image is exactly the habit
        the paper argues against, so the mask is not an implementation detail.

L_ans   binary cross entropy on "does this observation carry enough evidence".

L_rank  a margin on sufficiency *within* a base judgment: the intact image must
        score above its evidence-degraded twin.  Being a within-pair term it is
        insensitive to how hard the question is, which an absolute BCE is not.

L_cons  symmetric KL between the decision distributions of variants that are
        supposed to mean the same thing: an equal-area degradation of an
        irrelevant region, a paraphrase, a permutation of the options.
        Deliberately NOT applied to evidence-degraded pairs -- demanding the
        same answer after the evidence is gone would train overconfidence.
"""
from __future__ import annotations

import collections

import torch
import torch.nn.functional as F

SEMANTIC_PRESERVING = ("candidate_permutation", "paraphrase")


def masked_log_softmax(logits: torch.Tensor, n_options: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, k in enumerate(n_options):
        mask[i, :k] = True
    return torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)


def decision_logits(out: dict, qtypes: list[str], n_options: list[int]) -> torch.Tensor:
    """Route each row to its typed head and return a common-width logit tensor."""
    choice, claim = out["choice"], out["claim"]
    width = choice.shape[1]
    z = choice.new_full((choice.shape[0], width), -1e4)
    for i, qt in enumerate(qtypes):
        if qt == "claim":
            z[i, : claim.shape[1]] = claim[i]
        else:
            z[i, : n_options[i]] = choice[i, : n_options[i]]
    return z


def answer_sft_loss(lm_logits: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor, option_token_ids: torch.Tensor) -> torch.Tensor:
    """Ordinary next-token cross entropy on the answer label.

    This is the control for "is the decision head doing anything, or is the
    gain just from training on this data at all". Same prompt, same readout
    position, same supervision -- only the head differs, and the loss is over
    the full vocabulary rather than a pre-restricted option set, so nothing is
    handed to the baseline that a normal fine-tune would not have.
    """
    if mask.sum() == 0:
        return lm_logits.sum() * 0.0
    tgt = option_token_ids.to(lm_logits.device)[labels.clamp(min=0)]
    nll = F.cross_entropy(lm_logits.float(), tgt, reduction="none")
    return (nll * mask.float()).sum() / mask.float().sum()


def decision_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                  n_options: list[int]) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    logp = masked_log_softmax(logits, n_options)
    tgt = labels.clamp(min=0)
    nll = -logp.gather(1, tgt[:, None]).squeeze(1)
    return (nll * mask.float()).sum() / mask.float().sum()


def answerability_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = target >= 0
    if m.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[m], target[m])


def _pairs_by_base(base_ids: list[str], interventions: list[str]):
    idx = collections.defaultdict(dict)
    for i, (b, iv) in enumerate(zip(base_ids, interventions)):
        idx[b][iv] = i
    return idx


def sufficiency_rank_loss(ans_logit: torch.Tensor, base_ids: list[str],
                          interventions: list[str], margin: float = 1.0) -> torch.Tensor:
    """hinge(margin - (s_original - s_evidence_degraded)) over in-batch pairs."""
    idx = _pairs_by_base(base_ids, interventions)
    lo, hi = [], []
    for _, d in idx.items():
        if "none" not in d:
            continue
        for iv, j in d.items():
            if iv.endswith("_relevant"):
                hi.append(d["none"])
                lo.append(j)
    if not hi:
        return ans_logit.sum() * 0.0
    hi_t = ans_logit[torch.tensor(hi, device=ans_logit.device)]
    lo_t = ans_logit[torch.tensor(lo, device=ans_logit.device)]
    return F.relu(margin - (hi_t - lo_t)).mean()


def consistency_loss(logits: torch.Tensor, base_ids: list[str], interventions: list[str],
                     n_options: list[int], permutations: list[list[int] | None]) -> torch.Tensor:
    """Symmetric KL between variants that must mean the same thing."""
    idx = _pairs_by_base(base_ids, interventions)
    logp = masked_log_softmax(logits, n_options)
    terms = []
    for _, d in idx.items():
        if "none" not in d:
            continue
        a = d["none"]
        for iv, b in d.items():
            if iv == "none":
                continue
            if not (iv.endswith("_irrelevant") or iv in SEMANTIC_PRESERVING):
                continue
            k = min(n_options[a], n_options[b])
            pa, pb = logp[a, :k], logp[b, :k]
            if iv == "candidate_permutation":
                order = permutations[b]
                if order is None or len(order) != k:
                    continue
                # The variant's slot j holds the option that sat at order[j] in
                # the original, so reorder the original into the variant's slot
                # space before comparing.
                pa = logp[a, :k][torch.tensor(order, device=logits.device)]
            ea, eb = pa.exp(), pb.exp()
            terms.append(0.5 * ((ea * (pa - pb)).sum() + (eb * (pb - pa)).sum()))
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def total_loss(out: dict, batch: dict, cfg) -> tuple[torch.Tensor, dict[str, float]]:
    qtypes, n_opts = batch["qtypes"], batch["n_options"]
    if getattr(cfg, "readout", "head") == "lm":
        dev = out["lm_logits"].device
        l_dec = answer_sft_loss(out["lm_logits"], batch["labels"].to(dev),
                                batch["decision_mask"].to(dev), cfg.option_token_ids)
        parts = {"dec": float(l_dec.detach()), "total": float(l_dec.detach())}
        return l_dec, parts
    z = decision_logits(out, qtypes, n_opts)
    dev = z.device
    l_dec = decision_loss(z, batch["labels"].to(dev), batch["decision_mask"].to(dev), n_opts)
    parts = {"dec": float(l_dec.detach())}
    loss = l_dec

    if cfg.lambda_a > 0:
        l_ans = answerability_loss(out["answerable"], batch["answerable"].to(dev))
        loss = loss + cfg.lambda_a * l_ans
        parts["ans"] = float(l_ans.detach())
    if cfg.lambda_r > 0:
        l_rank = sufficiency_rank_loss(out["answerable"], batch["base_ids"], batch["interventions"],
                                       margin=cfg.rank_margin)
        loss = loss + cfg.lambda_r * l_rank
        parts["rank"] = float(l_rank.detach())
    if cfg.lambda_c > 0:
        l_cons = consistency_loss(z, batch["base_ids"], batch["interventions"], n_opts,
                                  batch.get("permutations", [None] * len(qtypes)))
        loss = loss + cfg.lambda_c * l_cons
        parts["cons"] = float(l_cons.detach())
    parts["total"] = float(loss.detach())
    return loss, parts

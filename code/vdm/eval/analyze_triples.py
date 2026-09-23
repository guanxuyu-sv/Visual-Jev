"""The go/no-go analysis (plan 12.1 step 7).

Question: when the evidence a question depends on is removed, does the backbone
notice?  We compare three arms of the same base judgment:

  original    evidence present
  relevant    evidence region degraded
  irrelevant  an equal-area region degraded, evidence intact

If accuracy falls on the relevant arm while confidence does not, the failure the
paper is about is real and measurable.  If confidence falls just as much on the
irrelevant arm, the model is tracking degradation, not evidence.
"""
from __future__ import annotations

import argparse, collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm.eval import metrics as M


def load(pred_path: str, items_path: str):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(items_path)}
    rows = [json.loads(l) for l in open(pred_path) if "error" not in json.loads(l)]
    return items, rows


def arm_of(intervention: str) -> str:
    if intervention == "none":
        return "original"
    if intervention.endswith("_relevant"):
        return "relevant"
    if intervention.endswith("_irrelevant"):
        return "irrelevant"
    return intervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--logit_key", default="lm_option_logits")
    ap.add_argument("--suff_key", default="conf",
                    choices=["conf", "answerable"],
                    help="which score plays the role of the sufficiency signal: "
                         "the decision confidence, or the trained answerability head")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items, rows = load(args.pred, args.items)
    by_base: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        if args.logit_key not in r:
            continue
        z = np.asarray(r[args.logit_key], dtype=float)
        p = np.exp(z - z.max())
        p /= p.sum()
        suff = float(p.max())
        if args.suff_key == "answerable":
            if "answerable_logit" not in r:
                continue
            suff = 1.0 / (1.0 + np.exp(-r["answerable_logit"]))
        rec = {
            "p": p,
            "conf": suff,
            "decision_conf": float(p.max()),
            "correct": int(np.argmax(p) == r["label"]),
            "entropy": float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()),
            "margin": float(np.sort(p)[-1] - np.sort(p)[-2]) if len(p) > 1 else 1.0,
            "label": r["label"],
            "task_family": r["task_family"],
            "mode": r["intervention"].replace("_relevant", "").replace("_irrelevant", ""),
        }
        by_base[r["base_id"]][arm_of(r["intervention"])] = rec

    triples = {b: d for b, d in by_base.items() if {"original", "relevant", "irrelevant"} <= set(d)}
    print(f"{len(by_base)} base judgments, {len(triples)} complete triples\n")

    def col(arm, key):
        return np.array([d[arm][key] for d in triples.values()], dtype=float)

    sig = "sufficiency" if args.suff_key == "answerable" else "confidence"
    print(f"{'arm':<12} {'accuracy':>9} {sig:>12} {'entropy':>9} {'margin':>8}")
    for arm in ("original", "relevant", "irrelevant"):
        print(f"{arm:<12} {col(arm,'correct').mean():>9.3f} {col(arm,'conf').mean():>12.3f} "
              f"{col(arm,'entropy').mean():>9.3f} {col(arm,'margin').mean():>8.3f}")

    acc_o, acc_r, acc_i = col("original", "correct"), col("relevant", "correct"), col("irrelevant", "correct")
    cf_o, cf_r, cf_i = col("original", "conf"), col("relevant", "conf"), col("irrelevant", "conf")

    print("\n--- the phenomenon the paper is about ---")
    print(f"accuracy drop, relevant arm     : {acc_o.mean()-acc_r.mean():+.3f}")
    print(f"accuracy drop, irrelevant arm   : {acc_o.mean()-acc_i.mean():+.3f}")
    print(f"{sig} drop, relevant arm   : {cf_o.mean()-cf_r.mean():+.3f}")
    print(f"{sig} drop, irrelevant arm : {cf_o.mean()-cf_i.mean():+.3f}")

    # Of the items the model got right with evidence and wrong without it,
    # how many does it still answer confidently?
    broke = (acc_o == 1) & (acc_r == 0)
    print(f"\nitems correct WITH evidence but wrong WITHOUT it: {int(broke.sum())} "
          f"({broke.mean():.1%} of triples)")
    if broke.any():
        print(f"  their mean confidence when wrong : {cf_r[broke].mean():.3f}")
        print(f"  fraction still above conf 0.9    : {(cf_r[broke] > 0.9).mean():.1%}")
        print(f"  fraction still above conf 0.99   : {(cf_r[broke] > 0.99).mean():.1%}")

    spec = M.intervention_specificity(cf_o, cf_r, cf_i)
    print("\n--- specificity of the confidence signal (E3 preview) ---")
    for k, v in spec.items():
        print(f"  {k:<26} {v:+.4f}")

    clusters = [items[f"{b}:orig"]["parent_image_id"] if f"{b}:orig" in items else b for b in triples]
    d_acc = M.cluster_bootstrap_ci(lambda ix: float(acc_o[ix].mean() - acc_r[ix].mean()), clusters, n_boot=500)
    d_cf = M.cluster_bootstrap_ci(lambda ix: float(cf_o[ix].mean() - cf_r[ix].mean()), clusters, n_boot=500)
    print(f"\naccuracy drop (relevant)   {d_acc[0]:+.3f}  95% CI [{d_acc[1]:+.3f}, {d_acc[2]:+.3f}]")
    print(f"confidence drop (relevant) {d_cf[0]:+.3f}  95% CI [{d_cf[1]:+.3f}, {d_cf[2]:+.3f}]")

    print("\n--- by degradation mode ---")
    by_mode = {}
    modes = sorted({d["relevant"]["mode"] for d in triples.values()})
    for mo in modes:
        ix = [i for i, d in enumerate(triples.values()) if d["relevant"]["mode"] == mo]
        if not ix:
            continue
        ix = np.asarray(ix)
        print(f"  {mo:<12} n={len(ix):<5} acc {acc_o[ix].mean():.3f}->{acc_r[ix].mean():.3f} (irr {acc_i[ix].mean():.3f})"
              f"   conf {cf_o[ix].mean():.3f}->{cf_r[ix].mean():.3f} (irr {cf_i[ix].mean():.3f})")
        # occlusion is the only degradation seen in training; blur and
        # downscale are the Intervention-OOD arm
        by_mode[mo] = {
            "n": int(len(ix)),
            "acc_orig": float(acc_o[ix].mean()), "acc_rel": float(acc_r[ix].mean()),
            "acc_irr": float(acc_i[ix].mean()),
            "suff_orig": float(cf_o[ix].mean()), "suff_rel": float(cf_r[ix].mean()),
            "suff_irr": float(cf_i[ix].mean()),
            "specificity": M.intervention_specificity(cf_o[ix], cf_r[ix], cf_i[ix]),
            "trained_on": mo == "occlude",
        }

    if args.out:
        res = {
            "n_triples": len(triples),
            "arms": {a: {k: float(col(a, k).mean()) for k in ("correct", "conf", "entropy", "margin")}
                     for a in ("original", "relevant", "irrelevant")},
            "acc_drop_relevant_ci": d_acc, "conf_drop_relevant_ci": d_cf,
            "specificity": spec,
            "by_mode": by_mode,
            "suff_key": args.suff_key, "logit_key": args.logit_key,
            "broke_n": int(broke.sum()),
            "broke_conf_mean": float(cf_r[broke].mean()) if broke.any() else None,
            "broke_frac_conf_gt_09": float((cf_r[broke] > 0.9).mean()) if broke.any() else None,
            "broke_frac_conf_gt_099": float((cf_r[broke] > 0.99).mean()) if broke.any() else None,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()

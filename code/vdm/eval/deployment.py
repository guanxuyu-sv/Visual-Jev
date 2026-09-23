"""Selective prediction on a deployment mix (plan 8.2, E2 and E7).

The in-distribution test split contains only intact observations, so a
sufficiency signal has nothing to do there.  The setting it exists for is the
one a deployed system actually sees: a stream in which some observations do
carry the evidence and some do not.

We build that stream from the intervention triples -- original, evidence
degraded, and the equal-area control -- and score every item against the
*original* gold label.  A model is free to answer a degraded item and will
sometimes be right by prior; what we measure is whether it can order the stream
so that the items it answers are the ones it gets right.

Thresholds, temperatures and gates are fitted on a calibration split of parent
images and applied unchanged, exactly as in report.py.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm.eval import metrics as M
from vdm.eval.report import apply_gate, calib_mask, fit_gate, load_pred

RISKS = (0.05, 0.10, 0.20)


def build(path: str, name: str):
    key = "lm_option_logits" if name.startswith(("B0", "B1")) else "choice_logits"
    d = load_pred(path, key)
    if d is None:
        d = load_pred(path, "lm_option_logits")
    if d is None:
        return None
    arms = {"none": "original"}
    keep = [i for i, iv in enumerate(d["intervention"])
            if iv == "none" or iv.endswith(("_relevant", "_irrelevant"))]
    # only base judgments for which all three arms survived
    from collections import defaultdict
    seen = defaultdict(set)
    for i in keep:
        iv = d["intervention"][i]
        arm = "original" if iv == "none" else ("relevant" if iv.endswith("_relevant") else "irrelevant")
        seen[d["base"][i]].add(arm)
    full = {b for b, s in seen.items() if len(s) == 3}
    idx = np.array([i for i in keep if d["base"][i] in full])
    return d, idx


def signals(d, idx, calib_idx, test_idx, use_gate: bool):
    """Return (conf_calib, correct_calib, conf_test, correct_test, label)."""
    T = M.fit_temperature(np.where(d["mask"][calib_idx], d["logits"][calib_idx], -1e4),
                          d["label"][calib_idx])
    def block(ix):
        p = M.softmax(d["logits"][ix], T, d["mask"][ix])
        correct = (p.argmax(1) == d["label"][ix]).astype(int)
        srt = np.sort(p, 1)[:, ::-1]
        feats = [srt[:, 0], srt[:, 0] - srt[:, 1],
                 -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)]
        if "ans_logit" in d:
            feats.append(1.0 / (1.0 + np.exp(-d["ans_logit"][ix])))
        return p, correct, np.stack(feats, 1)
    _, cc, Xc = block(calib_idx)
    _, ct, Xt = block(test_idx)
    if use_gate and "ans_logit" in d:
        g = fit_gate(Xc, cc)
        return apply_gate(g, Xc), cc, apply_gate(g, Xt), ct, T, g["w"]
    return Xc[:, 0], cc, Xt[:, 0], ct, T, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    args = ap.parse_args()

    out: dict[str, object] = {"variants": {}}
    store = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        b = build(path, name)
        if b is None:
            print(f"!! {name}: no rows"); continue
        d, idx = b
        cm = calib_mask([d["parent"][i] for i in idx])
        calib, test = idx[cm], idx[~cm]
        entry = {"n_calib": int(len(calib)), "n_test": int(len(test))}

        for tag, use_gate in (("maxprob", False), ("gate", True)):
            if use_gate and "ans_logit" not in d:
                continue
            cfc, cc, cft, ct, T, w = signals(d, idx, calib, test, use_gate)
            e = {"temperature": T, "gate_w": w,
                 "full_accuracy": float(ct.mean()),
                 "aurc": M.aurc(cft, ct),
                 # Ordering quality independent of how accurate the model is.
                 # AURC alone rewards a model for being right more often, which
                 # confounds "orders its stream well" with "is simply better"
                 # (plan 13.1 asks for AURC at matched full-sample accuracy).
                 "conf_auroc": M.auroc(cft, ct),
                 # Excess area over the oracle ordering for this same accuracy,
                 # i.e. how much of the achievable ordering gain is realised.
                 "aurc_oracle": M.aurc(ct.astype(float), ct),
                 "aurc_excess": M.aurc(cft, ct) - M.aurc(ct.astype(float), ct),
                 "at_risk": {}, "at_coverage": {c: M.selective_accuracy_at_coverage(cft, ct, c)
                                                for c in (0.3, 0.5, 0.7, 0.9)}}
            for r in RISKS:
                thr = M.threshold_for_risk(cfc, cc, r)
                e["at_risk"][r] = M.apply_threshold(cft, ct, thr)
            entry[tag] = e
            store[(name, tag)] = (cft, ct, [d["parent"][i] for i in test],
                                  [d["item_id"][i] for i in test])
        # accuracy per arm, so the mix is interpretable
        arm_acc = {}
        for arm, pred in (("original", lambda iv: iv == "none"),
                          ("relevant", lambda iv: iv.endswith("_relevant")),
                          ("irrelevant", lambda iv: iv.endswith("_irrelevant"))):
            sel = np.array([i for i in test if pred(d["intervention"][i])])
            if len(sel):
                p = M.softmax(d["logits"][sel], 1.0, d["mask"][sel])
                arm_acc[arm] = float((p.argmax(1) == d["label"][sel]).mean())
        entry["accuracy_by_arm"] = arm_acc
        out["variants"][name] = entry

        best = entry.get("gate") or entry["maxprob"]
        print(f"{name:<8} mix_acc={best['full_accuracy']:.4f} AURC={best['aurc']:.4f} "
              f"confAUROC={best['conf_auroc']:.4f} "
              f"cov@5%={best['at_risk'][0.05]['coverage']:.3f} "
              f"cov@10%={best['at_risk'][0.10]['coverage']:.3f}  arms={arm_acc}", flush=True)

    # paired comparisons on the mix
    print("\n=== paired (clustered bootstrap) ===")
    comps = []
    keys = list(store)
    for a in keys:
        for b in keys:
            if a >= b:
                continue
            (ca, ra, pa, ia), (cb, rb, pb, ib) = store[a], store[b]
            ma = {k: j for j, k in enumerate(ia)}
            mb = {k: j for j, k in enumerate(ib)}
            common = sorted(set(ma) & set(mb))
            if len(common) < 200:
                continue
            ja = np.array([ma[k] for k in common]); jb = np.array([mb[k] for k in common])
            cl = [pa[j] for j in ja]
            d_aurc = M.paired_cluster_bootstrap_delta(
                lambda ix: M.aurc(ca[ja][ix], ra[ja][ix]),
                lambda ix: M.aurc(cb[jb][ix], rb[jb][ix]), cl, n_boot=args.n_boot)
            comps.append({"a": list(a), "b": list(b), "delta_aurc": d_aurc, "n": len(common)})
            sig = "  *" if (d_aurc[1] > 0) or (d_aurc[2] < 0) else ""
            print(f"  {a[0]}/{a[1]:<8} - {b[0]}/{b[1]:<8} dAURC={d_aurc[0]:+.4f} "
                  f"[{d_aurc[1]:+.4f},{d_aurc[2]:+.4f}]{sig}", flush=True)
    out["paired"] = comps
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

"""Turn raw per-example predictions into the tables the paper reports.

Discipline enforced here rather than left to the writing:

* The calibration split is carved out by parent image and is used for the
  temperature, the gate and the risk threshold.  Nothing is fitted on test.
* Full-sample accuracy is always reported next to any selective number.
* Comparisons that the paper claims as gains get a *paired* clustered
  bootstrap CI on the difference, not two separate intervals (plan 13.1).
* ECE is printed but never used to decide anything; NLL, Brier and AURC are
  the proper scores.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm.eval import metrics as M

RISK_TARGETS = (0.05, 0.10)


def calib_mask(parent_ids: list[str], frac: float = 0.3, salt: str = "vdm-calib") -> np.ndarray:
    """Deterministic image-level calibration split, stable across variants."""
    out = np.zeros(len(parent_ids), dtype=bool)
    for i, p in enumerate(parent_ids):
        h = hashlib.sha1((salt + str(p)).encode()).hexdigest()
        out[i] = (int(h[:8], 16) % 1000) < frac * 1000
    return out


def load_pred(path: str, logit_key: str) -> dict[str, np.ndarray] | None:
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if "error" in r or logit_key not in r:
                continue
            rows.append(r)
    if not rows:
        return None
    K = max(len(r[logit_key]) for r in rows)
    Z = np.full((len(rows), K), -1e4, dtype=float)
    mask = np.zeros((len(rows), K), dtype=bool)
    for i, r in enumerate(rows):
        k = len(r[logit_key])
        Z[i, :k] = r[logit_key]
        mask[i, :k] = True
    d = {
        "logits": Z, "mask": mask,
        "label": np.array([r["label"] for r in rows]),
        "answerable": np.array([-1 if r["answerable"] is None else r["answerable"] for r in rows]),
        "parent": [r["parent_image_id"] for r in rows],
        "base": [r["base_id"] for r in rows],
        "family": [r["task_family"] for r in rows],
        "source": [r["source"] for r in rows],
        "intervention": [r["intervention"] for r in rows],
        "item_id": [r["item_id"] for r in rows],
        "n_options": np.array([r["n_options"] for r in rows]),
    }
    if "answerable_logit" in rows[0]:
        d["ans_logit"] = np.array([r.get("answerable_logit", np.nan) for r in rows])
    return d


def decision_quality(d: dict, idx: np.ndarray, T: float = 1.0) -> dict:
    p = M.softmax(d["logits"][idx], T, d["mask"][idx])
    y = d["label"][idx]
    return {
        "n": int(len(idx)),
        "accuracy": M.accuracy(p, y),
        "macro_f1": M.macro_f1(p, y),
        "nll": M.nll(p, y),
        "brier": M.brier(p, y),
        "ece": M.ece(p, y),
    }


def conf_features(d: dict, idx: np.ndarray, T: float = 1.0) -> np.ndarray:
    p = M.softmax(d["logits"][idx], T, d["mask"][idx])
    srt = np.sort(p, axis=1)[:, ::-1]
    maxp = srt[:, 0]
    margin = srt[:, 0] - srt[:, 1] if p.shape[1] > 1 else np.ones_like(maxp)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    feats = [maxp, margin, ent]
    if "ans_logit" in d:
        a = 1.0 / (1.0 + np.exp(-d["ans_logit"][idx]))
        feats.append(a)
    return np.stack(feats, axis=1)


def fit_gate(X: np.ndarray, correct: np.ndarray, iters: int = 800, lr: float = 0.5):
    """Tiny logistic regression on confidence features, fitted on calibration.

    Deliberately small and inspectable: the paper claims a *gate*, not a second
    model.  It is also the only place where answerability is allowed to combine
    with the decision distribution, because answerability alone estimates
    evidence sufficiency, not answer correctness (plan 7.4).
    """
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
    w = np.zeros(Xs.shape[1])
    y = correct.astype(float)
    for _ in range(iters):
        z = Xs @ w
        pr = 1.0 / (1.0 + np.exp(-z))
        g = Xs.T @ (pr - y) / len(y)
        w -= lr * g
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}


def apply_gate(gate: dict, X: np.ndarray) -> np.ndarray:
    mu, sd, w = np.array(gate["mu"]), np.array(gate["sd"]), np.array(gate["w"])
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    return 1.0 / (1.0 + np.exp(-(Xs @ w)))


def selective_block(d: dict, calib: np.ndarray, test: np.ndarray, T: float, use_gate: bool):
    p_te = M.softmax(d["logits"][test], T, d["mask"][test])
    correct_te = (p_te.argmax(1) == d["label"][test]).astype(int)
    p_ca = M.softmax(d["logits"][calib], T, d["mask"][calib])
    correct_ca = (p_ca.argmax(1) == d["label"][calib]).astype(int)

    if use_gate and "ans_logit" in d:
        gate = fit_gate(conf_features(d, calib, T), correct_ca)
        conf_ca = apply_gate(gate, conf_features(d, calib, T))
        conf_te = apply_gate(gate, conf_features(d, test, T))
        gate_desc = {"features": ["maxp", "margin", "entropy", "answerability"], "w": gate["w"]}
    else:
        conf_ca, conf_te, gate_desc = p_ca.max(1), p_te.max(1), None

    out = {
        "full_accuracy": float(correct_te.mean()),
        "aurc": M.aurc(conf_te, correct_te),
        "gate": gate_desc,
        "at_risk": {},
        "at_coverage": {c: M.selective_accuracy_at_coverage(conf_te, correct_te, c)
                        for c in (0.5, 0.7, 0.9)},
    }
    for tr in RISK_TARGETS:
        thr = M.threshold_for_risk(conf_ca, correct_ca, tr)     # fitted on calib only
        out["at_risk"][tr] = M.apply_threshold(conf_te, correct_te, thr)
    return out, conf_te, correct_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="NAME=path/to/pred.jsonl, NAME is the table row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=800)
    args = ap.parse_args()

    variants: dict[str, str] = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        variants[name] = path

    report: dict[str, object] = {"variants": {}}
    keep: dict[str, tuple] = {}

    for name, path in variants.items():
        # B0/B1 read the LM head; trained variants read the decision head.
        logit_key = "lm_option_logits" if name.startswith(("B0", "B1")) else "choice_logits"
        d = load_pred(path, logit_key)
        if d is None:
            d = load_pred(path, "lm_option_logits")
            logit_key = "lm_option_logits"
        if d is None:
            print(f"!! no usable rows in {path}")
            continue

        orig = np.array([i for i, iv in enumerate(d["intervention"]) if iv == "none"])
        ca_m = calib_mask([d["parent"][i] for i in orig])
        calib, test = orig[ca_m], orig[~ca_m]

        # temperature fitted on the calibration split, applied unchanged
        T = M.fit_temperature(np.where(d["mask"][calib], d["logits"][calib], -1e4), d["label"][calib])

        entry: dict[str, object] = {"logit_key": logit_key, "temperature": T,
                                    "n_calib": int(len(calib)), "n_test": int(len(test))}
        entry["quality_raw"] = decision_quality(d, test, 1.0)
        entry["quality_temp"] = decision_quality(d, test, T)

        # per task family, and grouped into the three distributions
        fam = collections.defaultdict(list)
        for i in test:
            fam[d["family"][i]].append(i)
        entry["by_family"] = {k: decision_quality(d, np.array(v), T) for k, v in fam.items()}

        sel_raw, cf_raw, co_raw = selective_block(d, calib, test, 1.0, False)
        sel_temp, cf_t, co_t = selective_block(d, calib, test, T, False)
        entry["selective_raw"] = sel_raw
        entry["selective_temp"] = sel_temp
        keep[name] = (d, calib, test, T, cf_t, co_t)
        if "ans_logit" in d:
            sel_gate, cf_g, co_g = selective_block(d, calib, test, T, True)
            entry["selective_gate"] = sel_gate
            keep[name] = (d, calib, test, T, cf_g, co_g)

            # answerability as its own task: does it detect removed evidence?
            has_ans = np.array([i for i, a in enumerate(d["answerable"]) if a in (0, 1)])
            if len(has_ans) > 0:
                s = d["ans_logit"][has_ans]
                y = d["answerable"][has_ans].astype(int)
                entry["answerability"] = {
                    "n": int(len(has_ans)),
                    "auroc": M.auroc(s, y), "auprc": M.auprc(s, y),
                    "pos_rate": float(y.mean()),
                }
        report["variants"][name] = entry
        q = entry["quality_temp"]
        print(f"{name:<10} acc={q['accuracy']:.4f} f1={q['macro_f1']:.4f} nll={q['nll']:.4f} "
              f"brier={q['brier']:.4f}  AURC={ (entry.get('selective_gate') or sel_temp)['aurc']:.4f}", flush=True)

    # ------------------- paired comparisons with CIs ------------------- #
    print("\n=== paired comparisons (clustered bootstrap on parent image) ===")
    comps = []
    names = list(keep)
    for a in names:
        for b in names:
            if a >= b:
                continue
            da, ca_, ta, Ta, cfa, coa = keep[a]
            db, cb_, tb, Tb, cfb, cob = keep[b]
            # align on item_id so the bootstrap really is paired
            ia = {da["item_id"][i]: j for j, i in enumerate(ta)}
            ib = {db["item_id"][i]: j for j, i in enumerate(tb)}
            common = sorted(set(ia) & set(ib))
            if len(common) < 100:
                continue
            ja = np.array([ia[k] for k in common])
            jb = np.array([ib[k] for k in common])
            clusters = [da["parent"][ta[j]] for j in ja]
            acc_a, acc_b = coa[ja], cob[jb]
            cfa_, cfb_ = cfa[ja], cfb[jb]
            d_acc = M.paired_cluster_bootstrap_delta(
                lambda ix: float(acc_a[ix].mean()), lambda ix: float(acc_b[ix].mean()),
                clusters, n_boot=args.n_boot)
            d_aurc = M.paired_cluster_bootstrap_delta(
                lambda ix: M.aurc(cfa_[ix], acc_a[ix]), lambda ix: M.aurc(cfb_[ix], acc_b[ix]),
                clusters, n_boot=args.n_boot)
            comps.append({"a": a, "b": b, "n_common": len(common),
                          "delta_accuracy": d_acc, "delta_aurc": d_aurc})
            print(f"  {a:<8} - {b:<8}  dAcc={d_acc[0]:+.4f} [{d_acc[1]:+.4f},{d_acc[2]:+.4f}]   "
                  f"dAURC={d_aurc[0]:+.4f} [{d_aurc[1]:+.4f},{d_aurc[2]:+.4f}]", flush=True)
    report["paired_comparisons"] = comps

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

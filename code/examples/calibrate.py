"""Turn option logits into accept-or-review decisions with a target risk.

The adapters' raw probabilities are overconfident, and a temperature fitted on
one task calibrates others poorly. This script fits the two numbers a
deployment needs on a few hundred labelled questions from *your* task, then
applies them unchanged to new questions:

    temperature  one T, softmax(logits / T), fitted by minimising NLL
    threshold    the lowest top probability at which the accepted questions'
                 error rate on the calibration data is at most the target

Input is JSON lines with one question each: {"logits": [...], "label": i},
where i indexes the correct option. `quickstart.py --save-logits` writes the
logits; add the labels yourself. Keep the option order fixed between fitting
and deployment: reordering the options changes some decisions.

    python code/examples/calibrate.py fit   --scores calib.jsonl --risk 0.05 --out calibration.json
    python code/examples/calibrate.py apply --scores new.jsonl --calibration calibration.json --risk 0.05

The thresholds are empirical: they held their targets on held-out data in our
measurements given several hundred calibration questions from the same task,
and missed them on a 160-question calibration set. They are not guarantees.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdm.eval import metrics as M


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    k = max(len(r["logits"]) for r in rows)
    z = np.full((len(rows), k), -1e4)                # padding gets no probability
    mask = np.zeros((len(rows), k), dtype=bool)
    for i, r in enumerate(rows):
        z[i, :len(r["logits"])] = r["logits"]
        mask[i, :len(r["logits"])] = True
    labels = np.array([r.get("label", -1) for r in rows])
    return rows, z, mask, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("--scores", required=True, help="JSON lines with logits (and labels, for fit)")
    ap.add_argument("--risk", type=float, nargs="+", default=[0.05], help="target error rate(s)")
    ap.add_argument("--out", default="calibration.json", help="fit: where to write T and thresholds")
    ap.add_argument("--calibration", help="apply: the file written by fit")
    a = ap.parse_args()
    rows, z, mask, y = load(a.scores)

    if a.mode == "fit":
        if (y < 0).any():
            sys.exit("fit needs a label for every question")
        t = M.fit_temperature(z, y)
        p = M.softmax(z, t, mask)
        conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
        res = {"temperature": t, "n_calibration": len(y), "thresholds": {}}
        for r in a.risk:
            thr = M.threshold_for_risk(conf, correct, r)
            res["thresholds"][str(r)] = {"threshold": thr, **M.apply_threshold(conf, correct, thr)}
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"T = {t:.3f} on {len(y)} questions")
        for r, v in res["thresholds"].items():
            print(f"  target risk {r}: threshold {v['threshold']:.3f}, accepts {v['coverage']:.1%}, "
                  f"calibration risk {v['selective_risk']:.3f}")
        print("wrote", a.out)
        return

    cal = json.load(open(a.calibration))
    p = M.softmax(z, cal["temperature"], mask)
    for r in a.risk:
        thr = cal["thresholds"][str(r)]["threshold"]
        accept = p.max(1) >= thr
        print(f"target risk {r}: accept {int(accept.sum())} of {len(rows)}, review the rest")
        if (y >= 0).all():
            print(f"  realized risk among accepted: {1 - (p.argmax(1) == y)[accept].mean():.3f}"
                  if accept.any() else "  nothing accepted")
    for row, pi in zip(rows, p):
        k = int(pi.argmax())
        row.update(answer=k, probability=float(pi[k]),
                   decision={str(r): "accept" if pi[k] >= cal["thresholds"][str(r)]["threshold"] else "review"
                             for r in a.risk})
        print(json.dumps({key: row[key] for key in ("question", "answer", "probability", "decision") if key in row}))


if __name__ == "__main__":
    main()

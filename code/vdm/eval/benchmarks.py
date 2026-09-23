"""Per-benchmark accuracy with sample counts and clustered confidence intervals.

The single mixed number the earlier draft reported cannot say which ability a
system has. This splits it, weights the benchmarks equally in the macro
average so the largest or easiest one cannot dominate, and records for every
column whether post-training ever saw that task.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.eval import metrics as M
from vdm.eval.report import calib_mask, load_pred

# name -> (prediction file stem, whether post-training saw this task)
BENCHMARKS = [
    ("GQA-Choice",     "gqa_val",      True),
    ("SNLI-VE",        "snli_ve",      True),
    ("TextVQA-Choice", "textvqa_k8",   False),
    ("TallyQA-Choice", "tallyqa",      False),
]


# Which output a system is actually read through. Getting this wrong scores a
# model on a head it never trained: the answer-SFT baseline has no decision
# head at all, and reading one gives a number that says nothing about it.
READOUT = {
    "B1": "lm_option_logits",     # untrained backbone, LM head
    "B1_8b": "lm_option_logits",  # same, larger backbone
    "A0": "lm_option_logits",
    "A1": "lm_option_logits",
    "a2": "lm_option_logits",     # answer SFT: trained through the LM head
    "a2_8b": "lm_option_logits",
}


def readout_key(name: str) -> str:
    stem = name.split("_s")[0]
    return READOUT.get(stem, READOUT.get(name, "choice_logits"))


def score_one(path: str, name: str, held_out_test_only: bool = True):
    """Accuracy on the locked test portion, with a per-image bootstrap CI."""
    key = readout_key(name)
    d = load_pred(path, key)
    if d is None:
        d = load_pred(path, "lm_option_logits")
    if d is None:
        return None
    orig = np.array([i for i, iv in enumerate(d["intervention"]) if iv == "none"])
    if len(orig) == 0:
        return None
    # same deterministic calibration split as everywhere else; report on test
    ca = calib_mask([d["parent"][i] for i in orig])
    idx = orig[~ca] if held_out_test_only else orig
    p = M.softmax(d["logits"][idx], 1.0, d["mask"][idx])
    correct = (p.argmax(1) == d["label"][idx]).astype(int)
    clusters = [d["parent"][i] for i in idx]
    pt, lo, hi = M.cluster_bootstrap_ci(
        lambda ix: float(correct[ix].mean()), clusters, n_boot=400)
    return {"accuracy": pt, "ci": [lo, hi], "n": int(len(idx)),
            "n_images": len(set(clusters))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_root", default=paths.PREDS)
    ap.add_argument("--variants", nargs="+", required=True,
                    help="NAME=dir, dir under preds_root holding <bench>.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res: dict[str, object] = {"benchmarks": [b[0] for b in BENCHMARKS],
                              "seen_in_training": {b[0]: b[2] for b in BENCHMARKS},
                              "variants": {}}
    for spec in args.variants:
        name, sub = spec.split("=", 1)
        per: dict[str, object] = {}
        for bench, stem, _seen in BENCHMARKS:
            f = os.path.join(args.preds_root, sub, f"{stem}.jsonl")
            r = score_one(f, name) if os.path.exists(f) else None
            per[bench] = r
        got = [v["accuracy"] for v in per.values() if v]
        per["macro_avg"] = float(np.mean(got)) if got else None
        per["n_benchmarks"] = len(got)
        res["variants"][name] = per
        per["readout"] = readout_key(name)
        cells = "  ".join(
            f"{b}={per[b]['accuracy']:.4f}(n={per[b]['n']})" if per[b] else f"{b}=--"
            for b, _, _ in BENCHMARKS)
        mac = f"{per['macro_avg']:.4f}" if per["macro_avg"] else "--"
        print(f"{name:<12} [{per['readout'].replace('_logits','')}] macro={mac}  {cells}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

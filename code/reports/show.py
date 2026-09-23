"""Console views of the result JSONs (no LaTeX, just for reading)."""
import argparse, json, os, sys
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdm import paths


def deploy(path):
    d = json.load(open(path))
    hdr = ("variant", "signal", "mixAcc", "AURC", "excAURC", "cAUROC", "cov@5%", "cov@10%")
    print("%-9s %-8s %7s %8s %8s %8s %8s %8s" % hdr)
    for n, v in d["variants"].items():
        for tag in ("maxprob", "gate"):
            if tag not in v:
                continue
            e = v[tag]
            print("%-9s %-8s %7.4f %8.4f %8.4f %8.4f %8.3f %8.3f" % (
                n, tag, e["full_accuracy"], e["aurc"],
                e.get("aurc_excess", float("nan")), e.get("conf_auroc", float("nan")),
                e["at_risk"]["0.05"]["coverage"], e["at_risk"]["0.1"]["coverage"]))
        a = v["accuracy_by_arm"]
        print("          arms: " + "  ".join(f"{k}={x:.3f}" for k, x in a.items()))


def report(path):
    d = json.load(open(path))
    print("%-9s %8s %8s %8s %8s %9s" % ("variant", "acc", "macroF1", "NLL", "Brier", "AURC"))
    for n, v in d["variants"].items():
        q = v["quality_temp"]
        sel = v.get("selective_gate") or v["selective_temp"]
        print("%-9s %8.4f %8.4f %8.4f %8.4f %9.4f" % (
            n, q["accuracy"], q["macro_f1"], q["nll"], q["brier"], sel["aurc"]))
        if "answerability" in v:
            a = v["answerability"]
            print("          answerability AUROC=%.4f AUPRC=%.4f (n=%d, pos=%.2f)" % (
                a["auroc"], a["auprc"], a["n"], a["pos_rate"]))
        fam = v.get("by_family", {})
        if fam:
            print("          by family: " + "  ".join(
                f"{k}={q2['accuracy']:.3f}(n={q2['n']})" for k, q2 in sorted(fam.items())))


def runs(root=paths.RUNS):
    """Final loss values and checkpoint presence for every training run."""
    import glob
    print("%-10s %6s %7s %7s %7s %7s %6s" % (
        "run", "steps", "dec", "ans", "rank", "cons", "ckpt"))
    for d in sorted(glob.glob(os.path.join(root, "*_s*/"))):
        n = os.path.basename(d.rstrip("/"))
        lg = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(lg):
            continue
        rows = [json.loads(l) for l in open(lg)]
        last = rows[-1]
        ok = os.path.exists(os.path.join(d, "heads.pt")) and os.path.isdir(os.path.join(d, "lora"))
        g = lambda k: ("%.4f" % last[k]) if k in last else "-"
        print("%-10s %6d %7s %7s %7s %7s %6s" % (
            n, last["step"], g("dec"), g("ans"), g("rank"), g("cons"), "yes" if ok else "NO"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["deploy", "report", "runs"])
    ap.add_argument("path", nargs="?", default=paths.RUNS)
    a = ap.parse_args()
    {"deploy": deploy, "report": report, "runs": runs}[a.kind](a.path)

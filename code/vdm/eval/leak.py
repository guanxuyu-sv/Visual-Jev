"""Candidate-set leakage diagnostic (plan 6.3).

Accuracy that survives replacing the image with a uniform grey field is
accuracy the option list gave away.  We report it as a ratio to the sighted
accuracy and against the chance rate implied by the option count, because a
four-way question with a strong language prior can look impressive while
carrying no visual information at all.
"""
from __future__ import annotations

import argparse, json, sys

import numpy as np


def score(path: str) -> tuple[float, float, int]:
    acc, chance, n = 0, 0.0, 0
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if "error" in r or "lm_option_logits" not in r:
                continue
            z = np.asarray(r["lm_option_logits"])
            acc += int(z.argmax() == r["label"])
            chance += 1.0 / max(1, r["n_options"])
            n += 1
    return acc / n, chance / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="NAME=sighted.jsonl:blind.jsonl (blind may be omitted)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"{'evaluation set':<26} {'chance':>7} {'blind':>8} {'sighted':>8} {'visual gain':>12}")
    res = {}
    for spec in args.pairs:
        name, paths = spec.split("=", 1)
        parts = paths.split(":")
        a, ch, n = score(parts[0])
        if len(parts) > 1 and parts[1]:
            b, _, _ = score(parts[1])
            # how much of the headroom above the blind score the image buys
            gain = (a - b) / max(1e-9, 1.0 - b)
            print(f"{name:<26} {ch:>7.3f} {b:>8.4f} {a:>8.4f} {gain:>11.1%}")
            res[name] = {"chance": ch, "blind": b, "sighted": a, "visual_gain": gain, "n": n}
        else:
            print(f"{name:<26} {ch:>7.3f} {'--':>8} {a:>8.4f} {'--':>12}")
            res[name] = {"chance": ch, "sighted": a, "n": n}
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()

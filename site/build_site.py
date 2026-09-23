"""Generate the project page from the same result files the paper is built from.

Every figure on the page is injected here rather than typed into the template,
so the site cannot drift from the runs on disk any more than the paper can. The
template holds the markup and `__TOKENS__` where numbers go.
"""
from __future__ import annotations

import argparse, json, os, shutil, statistics as st, sys


def load(reports: str):
    j = lambda n: json.load(open(os.path.join(reports, n)))
    return j("bench.json"), j("bench_8b.json"), j("benchmarks.json")


def build(reports: str, template: str, out: str, demos: str | None = None) -> None:
    B, B8, M = load(reports)
    sweep4 = [{"N": r["N"], "path": r["path"], "ms": round(r["per_question_ms"], 1),
               "gib": round(r["peak_gib"], 2)} for r in B["sweep"]]
    sweep8 = [{"N": r["N"], "path": r["path"], "ms": round(r["per_question_ms"], 1),
               "gib": round(r["peak_gib"], 2)} for r in B8["sweep"]]
    V = M["variants"]

    def agg(pfx, key):
        vals = []
        for k in V:
            if k.split("_s")[0] != pfx:
                continue
            e = V[k].get(key)
            if e is None:
                continue
            vals.append(e["accuracy"] if isinstance(e, dict) else float(e))
        return round(st.mean(vals), 4) if vals else None

    bench = {}
    for pfx in ("B1", "a2", "b2k", "mk", "b2", "B1_8b", "a2_8b"):
        row = {"macro": agg(pfx, "macro_avg"),
               "seeds": len([k for k in V if k.split("_s")[0] == pfx])}
        for b in M["benchmarks"]:
            row[b] = agg(pfx, b)
        if row["macro"] is not None:
            bench[pfx] = row

    ms = lambda sw, p, n: next(r["ms"] for r in sw if r["path"] == p and r["N"] == n)
    gib = lambda sw, p, n: next(r["gib"] for r in sw if r["path"] == p and r["N"] == n)
    nmax = max(r["N"] for r in sweep4)
    ind, shared = ms(sweep4, "independent", nmax), ms(sweep4, "prefix_share_batch", nmax)
    indb = ms(sweep4, "independent_batch", nmax)

    def delta(x):
        """A difference that came out exactly zero is reported as a word.

        Written as +0.000 it reads like a measurement that landed just barely
        on the positive side, which is a claim the number does not make; the
        signed form is kept for every value that is actually non-zero.
        """
        return "0 drop" if round(x, 3) == 0 else f"{x:+.3f}"

    facts = {
        "total": round(ind / shared, 1), "batching": round(ind / indb, 1),
        "sharing": round(indb / shared, 1), "ms_ind": ind, "ms_shared": shared,
        "qps_ind": round(1000 / ind), "qps_shared": round(1000 / shared),
        "ms_gen": ms(sweep4, "prefix_share_batch_gen", nmax),
        "ms_n1_ind": ms(sweep4, "independent", 1),
        "ms_n1_shared": ms(sweep4, "prefix_share_batch", 1),
        "head_delta": round(bench["b2k"]["macro"] - bench["a2"]["macro"], 4),
        "scale_delta": round(bench["a2_8b"]["macro"] - bench["a2"]["macro"], 4),
        "scale_ms": ms(sweep8, "prefix_share_batch", nmax),
        "scale_lat_pct": round((ms(sweep8, "prefix_share_batch", nmax) / shared - 1) * 100),
        "scale_mem_pct": round((gib(sweep8, "prefix_share_batch", nmax)
                                / gib(sweep4, "prefix_share_batch", nmax) - 1) * 100),
        "suff_delta": round(bench["mk"]["macro"] - bench["b2k"]["macro"], 4),
        "slot_delta": round(bench["b2"]["TextVQA-Choice"] - bench["b2k"]["TextVQA-Choice"], 4),
    }

    demo_blob = "{}"
    if demos and os.path.exists(demos):
        demo_blob = json.dumps(json.load(open(demos)), separators=(",", ":"))

    subs = {
        "__DEMOS__": demo_blob,
        "__DATA__": json.dumps({"sweep4": sweep4, "sweep8": sweep8, "bench": bench,
                                "seen": M["seen_in_training"], "facts": facts},
                               separators=(",", ":")),
        "__TOTAL__": str(facts["total"]),
        "__HEAD__": delta(facts["head_delta"]),
        "__SUFF__": delta(facts["suff_delta"]),
        "__SCALE__": delta(facts["scale_delta"]),
        "__SLOT__": delta(facts["slot_delta"]),
        "__SCALE_LAT__": str(facts["scale_lat_pct"]),
        "__SCALE_MEM__": str(facts["scale_mem_pct"]),
        "__MS_GEN__": str(facts["ms_gen"]),
        "__MS_SHARED__": str(facts["ms_shared"]),
        "__MS_N1_SHARED__": str(facts["ms_n1_shared"]),
        "__MS_N1_IND__": str(facts["ms_n1_ind"]),
        "__M_B1__": f'{bench["B1"]["macro"]:.3f}',
        "__M_B2K__": f'{bench["b2k"]["macro"]:.3f}',
        "__M_A2__": f'{bench["a2"]["macro"]:.3f}',
    }
    html = open(template).read()
    for k, v in subs.items():
        html = html.replace(k, v)
    left = [t for t in subs if t in html]
    if left:
        raise SystemExit(f"unsubstituted tokens remain: {left}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(html)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures = os.path.join(repo_root, "assets", "figures")
    for asset in ("multi_question_inference.gif", "multi_question_inference.png"):
        shutil.copy2(os.path.join(figures, asset), os.path.join(os.path.dirname(out) or ".", asset))
    print(f"wrote {out} ({len(html)} bytes)")
    print(f"  {facts['total']}x = {facts['batching']}x batching "
          f"x {facts['sharing']}x sharing | head {facts['head_delta']:+.3f}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=os.path.join(here, "..", "reports", "results"))
    ap.add_argument("--template", default=os.path.join(here, "template.html"))
    # docs/ is what GitHub Pages serves with no further configuration
    ap.add_argument("--out", default=os.path.join(here, "..", "docs", "index.html"))
    ap.add_argument("--demos", default=os.path.join(here, "..", "reports", "results", "demos.json"),
                    help="real predictions and images for the demo sections")
    a = ap.parse_args()
    build(a.reports, a.template, a.out, a.demos)

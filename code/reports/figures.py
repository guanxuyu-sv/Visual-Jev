"""Main-text figures, drawn from the same raw predictions as the tables.

Academic-figure constraints drive the styling: column width, 8pt type to match
the body text, vector output, and no decoration that does not carry data.

Two rules from the viz method matter here and are followed rather than
approximated. Categorical hues are assigned in fixed slot order and never
cycled; and because three of the slots sit below 3:1 against a white surface,
every series is *directly labelled* rather than relying on a legend box -- the
relief the contrast check requires. Line dashes carry the same identity as the
hue, so the figures survive greyscale printing.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdm import paths
from vdm.eval import metrics as M
from vdm.eval.deployment import build
from vdm.eval.report import apply_gate, calib_mask, fit_gate

# categorical slots, fixed order, from the validated reference palette
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DASH = [(None, None), (5, 1.6), (1.6, 1.4), (6, 1.5, 1.5, 1.5), (3, 1.2), (8, 2)]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

COL_W, FULL_W = 3.15, 6.3      # ACL column and text width, inches


def style():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": GRID, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def place_labels(ax, points, fontsize=7, log_y=False, min_gap_frac=0.055):
    """Direct labels at line ends, pushed apart when curves converge.

    Series that end on top of each other are exactly the ones a reader most
    needs told apart, so the labels are separated rather than overprinted.
    """
    if not points:
        return
    lo, hi = ax.get_ylim()
    if log_y:
        tf, inv = (lambda v: np.log10(max(v, 1e-9))), (lambda v: 10 ** v)
        lo_t, hi_t = np.log10(lo), np.log10(hi)
    else:
        tf, inv = (lambda v: v), (lambda v: v)
        lo_t, hi_t = lo, hi
    gap = (hi_t - lo_t) * min_gap_frac

    order = sorted(range(len(points)), key=lambda i: tf(points[i][1]))
    ys = [tf(points[i][1]) for i in order]
    for k in range(1, len(ys)):                     # push up
        if ys[k] - ys[k - 1] < gap:
            ys[k] = ys[k - 1] + gap
    overflow = ys[-1] - hi_t
    if overflow > 0:                                 # then slide the block back down
        ys = [y - overflow for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            if ys[k + 1] - ys[k] < gap:
                ys[k] = ys[k + 1] - gap
    for slot, k in enumerate(order):
        x, _, name, colour = points[k]
        ax.annotate(name, xy=(x, inv(ys[slot])), xytext=(3, 0),
                    textcoords="offset points", color=colour,
                    fontsize=fontsize, va="center", ha="left",
                    annotation_clip=False)


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
def curves_for(pred_path: str, name: str):
    """Confidence and correctness on the deployment-mix test split."""
    b = build(pred_path, name)
    if b is None:
        return None
    d, idx = b
    cm = calib_mask([d["parent"][i] for i in idx])
    calib, test = idx[cm], idx[~cm]
    T = M.fit_temperature(np.where(d["mask"][calib], d["logits"][calib], -1e4), d["label"][calib])

    def block(ix):
        p = M.softmax(d["logits"][ix], T, d["mask"][ix])
        correct = (p.argmax(1) == d["label"][ix]).astype(int)
        srt = np.sort(p, 1)[:, ::-1]
        feats = [srt[:, 0], srt[:, 0] - srt[:, 1], -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)]
        if "ans_logit" in d:
            feats.append(1.0 / (1.0 + np.exp(-d["ans_logit"][ix])))
        return correct, np.stack(feats, 1)

    cc, Xc = block(calib)
    ct, Xt = block(test)
    if "ans_logit" in d:
        g = fit_gate(Xc, cc)
        conf = apply_gate(g, Xt)
    else:
        conf = Xt[:, 0]
    return conf, ct, d, test


def fig_risk_coverage(preds: dict[str, str], out: str):
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    end = []
    for k, (name, path) in enumerate(preds.items()):
        r = curves_for(path, name)
        if r is None:
            continue
        conf, correct, _, _ = r
        cov, risk = M.risk_coverage(conf, correct)
        keep = cov >= 0.05                      # the far-left tail is noise
        ax.plot(cov[keep], risk[keep], color=SLOT[k], lw=1.2,
                dashes=DASH[k] if DASH[k][0] else (), zorder=3)
        end.append((cov[-1], risk[-1], name, SLOT[k]))
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_xlim(0.05, 1.28)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    place_labels(ax, end, fontsize=7)
    despine(ax)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_sweep(bench_path: str, out: str):
    """Latency against concurrency, with the 2x3 design encoded in the marks.

    Hue says what a path reuses across the N questions; dash says whether it
    runs them as one batch. That composite encoding is what lets the reader see
    the decomposition -- vertical distance within a hue is batching, distance
    between hues at the same dash is sharing -- instead of eight arbitrary
    colours they have to cross-reference against a legend.
    """
    sw = json.load(open(bench_path))["sweep"]
    # hue = what is reused, dash = batched or not
    FAM = [("independent", SLOT[0]), ("vision_cache", SLOT[2]), ("prefix_share", SLOT[1])]
    NAME = {"independent": "independent", "vision_cache": "vision cache",
            "prefix_share": "prefix share"}
    GREY = "#8a8985"

    fig, ax = plt.subplots(figsize=(FULL_W * 0.52, 2.5))
    ends = []
    for base, colour in FAM:
        for batched in (False, True):
            key = base + ("_batch" if batched else "")
            pts = sorted([(r["N"], r["per_question_ms"]) for r in sw if r["path"] == key])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colour, lw=1.5 if batched else 1.1,
                    dashes=() if batched else (2.4, 1.6),
                    marker="o" if batched else "s", ms=3.0 if batched else 2.2,
                    mfc=colour if batched else "white", mew=0.8, zorder=3)
            ends.append((xs[-1], ys[-1],
                         NAME[base] + (", batched" if batched else ""), colour))
    for key, lbl in (("gen_one_token_each", "generate, 1 token"),
                     ("gen_compact_joint", "generate, joint")):
        pts = sorted([(r["N"], r["per_question_ms"]) for r in sw if r["path"] == key])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=GREY, lw=0.9, dashes=(1.2, 1.4), zorder=2)
        ends.append((xs[-1], ys[-1], lbl, GREY))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("questions sharing one image ($N$)")
    ax.set_ylabel("amortized time / question (ms)")
    ax.set_xlim(0.9, 260)
    place_labels(ax, ends, fontsize=6.2, log_y=True, min_gap_frac=0.062)
    # a second key for the dash, since hue alone does not carry "batched"
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.1, dashes=(2.4, 1.6),
                              marker="s", ms=2.2, mfc="white", label="serial"),
                       Line2D([], [], color=INK2, lw=1.5, marker="o", ms=3.0, label="batched")],
              loc="lower left", frameon=False, handlelength=2.2,
              borderpad=0.1, labelspacing=0.25)
    despine(ax)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_sufficiency(triples_path: str, out: str):
    """Where the two signals put each arm of the same judgement."""
    ta = json.load(open(triples_path))
    panels = [("B1_conf", "backbone confidence"), ("m_s0", "trained sufficiency head")]
    arms = [("original", "original"), ("irrelevant", "control"), ("relevant", "evidence removed")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.72, 1.85), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        res = ta.get(key)
        if res is None:
            continue
        for j, (arm, lbl) in enumerate(arms):
            v = res["arms"][arm]["conf"]
            ax.barh(j, v, height=0.52, color=SLOT[j], zorder=3)
            ax.annotate(f"{v:.3f}", xy=(v, j), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([l for _, l in arms])
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, color=INK2, pad=4)
        ax.grid(axis="y", visible=False)
        despine(ax)
    axes[0].set_xlabel("mean score")
    axes[1].set_xlabel("mean score")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)




def fig_frontier(bench_path: str, benchmarks_path: str, out: str,
                 bench_8b_path: str | None = None):
    """Quality against cost, for both backbones.

    Vertical position is what training or scale buys; horizontal position is
    what the execution path buys. Drawing them together is the claim: the
    horizontal run is free and an order of magnitude long, while doubling the
    parameters moves a fraction as far and costs latency and memory to do it.
    A backbone is one hue; answer-SFT is filled and the original backbone is hollow.
    """
    bm = json.load(open(benchmarks_path))["variants"]

    def sweep_of(path_json):
        return json.load(open(path_json))["sweep"] if path_json else None

    sw4, sw8 = sweep_of(bench_path), sweep_of(bench_8b_path)
    nmax = max(r["N"] for r in sw4)

    def lat(sw, path, n):
        r = next((x for x in sw if x["N"] == n and x["path"] == path), None)
        return r["per_question_ms"] if r else None

    def macro(prefix):
        v = [bm[k]["macro_avg"] for k in bm
             if k.split("_s")[0] == prefix and bm[k].get("macro_avg")]
        return float(np.mean(v)) if v else None

    paths = [("independent", "independent"), ("vision_cache", "vision\ncache"),
             ("prefix_share_batch", "shared +\nbatched")]
    families = [(sw4, SLOT[0], "4B", [("B1", False, "original backbone"), ("a2", True, "answer SFT")]),
                (sw8, SLOT[1], "8B", [("B1_8b", False, "original backbone"), ("a2_8b", True, "answer SFT")])]
    families = [f for f in families if f[0]]

    ys = [macro(k) for _, _, _, lv in families for k, _, _ in lv]
    ys = [y for y in ys if y]
    span = max(ys) - min(ys)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.90, 2.75), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.90, wspace=0.48)
    for ax, n in zip(axes, [1, nmax]):
        xticks = []
        for sw, colour, fam, levels in families:
            xs = [(lat(sw, p, n), lbl) for p, lbl in paths]
            xs = [(x, l) for x, l in xs if x]
            xticks += xs
            for key, filled, lvl in levels:
                y = macro(key)
                if y is None:
                    continue
                gx = [x for x, _ in xs]
                ax.plot(gx, [y] * len(gx), color=colour, lw=1.0, alpha=0.4, zorder=2)
                ax.scatter(gx, [y] * len(gx), s=22, zorder=3, linewidth=0.9,
                           facecolor=colour if filled else "white", edgecolor=colour)
                ax.annotate(f"{fam} {lvl}", xy=(max(gx), y), xytext=(-4, 3),
                            textcoords="offset points", color=colour, fontsize=5.9,
                            ha="right")
        ax.set_xscale("log")
        ax.set_title(f"$N{{=}}{n}$", color=INK2, pad=3)
        lo = min(x for x, _ in xticks); hi = max(x for x, _ in xticks)
        ax.set_xlim(lo * 0.5, hi * 2.6)
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        seen = set()
        for rank, (x, lbl) in enumerate(sorted(xticks)):
            if lbl in seen:
                continue
            seen.add(lbl)
            dx, dy, ha = {
                "independent": (-3, -11, "right"),
                "vision\ncache": (0, -29, "center"),
                "shared +\nbatched": (3, -11, "left"),
            }[lbl]
            ax.annotate(lbl, xy=(x, min(ys)), xytext=(dx, dy),
                        textcoords="offset points", color=INK2, fontsize=5.6,
                        ha=ha, va="top", linespacing=0.95)
            ax.plot([x, x], [min(ys), min(ys) - span * 0.06], color=GRID,
                    lw=0.5, zorder=1, clip_on=False)
        despine(ax)

    axes[0].set_ylim(min(ys) - span * 0.85, max(ys) + span * 0.30)
    axes[0].set_ylabel("macro accuracy")
    fig.text(0.54, 0.035, "amortized time / question (ms)",
             ha="center", va="bottom", fontsize=8, color=INK)

    a = axes[1]
    x_hi = max(lat(sw4, "independent", nmax), 1)
    x_lo = max(lat(sw4, "prefix_share_batch", nmax), 0.1)
    y_arrow = max(ys) + span * 0.13
    a.annotate("", xy=(x_lo, y_arrow), xytext=(x_hi, y_arrow),
               arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=2, shrinkB=2))
    a.annotate("execution: 8.9$\\times$ lower time", xy=((x_lo * x_hi) ** 0.5, y_arrow), xytext=(0, 3),
               textcoords="offset points", color=INK2, fontsize=6.2, ha="center")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--preds_root", default=paths.PREDS)
    ap.add_argument("--bench", default=paths.under("reports", "bench.json"))
    ap.add_argument("--triples", default=paths.under("reports", "triples_all.json"))
    ap.add_argument("--benchmarks", default=paths.under("reports", "benchmarks.json"))
    ap.add_argument("--bench_8b", default=paths.under("reports", "bench_8b.json"))
    ap.add_argument("--figures", nargs="+", default=["risk_coverage", "sweep", "sufficiency", "frontier"],
                    choices=["risk_coverage", "sweep", "sufficiency", "frontier"],
                    help="select figures when only some source files are available")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    style()

    P = a.preds_root
    if "risk_coverage" in a.figures:
        fig_risk_coverage({"B1": f"{P}/B1/gqa_val.jsonl",
                           "B2": f"{P}/b2_s0/gqa_val.jsonl",
                           "B5": f"{P}/b5_s0/gqa_val.jsonl",
                           "M": f"{P}/m_s0/gqa_val.jsonl"},
                          os.path.join(a.out_dir, "fig_riskcoverage.pdf"))
    if "sweep" in a.figures:
        fig_sweep(a.bench, os.path.join(a.out_dir, "fig_sweep.pdf"))
    if "sufficiency" in a.figures:
        fig_sufficiency(a.triples, os.path.join(a.out_dir, "fig_sufficiency.pdf"))
    if "frontier" in a.figures and os.path.exists(a.benchmarks):
        fig_frontier(a.bench, a.benchmarks, os.path.join(a.out_dir, "fig_frontier.pdf"),
                     a.bench_8b if os.path.exists(a.bench_8b) else None)


if __name__ == "__main__":
    main()

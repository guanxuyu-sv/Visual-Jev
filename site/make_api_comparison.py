"""Render the paper's 4B/API comparison for the README and project page.

Run from the repository root:
    uv run --with matplotlib python site/make_api_comparison.py
All plotted scores and headline gains come from data/api_comparison.json.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "data/api_comparison.json").read_text())
    systems = {s["key"]: s for s in data["systems"]}
    keys = ["M1", "API_jev", "API_luna"]
    names = ["Visual Jev 4B", "TypeSafe Jev", "OpenAI Decisions"]
    colors = ["#2563EB", "#8298AE", "#A3B1BF"]
    # Official marks are kept intact, including their native proportions.
    logos = [np.asarray(Image.open(root / "assets/brand" / name).convert("RGBA"))
             for name in ("visual-jev.png", "typesafe.png", "openai-black.png")]

    def row_logo(ax, index, y, size=13):
        bitmap = logos[index]
        # OpenAI's supplied canvas includes 25% clear space on every side.
        factor = 2 if index == 2 else 1
        box = OffsetImage(bitmap, zoom=size * factor / max(bitmap.shape[:2]))
        ax.add_artist(AnnotationBbox(box, (0, y), frameon=False,
                                   box_alignment=(.25 if index == 2 else 0, .5), pad=0))
    ink, muted = "#13263C", "#526479"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "visual-jev-api-comparison"})
    fig = plt.figure(figsize=(13.6, 7.4), facecolor="#FFFFFF")
    fig.text(.04, .935, "A 4B model ahead of hosted decision APIs", fontsize=25,
             weight="bold", color=ink)
    fig.text(.04, .89, "Same held-out questions. Four benchmarks. Equal-weight macro averages.",
             fontsize=12, color=muted)
    gains = data["headline_comparison"]["gains_pp"]
    cards = [(f'{100 * systems["M1"]["accuracy"]["macro"]:.1f}%', "Visual Jev 4B accuracy"),
             (f'+{gains["API_jev"]:.2f} pp', "accuracy vs. TypeSafe Jev"),
             (f'+{gains["API_luna"]:.2f} pp', "accuracy vs. OpenAI Decisions")]
    for x, (value, label) in zip((.04, .37, .70), cards):
        fig.text(x, .77, value, fontsize=32, weight="bold", color=colors[0])
        fig.text(x, .728, label, fontsize=11, color=muted)

    panels = [("Macro accuracy", "Higher is better", 1,
               [systems[k]["accuracy"]["macro"] for k in keys], True),
              ("Error-discrimination AUROC", "Higher is better", 1,
               [systems[k]["auroc"] for k in keys], False),
              ("Negative log-likelihood", "Lower is better", .5,
               [systems[k]["nll_ts"] for k in keys], False)]
    for x, (title, direction, limit, values, pct) in zip((.04, .37, .70), panels):
        ax = fig.add_axes((x, .30, .265, .295))
        ax.set_title(title + "\n" + direction, loc="left", fontsize=11,
                     color=ink, pad=20, linespacing=1.7, weight="medium")
        ax.barh([2, 1, 0], values, color=colors, height=.29, zorder=3)
        for index, (y, name, value) in enumerate(zip([2, 1, 0], names, values)):
            row_logo(ax, index, y + .30)
            ax.text(limit * .095, y + .23, name, fontsize=10, color=ink)
            ax.text(value + limit * .025, y,
                    f"{100 * value:.1f}%" if pct else f"{value:.3f}",
                    va="center", fontsize=11, weight="bold" if y == 2 else "normal",
                    color=colors[0] if y == 2 else ink)
        ax.set_xlim(0, limit * 1.19)
        ax.set_ylim(-.36, 2.6)
        ticks = [0, limit / 2, limit]
        ax.set_xticks(ticks, [f"{100 * v:.0f}%" if pct else f"{v:g}" for v in ticks],
                      fontsize=9, color=muted)
        ax.set_yticks([])
        ax.xaxis.grid(True, color="#E9ECEA", linewidth=.7, zorder=0)
        ax.tick_params(axis="both", length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    footnotes = [
        "Source: Visual Jev, Sec. 5.3 / Table 4 (arXiv:2609.25845). November 2026 API snapshot.",
        "AUROC and NLL use task-local temperature scaling; failed API requests receive uniform probabilities.",
        "Visual Jev uses task SFT; API training data are unknown. Jev has the lowest ECE (0.013 vs. 0.025 for 4B).",
        "OpenAI: GPT-6 Luna Decisions via OpenRouter. TypeSafe: jev-1.14.0, with image input marked pre-production.",
    ]
    for i, line in enumerate(footnotes):
        fig.text(.04, .19 - i * .032, line, fontsize=9, color=muted)
    out = root / "assets/figures/api_comparison"
    fig.savefig(out.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    fig.savefig(out.with_suffix(".png"), dpi=180, facecolor="white")
    plt.close(fig)

    # A vertical arrangement keeps chart labels readable on phones.
    fig = plt.figure(figsize=(5, 11.7), facecolor="white")
    fig.text(.08, .96, "A 4B model ahead of\nhosted decision APIs", fontsize=20,
             weight="bold", color=ink, va="top")
    fig.text(.08, .87, f'{100 * systems["M1"]["accuracy"]["macro"]:.1f}% macro accuracy\n'
             f'+{gains["API_jev"]:.2f} pp vs. Jev · +{gains["API_luna"]:.2f} pp vs. OpenAI', fontsize=12,
             color=colors[0], linespacing=1.6)
    for bottom, (title, direction, limit, values, pct) in zip((.62, .38, .14), panels):
        ax = fig.add_axes((.08, bottom, .82, .17))
        ax.set_title(title + " · " + direction.lower(), loc="left", fontsize=11, pad=18)
        ax.barh([2, 1, 0], values, color=colors, height=.26)
        for index, (y, name, value) in enumerate(zip([2, 1, 0], names, values)):
            row_logo(ax, index, y + .33, size=14)
            ax.text(limit * .09, y + .24, name, fontsize=11, color=ink)
            ax.text(value + limit * .025, y, f"{100 * value:.1f}%" if pct else f"{value:.3f}",
                    va="center", fontsize=11, color=colors[0] if y == 2 else ink)
        ax.set_xlim(0, limit * 1.2)
        ax.set_ylim(-.35, 2.7)
        ax.set_yticks([])
        ticks = [0, limit / 2, limit]
        ax.set_xticks(ticks, [f"{100 * v:.0f}%" if pct else f"{v:g}" for v in ticks],
                      fontsize=9, color=muted)
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.08, .075, "Paper Sec. 5.3 / Table 4 · November 2026\nSame held-out questions; equal-weight macro.\nNLL / AUROC after task-local calibration.\nTask SFT vs. APIs with unknown training data.\nJev leads on ECE. See protocol below.",
             fontsize=9, color=muted, va="top", linespacing=1.5)
    fig.savefig(out.with_name("api_comparison_mobile.svg"), facecolor="white", metadata={"Date": None})
    plt.close(fig)
    print(f"Wrote {out}.svg, .png and mobile SVG")


if __name__ == "__main__":
    main()

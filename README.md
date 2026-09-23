# Visual Jev — project page

Source for the project page. One image, one public context, many runtime-defined
questions, each answered independently — and what that workload costs to serve.

**Live page:** https://guanxuyu-sv.github.io/Visual-Jev/

## What it shows

- **One image, six questions.** Held-out GQA images with the questions that
  naturally occur on them, and the probabilities the model actually returned.
- **Confident without the evidence.** The same judgement under three
  conditions — original, the question's evidence region destroyed, and a
  control with an equal-area region elsewhere destroyed. Confidence stays above
  0.6 where the answer has already become wrong; the trained sufficiency output
  collapses instead.
- **Latency explorer.** The measured sweep: pick a question count and a
  backbone, and all eight execution paths redraw from the real numbers.
- **What didn't work.** Four results that did not go our way, reported because
  each changes what a reader should do next.

## Nothing on the page is typed by hand

Every figure is injected from `data/*.json`, which are the same result files
the paper's tables are computed from. `docs/index.html` is generated output —
edit `site/template.html` for markup, never the built file.

```bash
python3 site/build_site.py --reports data --demos data/demos.json
```

`site/make_demos.py` builds `data/demos.json` from the raw per-example
predictions and the images the model was shown; it needs the full evaluation
tree, which is not in this repo.

## Reproducing the experiments

`code/` holds the training, evaluation and reporting code, and
[REPRODUCE.md](REPRODUCE.md) walks through it end to end: building the question
records from the source corpora, training a system, predicting on the four
evaluation sets, scoring, and regenerating the tables and figures. Paths are
driven by one environment variable, so nothing points at the machine this ran
on.

The scored outputs are in `data/`, which means the numbers can be checked
without a GPU -- the page and the paper's tables are both computed from those
files.

## Layout

```
docs/index.html     generated, served by GitHub Pages from /docs
site/template.html  markup and copy, with __TOKENS__ where numbers go
site/build_site.py  injects every number from data/
site/make_demos.py  builds the demo payload from raw predictions
data/               the scored results the page and the tables are built from
code/vdm/           the package: data building, training, evaluation
code/reports/       table and figure generation
code/requirements.txt
REPRODUCE.md        the walkthrough
```

## Status

The paper is under review; this repo carries the page and the numbers behind
it, not the manuscript. Results here are from a single-machine study on
consumer GPUs and are reported with their seed spread and sample sizes — see
the page for what is held out from training and what is not.

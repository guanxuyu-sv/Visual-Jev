"""Every filesystem default in one place, driven by one environment variable.

The defaults used to name the machine this study ran on, which is fine until
somebody else clones the repository and finds that every command points at a
disk they do not have.  Now a single `VDM_ROOT` relocates the whole pipeline,
and `VDM_MODEL` picks the backbone -- by default the public Hub id rather than
a local snapshot, so a fresh checkout runs without editing anything.

    export VDM_ROOT=/scratch/vdm      # data, work, runs, preds, reports
    export VDM_MODEL=Qwen/Qwen3-VL-4B-Instruct

The layout underneath ROOT is the one the scripts and the paper's table
generator assume:

    $VDM_ROOT/data     source corpora as downloaded
    $VDM_ROOT/work     derived question records
    $VDM_ROOT/runs     training outputs (lora/, heads.pt, config.json)
    $VDM_ROOT/preds    per-example predictions, one directory per system
    $VDM_ROOT/reports  scored results the tables are built from
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.environ.get("VDM_ROOT", "work"))
MODEL = os.environ.get("VDM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
MODEL_8B = os.environ.get("VDM_MODEL_8B", "Qwen/Qwen3-VL-8B-Instruct")


def under(*parts: str) -> str:
    """A path under ROOT, e.g. under("work", "gqa") -> $VDM_ROOT/work/gqa."""
    return os.path.join(ROOT, *parts)


DATA = under("data")
WORK = under("work")
RUNS = under("runs")
PREDS = under("preds")
REPORTS = under("reports")

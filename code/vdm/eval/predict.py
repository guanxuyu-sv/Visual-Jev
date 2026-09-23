"""Run a model over an item file and save raw per-example predictions.

Raw logits are saved, never just the argmax: every table in the paper is
regenerated from these files, so calibration, selective prediction and the
intervention analysis never require a re-run (plan 11.2).

Grouping by image is not an optimisation here -- it is what the model is *for*.
Items sharing an image and a shared context are answered from one visual
encoding, which is also the setting the latency benchmark measures.
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.data.schema import read_jsonl
from vdm.models.vdm_model import VDM


BLIND = [False]

PATHS = {
    "independent": "run_independent",
    "independent_batch": "run_independent_batch",
    "vision_cache": "run_vision_cache",
    "vision_cache_batch": "run_vision_cache_batch",
    "prefix_share": "run_prefix_share",
    "prefix_share_batch": "run_prefix_share_batch",
}


def group_items(items: list[dict], max_group: int) -> list[list[dict]]:
    by_key: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for it in items:
        by_key[(it["image_path"], it["shared_context"])].append(it)
    groups = []
    for _, its in by_key.items():
        for i in range(0, len(its), max_group):
            groups.append(its[i : i + max_group])
    return groups


@torch.no_grad()
def run(model: VDM, items: list[dict], path: str, max_group: int, log_every: int = 50):
    fn = getattr(model, PATHS[path])
    groups = group_items(items, max_group)
    out_rows = []
    t0 = time.time()
    for gi, grp in enumerate(groups):
        if gi % log_every == 0:
            print(f"  group {gi}/{len(groups)}  ({time.time()-t0:.0f}s)", flush=True)
        try:
            with Image.open(grp[0]["image_path"]) as im:
                img = im.convert("RGB")
                if BLIND[0]:
                    img = Image.new("RGB", img.size, (128, 128, 128))
                qs = [
                    {
                        "qtype": it["qtype"],
                        "instruction": it["instruction"],
                        "candidates": it["candidates"],
                    }
                    for it in grp
                ]
                g = model.prepare_group(img, qs, grp[0]["shared_context"])
                o = fn(g)
        except Exception as e:                      # keep going, record the failure
            for it in grp:
                out_rows.append({"item_id": it["item_id"], "error": repr(e)})
            continue
        for i, it in enumerate(grp):
            k = g.n_options[i]
            row = {
                "item_id": it["item_id"],
                "base_id": it["base_id"],
                "parent_image_id": it["parent_image_id"],
                "source": it["source"],
                "task_family": it["task_family"],
                "qtype": it["qtype"],
                "intervention": it["intervention"],
                "label": it["label"],
                "answerable": it["answerable"],
                "n_options": k,
                "lm_option_logits": o["lm_option_logits"][i, :k].tolist(),
            }
            if "choice" in o:
                row["choice_logits"] = o["choice"][i, :k].tolist()
                row["claim_logits"] = o["claim"][i].tolist()
                row["answerable_logit"] = float(o["answerable"][i])
            out_rows.append(row)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=paths.MODEL)
    ap.add_argument("--ckpt", default=None, help="trained head+LoRA checkpoint")
    ap.add_argument("--path", default="vision_cache", choices=list(PATHS))
    ap.add_argument("--max_group", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter_split", default=None)
    ap.add_argument("--blind", action="store_true",
                    help="replace every image with a uniform grey field. Accuracy that "
                         "survives this is accuracy the candidate set gave away, not "
                         "accuracy the model read off the image (plan 6.3).")
    args = ap.parse_args()

    items = read_jsonl(args.items)
    if args.filter_split:
        items = [it for it in items if it["meta"].get("split") == args.filter_split]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items", flush=True)

    m = VDM(args.model)
    if args.ckpt:
        from vdm.training.checkpoint import load_into
        load_into(m, args.ckpt)
        print(f"loaded checkpoint {args.ckpt}", flush=True)
    m.backbone.eval()

    BLIND[0] = args.blind
    if args.blind:
        print("BLIND MODE: images replaced with uniform grey", flush=True)
    rows = run(m, items, args.path, args.max_group)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_err = sum(1 for r in rows if "error" in r)
    print(f"wrote {len(rows)} rows ({n_err} errors) -> {args.out}")


if __name__ == "__main__":
    main()

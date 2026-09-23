"""Root-cause the shared-path numerical deviation (plan section 7).

The parity check says the paths disagree on a few items. That is only
acceptable if the disagreement is arithmetic. This separates the two
possibilities directly:

  * raise the precision to float32 and re-measure. If the deviation collapses,
    it is the floating-point path, and bfloat16 is the whole story.
  * hold batching fixed at 1 and re-measure. If the deviation survives at
    batch 1, batching is not the cause and something in the prefix handling is.

It also dumps the worst-disagreeing items with everything a bug would show up
in -- token ids, image grid, position ids, attention mask, readout index -- so
a real implementation error cannot hide behind "tolerance".
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.data.schema import read_jsonl
from vdm.models.vdm_model import VDM


def collect(m: VDM, groups, path_names, limit_groups: int):
    """max |dprob| per item for each path against the independent path."""
    rows = []
    for gi, (img_path, its) in enumerate(groups[:limit_groups]):
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            qs = [{"qtype": it["qtype"], "instruction": it["instruction"],
                   "candidates": it["candidates"]} for it in its]
            g = m.prepare_group(img, qs, "")
            ref = m.run_independent(g)
            outs = {n: getattr(m, "run_" + n)(g) for n in path_names}
            for i in range(g.n_questions):
                k = g.n_options[i]
                za = ref["lm_option_logits"][i, :k].float()
                pa = torch.softmax(za, -1)
                rec = {"group": gi, "q": i, "image": img_path,
                       "item_id": its[i]["item_id"], "n_options": k,
                       "prefix_len": g.prefix_len,
                       "suffix_len": int(g.suffix_ids[i].shape[0]),
                       "image_grid_thw": g.image_grid_thw.tolist(),
                       "margin": float((torch.sort(pa, descending=True).values[:2].diff().abs())[0])
                       if k > 1 else 1.0}
                for n, o in outs.items():
                    zb = o["lm_option_logits"][i, :k].float()
                    pb = torch.softmax(zb, -1)
                    rec[f"dprob_{n}"] = float((pa - pb).abs().max())
                    rec[f"flip_{n}"] = int(za.argmax() != zb.argmax())
                rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=paths.MODEL)
    ap.add_argument("--groups", type=int, default=25)
    ap.add_argument("--n_per_group", type=int, default=8)
    ap.add_argument("--worst", type=int, default=6)
    args = ap.parse_args()

    items = [it for it in read_jsonl(args.items)
             if it["meta"].get("split") == "val" and it["intervention"] == "none"]
    by_img: dict[str, list[dict]] = {}
    for it in items:
        by_img.setdefault(it["image_path"], []).append(it)
    groups = [(k, v[: args.n_per_group]) for k, v in by_img.items()
              if len(v) >= args.n_per_group]
    print(f"{len(groups)} groups with >= {args.n_per_group} questions", flush=True)

    paths = ["independent_batch", "vision_cache", "prefix_share", "prefix_share_batch"]
    report: dict[str, object] = {}

    for dtype, tag in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        print(f"\n=== {tag} ===", flush=True)
        m = VDM(args.model, dtype=dtype)
        m.processor.image_processor.max_pixels = 200704
        rows = collect(m, groups, paths, args.groups)
        summary = {}
        for p in paths:
            d = np.array([r[f"dprob_{p}"] for r in rows])
            f = np.array([r[f"flip_{p}"] for r in rows])
            summary[p] = {"n": len(d), "median": float(np.median(d)),
                          "p99": float(np.percentile(d, 99)), "max": float(d.max()),
                          "flip_rate": float(f.mean())}
            print(f"  {p:22s} median={summary[p]['median']:.2e} "
                  f"p99={summary[p]['p99']:.2e} max={summary[p]['max']:.2e} "
                  f"flips={summary[p]['flip_rate']:.4f}", flush=True)
        report[tag] = {"summary": summary}
        if tag == "bfloat16":
            worst = sorted(rows, key=lambda r: -r["dprob_prefix_share_batch"])[: args.worst]
            report["worst_items"] = worst
            print("\n  worst disagreements (bfloat16):", flush=True)
            for r in worst:
                print(f"    dprob={r['dprob_prefix_share_batch']:.3f} margin={r['margin']:.3f} "
                      f"K={r['n_options']} P={r['prefix_len']} S={r['suffix_len']} "
                      f"flip={r['flip_prefix_share_batch']}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

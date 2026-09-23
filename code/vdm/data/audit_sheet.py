"""Render a human-auditable contact sheet of intervention triples.

Plan 6.5 asks for a human-verified subset.  We cannot staff two annotators, so
the honest substitute is: make every pair inspectable, sample a fixed subset,
and report what the inspection found.  This script produces the sheet.
"""
from __future__ import annotations

import argparse, json, os, random, sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm.data.schema import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = read_jsonl(args.items)
    by_base: dict[str, dict[str, dict]] = {}
    for it in items:
        by_base.setdefault(it["base_id"], {})[it["intervention"]] = it
    triples = [
        b for b, d in by_base.items()
        if "none" in d and any(k.endswith("_relevant") for k in d) and any(k.endswith("_irrelevant") for k in d)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(triples)
    triples = triples[: args.n]
    print(f"{len(by_base)} base ids, {len(triples)} complete triples sampled")

    C, pad, texth = args.cell, 8, 54
    sheet = Image.new("RGB", (3 * C + 4 * pad, len(triples) * (C + texth + pad) + pad), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    rows = []
    for r, b in enumerate(triples):
        dd = by_base[b]
        orig = dd["none"]
        rel = next(v for k, v in dd.items() if k.endswith("_relevant"))
        irr = next(v for k, v in dd.items() if k.endswith("_irrelevant"))
        y0 = pad + r * (C + texth + pad)
        gold = orig["candidates"][orig["label"]]
        d.text((pad, y0), f"[{r}] {orig['instruction'][:88]}", fill=(0, 0, 0))
        d.text((pad, y0 + 14), f"     gold={gold}  family={orig['task_family']}  mode={rel['intervention']}", fill=(60, 60, 60))
        v = rel["meta"].get("verification", {})
        d.text((pad, y0 + 28), f"     area_ratio={v.get('area_ratio',0):.3f}  ev_frac={v.get('evidence_frac_of_image',0):.3f}"
                               f"  overlap_ev={v.get('irrelevant_overlap_evidence_px',0)}", fill=(60, 60, 60))
        for c, (lbl, it) in enumerate([("original", orig), ("relevant", rel), ("irrelevant", irr)]):
            with Image.open(it["image_path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((C, C))
                sheet.paste(im, (pad + c * (C + pad), y0 + texth))
            d.text((pad + c * (C + pad), y0 + texth - 12), lbl, fill=(140, 0, 0))
        rows.append({"row": r, "base_id": b, "question": orig["instruction"], "gold": gold,
                     "mode": rel["intervention"], "verification": v})
    sheet.save(args.out, quality=92)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

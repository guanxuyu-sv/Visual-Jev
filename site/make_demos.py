"""Assemble the page's demos from real evaluation output.

Nothing here is illustrative. The questions, the candidate sets and the
probabilities are the model's actual predictions on held-out images, read back
from the per-example files the paper's tables are computed from; the images are
the ones it was shown, downscaled for the web.
"""
from __future__ import annotations

import argparse, base64, io, json, os, sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def thumb(path: str, width: int, quality: int) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load(pred_path, items_path):
    items = {}
    for line in open(items_path):
        d = json.loads(line)
        items[d["item_id"]] = d
    preds = {}
    for line in open(pred_path):
        r = json.loads(line)
        if "error" not in r and "choice_logits" in r:
            preds[r["item_id"]] = r
    return items, preds


def build_multi(items, preds, n_images, n_q, width, quality):
    """One image, several independent questions, real probabilities."""
    by_img = defaultdict(list)
    for iid, r in preds.items():
        it = items.get(iid)
        if not it or it["intervention"] != "none" or it["qtype"] != "choice":
            continue
        by_img[it["image_path"]].append((it, r))
    ranked = sorted(by_img.items(), key=lambda kv: -len(kv[1]))
    out = []
    for path, group in ranked:
        if len(out) >= n_images:
            break
        if len(group) < n_q or not os.path.exists(path):
            continue
        qs = []
        for it, r in group[:n_q]:
            p = softmax(r["choice_logits"])
            qs.append({"q": it["instruction"], "opts": it["candidates"],
                       "p": [round(float(x), 4) for x in p], "gold": it["label"]})
        out.append({"img": thumb(path, width, quality), "qs": qs,
                    "n_total": len(group)})
    return out


def build_triples(items, preds, n, width, quality, suff_preds=None):
    """Original / evidence removed / equal-area control, with both signals."""
    by_base = defaultdict(dict)
    for iid, r in preds.items():
        it = items.get(iid)
        if not it:
            continue
        iv = it["intervention"]
        arm = ("original" if iv == "none" else
               "relevant" if iv.endswith("_relevant") else
               "control" if iv.endswith("_irrelevant") else None)
        if arm:
            by_base[it["base_id"]][arm] = (it, r)
    out = []
    for base, d in by_base.items():
        if len(out) >= n or set(d) != {"original", "relevant", "control"}:
            continue
        o_it, o_r = d["original"]
        if not all(os.path.exists(d[a][0]["image_path"]) for a in d):
            continue
        # keep cases the model gets right with the evidence and wrong without:
        # those are what the appendix is about
        p_o = softmax(o_r["choice_logits"])
        p_r = softmax(d["relevant"][1]["choice_logits"])
        if not (int(np.argmax(p_o)) == o_it["label"] and int(np.argmax(p_r)) != o_it["label"]):
            continue
        arms = {}
        for arm in ("original", "relevant", "control"):
            it, r = d[arm]
            p = softmax(r["choice_logits"])
            a = r.get("answerable_logit")
            arms[arm] = {"img": thumb(it["image_path"], width, quality),
                         "p": [round(float(x), 4) for x in p],
                         "conf": round(float(p.max()), 4),
                         "pick": int(np.argmax(p)),
                         "suff": round(float(1 / (1 + np.exp(-a))), 4) if a is not None else None}
        out.append({"q": o_it["instruction"], "opts": o_it["candidates"],
                    "gold": o_it["label"], "arms": arms})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--n_q", type=int, default=6)
    ap.add_argument("--n_triples", type=int, default=3)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--quality", type=int, default=72)
    a = ap.parse_args()

    items, preds = load(a.pred, a.items)
    print(f"{len(items)} items, {len(preds)} predictions")
    multi = build_multi(items, preds, a.n_images, a.n_q, a.width, a.quality)
    triples = build_triples(items, preds, a.n_triples, a.width, a.quality)
    demos = {"multi": multi, "triples": triples}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(demos, open(a.out, "w"), separators=(",", ":"))
    kb = os.path.getsize(a.out) / 1024
    print(f"wrote {a.out}: {len(multi)} multi-question images, "
          f"{len(triples)} triples, {kb:.0f} KB")
    for m in multi:
        print(f"  image with {m['n_total']} questions, showing {len(m['qs'])}")
    for t in triples:
        s = t["arms"]
        print(f"  triple: {t['q'][:54]!r} conf {s['original']['conf']:.2f}"
              f" -> {s['relevant']['conf']:.2f} (control {s['control']['conf']:.2f})")


if __name__ == "__main__":
    main()

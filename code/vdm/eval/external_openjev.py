"""Score the external OpenJev visual-NLI checkpoint on our SNLI-VE protocol.

This is a related-work data point, not a head-to-head. The checkpoint is a
different backbone family trained on different data with a *fixed* three-way
entailment head, so the only cell of our benchmark matrix it can occupy is
SNLI-VE. Asking it for GQA or TallyQA would mean inventing an interface it
does not have and then reporting the result as its score.

We give it its own prompt template from its config rather than ours, and its
own label order, so the comparison is on its terms wherever the protocol
allows.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm.data.schema import read_jsonl

# our Claim order is supported / contradicted / not_determined
OURS = ["supported", "contradicted", "not_determined"]
THEIRS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoProcessor

    cfg = AutoConfig.from_pretrained(args.model)
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    template = getattr(cfg, "nli_template", "Premise: {premise}\nHypothesis: {hypothesis}")
    print("labels:", id2label, "\ntemplate:", repr(template), flush=True)
    # map their label positions onto ours so the accuracy is computed the same way
    perm = [None] * 3
    for idx, name in id2label.items():
        perm[THEIRS[name]] = idx
    print("their index for our [supported, contradicted, not_determined]:", perm, flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()

    items = read_jsonl(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items", flush=True)

    rows, n_err = [], 0
    with torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            chunk = items[i : i + args.batch_size]
            try:
                imgs = []
                texts = []
                for it in chunk:
                    with Image.open(it["image_path"]) as im:
                        imgs.append(im.convert("RGB"))
                    texts.append(template.format(
                        premise="", hypothesis=it["instruction"]).strip())
                enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
                enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
                logits = model(**enc).logits.float().cpu().numpy()
            except Exception as e:                       # record, keep going
                n_err += len(chunk)
                for it in chunk:
                    rows.append({"item_id": it["item_id"], "error": repr(e)})
                continue
            for it, lg in zip(chunk, logits):
                rows.append({
                    "item_id": it["item_id"], "base_id": it["base_id"],
                    "parent_image_id": it["parent_image_id"], "source": it["source"],
                    "task_family": it["task_family"], "qtype": it["qtype"],
                    "intervention": it["intervention"], "label": it["label"],
                    "answerable": it["answerable"], "n_options": 3,
                    # reordered into our label convention
                    "choice_logits": [float(lg[p]) for p in perm],
                })
            if i % (args.batch_size * 40) == 0:
                print(f"  {i}/{len(items)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    if ok:
        acc = np.mean([int(np.argmax(r["choice_logits"]) == r["label"]) for r in ok])
        print(f"\nSNLI-VE accuracy: {acc:.4f} on {len(ok)} items ({n_err} errors)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

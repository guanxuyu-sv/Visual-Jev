"""Assemble the GQA portion of the corpus: Choice items, paired interventions,
and image-isolated splits.

Sampling is *image-grouped* on purpose.  Plan 8.2 wants the shared-execution
sweep to run on naturally co-occurring questions rather than a question
repeated N times, so images that carry many usable questions are kept whole.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import os
import random
import sys
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vdm.data import gqa as G
from vdm.data import interventions as IV
from vdm.data.schema import Item, stable_id, write_jsonl


def load_instructions(pattern: str) -> list[dict]:
    import pyarrow.parquet as pq

    recs: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=4096):
            recs.extend(b.to_pylist())
    return recs


def same_name_boxes(
    sg_entry: dict, ref_ids: list[str], gold_answer: str | None = None
) -> list[tuple[int, int, int, int]]:
    """Boxes of every object that could substitute for the evidence.

    Two ways a background region stops being irrelevant: it holds a second
    instance of the referenced object (occluding it changes the answer to an
    existence or counting question), or it holds another instance of the gold
    answer category.  Both are excluded (plan 6.4).

    This filter is only as complete as the scene graph.  GQA does not annotate
    every instance, so a region can still contain an unlabelled duplicate; the
    audit in the paper reports how often that was observed rather than claiming
    the control arm is perfectly clean.
    """
    objs = sg_entry.get("objects") or {}
    names = {str(objs.get(str(i), {}).get("name", "")).lower() for i in ref_ids}
    if gold_answer:
        names.add(str(gold_answer).lower())
    names.discard("")
    out = []
    for oid, o in objs.items():
        if oid in ref_ids:
            continue
        if str(o.get("name", "")).lower() in names:
            x, y, w, h = o.get("x"), o.get("y"), o.get("w"), o.get("h")
            if None not in (x, y, w, h) and w > 0 and h > 0:
                out.append((int(x), int(y), int(w), int(h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gqa_hf", default=paths.under("data", "gqa_hf"))
    ap.add_argument("--sg_dir", default=paths.under("data", "gqa"))
    ap.add_argument("--out", default=paths.under("work", "gqa"))
    ap.add_argument("--n_train_images", type=int, default=6000)
    ap.add_argument("--n_eval_images", type=int, default=1200)
    ap.add_argument("--max_q_per_image", type=int, default=6)
    ap.add_argument("--intervene_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_choices", default="",
                    help="comma-separated option counts to sample per item, "
                         "e.g. 2,3,4,5,6,8; empty keeps the fixed count")
    args = ap.parse_args()
    k_choices = tuple(int(x) for x in args.k_choices.split(",") if x.strip()) or None

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.out, "images")
    iv_dir = os.path.join(args.out, "images_iv")

    t0 = time.time()
    print("loading scene graphs ...", flush=True)
    sg = G.load_scene_graphs(args.sg_dir, ("train", "val"))
    print(f"  {len(sg)} scene graphs in {time.time()-t0:.0f}s", flush=True)

    stats = collections.Counter()
    all_items: list[Item] = []
    pairs_meta: list[dict] = []

    for split, patt, n_images in [
        ("train", f"{args.gqa_hf}/train_balanced_instructions/*.parquet", args.n_train_images),
        ("val", f"{args.gqa_hf}/val_balanced_instructions/*.parquet", args.n_eval_images),
    ]:
        print(f"\n=== {split} ===", flush=True)
        recs = load_instructions(patt)
        print(f"  {len(recs)} raw questions", flush=True)
        stats[f"{split}_raw_questions"] = len(recs)

        # keep only questions whose evidence region is resolvable
        by_image: dict[str, list[dict]] = collections.defaultdict(list)
        for r in recs:
            iid = str(r["imageId"])
            if iid not in sg:
                continue
            if not G.referenced_object_ids(r):
                continue
            by_image[iid].append(r)
        stats[f"{split}_images_with_sg"] = len(by_image)

        # prefer images that carry many questions (for the N sweep)
        ordered = sorted(by_image.items(), key=lambda kv: -len(kv[1]))
        head = ordered[: n_images * 2]
        rng.shuffle(head)
        chosen = head[:n_images]
        print(f"  chose {len(chosen)} images, "
              f"median q/img={sorted(len(v) for _, v in chosen)[len(chosen)//2]}", flush=True)

        keep_ids = {iid for iid, _ in chosen}
        img_split_dir = os.path.join(img_dir, split)
        print("  exporting images ...", flush=True)
        image_path_of = G.export_images_from_parquet(
            f"{args.gqa_hf}/{split}_balanced_images/*.parquet", img_split_dir, keep_ids
        )
        print(f"  {len(image_path_of)} images on disk ({time.time()-t0:.0f}s)", flush=True)

        sel_recs = []
        for iid, rs in chosen:
            if iid not in image_path_of:
                continue
            rng.shuffle(rs)
            sel_recs.extend(rs[: args.max_q_per_image])

        items = G.build_items(sel_recs, sg, image_path_of, split, seed=args.seed,
                              k_choices=k_choices)
        stats[f"{split}_base_items"] = len(items)
        for it in items:
            stats[f"{split}_family_{it.task_family}"] += 1
            if it.candidates:
                stats[f"{split}_K{len(it.candidates)}"] += 1
        print(f"  {len(items)} base Choice items", flush=True)

        # ---------------- paired interventions ---------------- #
        n_iv = int(len(items) * args.intervene_frac)
        iv_targets = rng.sample(items, min(n_iv, len(items)))
        iv_split_dir = os.path.join(iv_dir, split)
        made = 0
        for k, it in enumerate(iv_targets):
            if k % 2000 == 0:
                print(f"    interventions {k}/{len(iv_targets)} made={made} ({time.time()-t0:.0f}s)", flush=True)
            sge = sg.get(it.parent_image_id) or {}
            forb = same_name_boxes(sge, it.meta["evidence_object_ids"], it.meta.get("gold_answer"))
            # occlude is the *trained* degradation; blur and downscale are held
            # out as the Intervention-OOD arm (plan 6.2).
            mode = "occlude" if it.meta["split"] == "train" else rng.choice(["occlude", "blur", "downscale"])
            sigma = {"occlude": 0.0, "blur": 14.0, "downscale": 8.0}[mode]
            res = IV.make_pair(
                it.parent_image_path, iv_split_dir, it.base_id,
                it.meta["evidence_boxes"], forb, mode=mode, sigma=sigma, rng=rng,
            )
            if res is None:
                stats[f"{split}_iv_rejected"] += 1
                continue
            made += 1
            stats[f"{split}_iv_made_{mode}"] += 1
            pairs_meta.append({"base_id": it.base_id, "split": split, **res["verification"],
                               "mode": mode, "task_family": it.task_family})

            # relevant arm: evidence removed -> not answerable, decision masked
            all_items.append(dataclasses.replace(
                it,
                item_id=it.base_id + f":rel_{mode}",
                image_path=res["relevant_path"],
                answerable=0,
                decision_supervised=0,
                intervention=f"{mode}_relevant",
                intervention_chain=[f"{mode}_relevant"],
                meta={**it.meta, "verification": res["verification"],
                      "applied_boxes": res["relevant_boxes"]},
            ))
            # irrelevant arm: same degradation elsewhere -> answer and
            # sufficiency must both survive
            all_items.append(dataclasses.replace(
                it,
                item_id=it.base_id + f":irr_{mode}",
                image_path=res["irrelevant_path"],
                answerable=1,
                decision_supervised=1,
                intervention=f"{mode}_irrelevant",
                intervention_chain=[f"{mode}_irrelevant"],
                meta={**it.meta, "verification": res["verification"],
                      "applied_boxes": res["irrelevant_boxes"]},
            ))

            # semantics-preserving variants for the consistency term
            if rng.random() < 0.5:
                new_c, new_l, order = IV.permute_candidates(it.candidates, it.label, rng)
                all_items.append(dataclasses.replace(
                    it,
                    item_id=it.base_id + ":perm",
                    candidates=new_c, label=new_l,
                    intervention="candidate_permutation",
                    intervention_chain=["candidate_permutation"],
                    meta={**it.meta, "permutation": order},
                ))
            if rng.random() < 0.5:
                all_items.append(dataclasses.replace(
                    it,
                    item_id=it.base_id + ":para",
                    instruction=IV.paraphrase(it.instruction, rng),
                    intervention="paraphrase",
                    intervention_chain=["paraphrase"],
                    meta={**it.meta, "original_instruction": it.instruction},
                ))
        print(f"  interventions: {made} pairs made, {stats[f'{split}_iv_rejected']} rejected", flush=True)
        all_items.extend(items)

    write_jsonl(os.path.join(args.out, "gqa_items.jsonl"), all_items)
    with open(os.path.join(args.out, "gqa_pairs_verification.jsonl"), "w") as fh:
        for p in pairs_meta:
            fh.write(json.dumps(p) + "\n")
    with open(os.path.join(args.out, "gqa_build_stats.json"), "w") as fh:
        json.dump(dict(stats), fh, indent=1)
    print(f"\nwrote {len(all_items)} items, {len(pairs_meta)} verified pairs "
          f"in {time.time()-t0:.0f}s", flush=True)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()

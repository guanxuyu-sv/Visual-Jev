"""GQA import: native Choice items plus scene-graph grounded evidence regions.

Why GQA is the backbone of this study: every balanced question carries
`annotations` mapping question / answer tokens to scene-graph object ids, and
the scene graph carries a bounding box per object.  That gives a *programmatic*
notion of "the region the question depends on", which is what makes the paired
relevant-vs-irrelevant evidence interventions of plan 6.4 constructible without
per-item human region labelling.

Structural types are kept apart on purpose (plan 6.3, "prefer native
candidates"):
  verify / logical  -> native yes/no Choice
  choose            -> native two-way Choice, options appear in the question
  query             -> open answer converted to Choice; reported separately
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import random
from typing import Any, Iterable

from .schema import K_MAX, Item, stable_id

NATIVE_STRUCTURAL = {"verify", "logical", "choose", "compare"}
YESNO = ("yes", "no")


def load_scene_graphs(sg_dir: str, splits: Iterable[str] = ("train", "val")) -> dict[str, dict]:
    sg: dict[str, dict] = {}
    for sp in splits:
        p = os.path.join(sg_dir, f"{sp}_sceneGraphs.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            sg.update(json.load(fh))
    return sg


def referenced_object_ids(rec: dict) -> list[str]:
    """Object ids the question and its answer actually depend on.

    `annotations` entries look like {"objectId": "3", "value": "329774"} where
    `value` is the scene-graph object id and `objectId` is the token position.
    We take question + answer + fullAnswer references; the union is the evidence
    region.  Objects mentioned only in `fullAnswer` still carry the answer, so
    they count as evidence.
    """
    ids: list[str] = []
    ann = rec.get("annotations") or {}
    for key in ("question", "answer", "fullAnswer"):
        entries = ann.get(key) or []
        if isinstance(entries, dict):        # some dumps store {pos: objid}
            entries = [{"value": v} for v in entries.values()]
        for e in entries:
            v = e.get("value") if isinstance(e, dict) else e
            if v:
                ids.append(str(v))
    # `semantic` arguments look like "bird (329774)" and sometimes reference
    # objects that annotations missed.
    for step in rec.get("semantic") or []:
        arg = str(step.get("argument", ""))
        if "(" in arg and ")" in arg:
            inner = arg[arg.rindex("(") + 1 : arg.rindex(")")]
            for tok in inner.replace(",", " ").split():
                if tok.isdigit():
                    ids.append(tok)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def object_boxes(sg_entry: dict, obj_ids: Iterable[str]) -> list[tuple[int, int, int, int]]:
    objs = sg_entry.get("objects") or {}
    boxes = []
    for oid in obj_ids:
        o = objs.get(str(oid))
        if not o:
            continue
        x, y, w, h = o.get("x"), o.get("y"), o.get("w"), o.get("h")
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def parse_choose_options(question: str) -> list[str] | None:
    """`choose` questions state their options inline: "Is it red or blue?"."""
    q = question.rstrip("?").strip()
    if " or " not in q:
        return None
    head, tail = q.rsplit(" or ", 1)
    # last comma/space separated noun phrase before " or "
    left = head.split(",")[-1].strip().split(" ")
    # take the shortest trailing span that looks like an option (1-3 tokens)
    for n in (1, 2, 3):
        if len(left) >= n:
            cand = " ".join(left[-n:]).strip(" ,")
            if cand and tail.strip():
                opts = [cand.lower(), tail.strip().lower()]
                if opts[0] != opts[1] and all(len(o) < 30 for o in opts):
                    return opts
    return None


def build_answer_pools(records: list[dict]) -> dict[str, collections.Counter]:
    """Answer frequency per `types.detailed`, used for distractor sampling."""
    pools: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        t = (r.get("types") or {}).get("detailed") or "unknown"
        pools[t][str(r["answer"]).lower()] += 1
    return pools


# Answers that are mutually compatible must never appear as distractors for one
# another (plan 6.3: filter non-exclusive candidates).
_SYNONYM_GROUPS = [
    {"man", "person", "guy", "male"},
    {"woman", "person", "lady", "female"},
    {"kid", "child", "boy", "girl"},
    {"large", "big"},
    {"small", "little", "tiny"},
    {"gray", "grey"},
    {"couch", "sofa"},
    {"bike", "bicycle"},
    {"cellphone", "phone", "cell phone"},
    {"tv", "television"},
]


def _compatible(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    for g in _SYNONYM_GROUPS:
        if a in g and b in g:
            return True
    return a in b or b in a


def sample_distractors(
    gold: str, pool: collections.Counter, k: int, rng: random.Random
) -> list[str] | None:
    gold = gold.lower()
    cands = [a for a, _ in pool.most_common() if not _compatible(a, gold)]
    if len(cands) < k:
        return None
    # Frequency-weighted but not purely top-k, so distractors are plausible and
    # position/length shortcuts are less systematic.
    head = cands[: max(k * 4, 12)]
    return rng.sample(head, k)


def build_items(
    records: list[dict],
    scene_graphs: dict[str, dict],
    image_path_of: dict[str, str],
    split: str,
    n_choice: int = 4,
    seed: int = 0,
    require_boxes: bool = True,
    k_choices: tuple[int, ...] | None = None,
) -> list[Item]:
    """Convert GQA balanced records into Choice `Item`s with evidence boxes.

    `k_choices` varies the option count per item. A fixed count is a trap for a
    slot-indexed decision head: train only on K<=4 and slots 4 and beyond never
    receive a gradient, so the head cannot transfer to a task that uses them
    even when the backbone underneath it can.
    """
    rng = random.Random(seed)
    pools = build_answer_pools(records)
    items: list[Item] = []
    for r in records:
        img_id = str(r["imageId"])
        if img_id not in image_path_of:
            continue
        sg = scene_graphs.get(img_id)
        obj_ids = referenced_object_ids(r)
        boxes = object_boxes(sg, obj_ids) if sg else []
        if require_boxes and not boxes:
            continue

        structural = (r.get("types") or {}).get("structural") or "unknown"
        detailed = (r.get("types") or {}).get("detailed") or "unknown"
        gold = str(r["answer"]).lower()

        if structural in ("verify", "logical") and gold in YESNO:
            cands, family = list(YESNO), "gqa_verify"
        elif structural == "choose":
            opts = parse_choose_options(str(r["question"]))
            if not opts or gold not in opts:
                continue
            cands, family = opts, "gqa_choose"
        elif structural == "query":
            k = rng.choice(k_choices) if k_choices else n_choice
            d = sample_distractors(gold, pools[detailed], k - 1, rng)
            if d is None and k > 2:            # fall back to a smaller set
                d = sample_distractors(gold, pools[detailed], n_choice - 1, rng)
            if d is None:
                continue
            cands, family = [gold] + d, "gqa_query_choice"
        else:
            continue

        if len(cands) > K_MAX:
            continue
        order = list(range(len(cands)))
        rng.shuffle(order)
        cands = [cands[i] for i in order]
        label = cands.index(gold)

        qid = str(r["id"])
        base_id = stable_id("gqa", qid)
        items.append(
            Item(
                item_id=base_id + ":orig",
                base_id=base_id,
                source="gqa",
                task_family=family,
                qtype="choice",
                image_path=image_path_of[img_id],
                parent_image_path=image_path_of[img_id],
                parent_image_id=img_id,
                question_id=qid,
                shared_context="",
                instruction=str(r["question"]),
                candidates=cands,
                label=label,
                answerable=1,
                decision_supervised=1,
                intervention="none",
                intervention_chain=[],
                meta={
                    "split": split,
                    "structural": structural,
                    "detailed": detailed,
                    "evidence_boxes": boxes,
                    "evidence_object_ids": obj_ids,
                    "image_wh": [sg.get("width"), sg.get("height")] if sg else None,
                    "gold_answer": gold,
                },
            )
        )
    return items


def export_images_from_parquet(
    parquet_glob: str, out_dir: str, keep_image_ids: set[str] | None = None
) -> dict[str, str]:
    """Materialise GQA images from the lmms-lab parquet shards to JPEG files.

    Returns imageId -> path.  Writing real files (rather than decoding parquet
    per batch) keeps the intervention pipeline, the dataloader and the latency
    benchmark reading identical bytes.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    mapping: dict[str, str] = {}
    for shard in sorted(glob.glob(parquet_glob)):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=64):
            d = batch.to_pydict()
            id_col = "id" if "id" in d else "imageId"
            for img_id, img in zip(d[id_col], d["image"]):
                img_id = str(img_id)
                if keep_image_ids is not None and img_id not in keep_image_ids:
                    continue
                dst = os.path.join(out_dir, f"{img_id}.jpg")
                if not os.path.exists(dst):
                    raw = img["bytes"] if isinstance(img, dict) else img
                    Image.open(io.BytesIO(raw)).convert("RGB").save(dst, quality=95)
                mapping[img_id] = dst
    return mapping

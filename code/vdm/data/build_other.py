"""SNLI-VE (Claim) and TextVQA (Task-OOD Choice).

SNLI-VE supplies the Claim output type.  One labelling decision matters and is
made explicitly here: a `neutral` pair becomes the decision label
`not_determined`, and its `answerable` flag stays 1.

That is deliberate.  Plan 4.2 separates three kinds of uncertainty, and SNLI-VE
neutral is the *second* kind -- the image genuinely underdetermines the
statement -- not the first, where the observation itself is too degraded to
read.  Treating neutral as "not answerable" would collapse the distinction the
paper is built on, and would teach the sufficiency head to fire on semantic
underdetermination instead of on missing evidence.

TextVQA is never trained on.  It is the Task-OOD arm: a different dataset, a
different image distribution, and a skill (reading text in the scene) that the
training mix does not contain.
"""
from __future__ import annotations

import argparse, collections, glob, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.data.gqa import _compatible
from vdm.data.schema import Item, stable_id, write_jsonl

LABEL_MAP = {"entailment": 0, "contradiction": 1, "neutral": 2}   # -> CLAIM_LABELS


def build_snli_ve(jsonl_path: str, image_dir: str, split: str, limit: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("gold_label") not in LABEL_MAP:
                continue
            rows.append(r)
    rng.shuffle(rows)

    items: list[Item] = []
    seen_missing = 0
    for r in rows:
        if limit and len(items) >= limit:
            break
        fid = str(r["Flickr30K_ID"])
        path = os.path.join(image_dir, f"{fid}.jpg")
        if not os.path.exists(path):
            seen_missing += 1
            continue
        qid = str(r["pairID"])
        base = stable_id("snlive", qid)
        items.append(Item(
            item_id=base + ":orig", base_id=base, source="snli_ve",
            task_family="snli_ve_claim", qtype="claim",
            image_path=path, parent_image_path=path, parent_image_id=fid,
            question_id=qid, shared_context="",
            instruction=str(r["sentence2"]).strip(),
            candidates=None, label=LABEL_MAP[r["gold_label"]],
            # The image is readable; what is uncertain is the statement, and
            # that uncertainty is already carried by the not_determined class.
            answerable=1, decision_supervised=1,
            intervention="none", intervention_chain=[],
            meta={"split": split, "gold_label": r["gold_label"],
                  "premise_caption": r.get("sentence1", "")},
        ))
    if seen_missing:
        print(f"  {seen_missing} rows skipped: image not on disk")
    return items


def build_textvqa(json_path: str, image_root: str, split: str, limit: int,
                  n_choice: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    data = json.load(open(json_path))["data"]

    # Distractors must be answers to questions of the *same shape*.  Drawing
    # them from the global answer pool produces options a language prior can
    # separate without reading the image -- our first attempt did exactly that
    # and every system scored above 99% (plan 6.3, candidate-set leakage).
    # Grouping by the question's leading tokens keeps brands against brands and
    # times against times.
    def template(q: str, n: int = 4) -> str:
        toks = [t for t in q.lower().strip("?").split() if t]
        return " ".join(toks[:n])

    by_tmpl_n: dict[int, dict[str, collections.Counter]] = {
        n: collections.defaultdict(collections.Counter) for n in (4, 3, 2)
    }
    pool = collections.Counter()
    skipped_thin = 0
    BAD = ("", "unanswerable", "answering does not require reading text in the image")
    for r in data:
        a = collections.Counter(x.strip().lower() for x in r["answers"] if x.strip())
        if not a:
            continue
        top = a.most_common(1)[0][0]
        if top in BAD:
            continue
        pool[top] += 1
        for n_tok in (4, 3, 2):
            by_tmpl_n[n_tok][template(r["question"], n_tok)][top] += 1
    common = [a for a, _ in pool.most_common() if a not in BAD]

    items: list[Item] = []
    rng.shuffle(data)
    for r in data:
        if limit and len(items) >= limit:
            break
        ans = collections.Counter(x.strip().lower() for x in r["answers"] if x.strip())
        if not ans:
            continue
        gold, n_agree = ans.most_common(1)[0]
        # Only keep questions with clear annotator agreement: a 3/10 majority
        # answer is not a reliable single-choice gold.
        if n_agree < 5 or gold in ("unanswerable",):
            continue
        # Distractors must come from answers to questions of the same shape.
        # We widen the template (4 -> 3 -> 2 leading tokens) until the pool is
        # large enough and drop the question if it never is.  Falling back to a
        # global pool is what produced the leak, so there is no fallback.
        uniq, tmpl = [], None
        for n_tok in (4, 3, 2):
            t = template(r["question"], n_tok)
            pool_t = by_tmpl_n[n_tok].get(t)
            if not pool_t:
                continue
            c = [x for x, _ in pool_t.most_common() if not _compatible(x, gold)]
            if len(c) >= n_choice - 1:
                uniq, tmpl = c, t
                break
        if len(uniq) < n_choice - 1:
            skipped_thin += 1
            continue
        distract = rng.sample(uniq[: max(40, n_choice * 8)], n_choice - 1)
        opts = [gold] + distract
        rng.shuffle(opts)
        img = os.path.join(image_root, f"{r['image_id']}.jpg")
        if not os.path.exists(img):
            continue
        qid = str(r.get("question_id", r["image_id"]))
        base = stable_id("textvqa", qid, r["question"])
        items.append(Item(
            item_id=base + ":orig", base_id=base, source="textvqa",
            task_family="textvqa_choice", qtype="choice",
            image_path=img, parent_image_path=img, parent_image_id=str(r["image_id"]),
            question_id=qid, shared_context="",
            instruction=str(r["question"]).strip(),
            candidates=opts, label=opts.index(gold),
            answerable=1, decision_supervised=1,
            intervention="none", intervention_chain=[],
            meta={"split": split, "gold_answer": gold, "annotator_agreement": n_agree,
                  "template": tmpl, "n_same_template_candidates": len(uniq),
                  "image_wh": [r.get("image_width"), r.get("image_height")]},
        ))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=paths.under("work", "other"))
    ap.add_argument("--snli_dir", default=paths.under("data", "snli_ve"))
    ap.add_argument("--flickr_dir", default=paths.under("data", "flickr30k", "flickr30k-images"))
    ap.add_argument("--textvqa_dir", default=paths.under("data", "textvqa"))
    ap.add_argument("--n_snli_train", type=int, default=9000)
    ap.add_argument("--n_snli_eval", type=int, default=2000)
    ap.add_argument("--n_textvqa", type=int, default=2500)
    ap.add_argument("--textvqa_k", type=int, default=4,
                    help="option count for the Task-OOD set; 4 leaves a strong "
                         "backbone at ceiling, so the reported arm uses more")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stats = {}

    print("SNLI-VE train ...", flush=True)
    tr = build_snli_ve(f"{args.snli_dir}/snli_ve_train.jsonl", args.flickr_dir, "train",
                       args.n_snli_train, args.seed)
    print("SNLI-VE test ...", flush=True)
    te = build_snli_ve(f"{args.snli_dir}/snli_ve_test.jsonl", args.flickr_dir, "val",
                       args.n_snli_eval, args.seed)
    # Flickr30k images are shared between SNLI-VE splits only by construction of
    # the original release; assert no image crosses our train/test line.
    tr_imgs = {i.parent_image_id for i in tr}
    te = [i for i in te if i.parent_image_id not in tr_imgs]
    stats["snli_train"] = len(tr)
    stats["snli_test_after_image_isolation"] = len(te)
    for name, items in (("snli_train", tr), ("snli_test", te)):
        c = collections.Counter(i.label for i in items)
        stats[f"{name}_label_dist"] = {["supported", "contradicted", "not_determined"][k]: v
                                       for k, v in sorted(c.items())}

    print("TextVQA (task-OOD) ...", flush=True)
    tv = build_textvqa(f"{args.textvqa_dir}/TextVQA_0.5.1_val.json",
                       f"{args.textvqa_dir}/train_images", "val", args.n_textvqa,
                       args.textvqa_k, args.seed)
    stats["textvqa_task_ood"] = len(tv)

    write_jsonl(os.path.join(args.out, "snli_ve_train.jsonl"), tr)
    write_jsonl(os.path.join(args.out, "snli_ve_test.jsonl"), te)
    write_jsonl(os.path.join(args.out, "textvqa_taskood.jsonl"), tv)
    json.dump(stats, open(os.path.join(args.out, "other_build_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

"""TallyQA as a fully held-out Choice benchmark.

Counting is the one ability nothing in post-training touches, which is what
makes it the Task-OOD arm the study needs. The construction decides whether
the arm means anything: distractors drawn at random from all integers make the
task trivial for a language prior, so the option set is built from counts
*near* the gold, where telling them apart requires actually counting.

Nothing here ever enters training. The split is the published test set.
"""
from __future__ import annotations

import argparse, collections, glob, io, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.data.schema import Item, stable_id, write_jsonl


def spell(n: int) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten"]
    return words[n] if 0 <= n < len(words) else str(n)


def build_options(gold: int, k: int, rng: random.Random, gold_set: list[int]) -> list[str] | None:
    """Options are the counts *closest to the gold among those that actually
    occur as answers*.

    Restricting to the observed answer set matters: this release samples counts
    {0, 5..15} and never 1..4, so drawing a near-miss of 3 or 4 would mark the
    wrong option as one that is never correct anywhere in the benchmark --- a
    cue that has nothing to do with counting.
    """
    others = [c for c in gold_set if c != gold]
    if len(others) < k - 1:
        return None
    others.sort(key=lambda c: (abs(c - gold), c))       # nearest first
    window = others[: max(k - 1, min(len(others), k + 2))]
    cands = rng.sample(window, k - 1)
    opts = [str(gold)] + [str(c) for c in cands]
    rng.shuffle(opts)
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=paths.under("data", "tallyqa", "tallyQA_short.parquet"))
    ap.add_argument("--out", default=paths.under("work", "tallyqa"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(args.parquet)):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=256):
            rows.extend(b.to_pylist())
    counts = collections.Counter(int(r["groundtruth"]) for r in rows)
    gold_set = sorted(counts)
    print(f"{len(rows)} rows; counts that occur as answers: {gold_set}")
    print("This release is a count-balanced subset, not TallyQA's natural "
          "distribution; it is reported as such and never compared to "
          "published TallyQA numbers.")

    items: list[Item] = []
    stats: collections.Counter = collections.Counter()
    for i, r in enumerate(rows):
        gold = int(r["groundtruth"])
        opts = build_options(gold, args.k, rng, gold_set)
        if opts is None:
            stats["skipped_no_options"] += 1
            continue
        path = os.path.join(img_dir, f"tally_{i:05d}.jpg")
        if not os.path.exists(path):
            raw = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
            Image.open(io.BytesIO(raw)).convert("RGB").save(path, quality=95)
        base = stable_id("tallyqa", i, r["question"])
        items.append(Item(
            item_id=base + ":orig", base_id=base, source="tallyqa",
            task_family="tallyqa_choice", qtype="choice",
            image_path=path, parent_image_path=path, parent_image_id=f"tally_{i:05d}",
            question_id=str(i), shared_context="",
            instruction=str(r["question"]).strip(),
            candidates=opts, label=opts.index(str(gold)),
            answerable=1, decision_supervised=1,
            intervention="none", intervention_chain=[],
            meta={"split": "val", "gold_count": gold,
                  "is_simple": bool(r.get("is_simple", True)),
                  "count_bucket": "0" if gold == 0 else ("5-9" if gold <= 9 else "10+"),
                  "subset": "count_balanced_short"},
        ))
        stats[f"K{len(opts)}"] += 1
        stats[items[-1].meta["count_bucket"]] += 1
        stats["simple" if items[-1].meta["is_simple"] else "complex"] += 1

    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "tallyqa.jsonl"), items)
    json.dump(dict(stats), open(os.path.join(args.out, "build_stats.json"), "w"), indent=1)
    print(f"wrote {len(items)} items -> {args.out}")
    print(json.dumps(dict(stats), indent=1))
    print("\nexamples:")
    for it in items[:3]:
        print(f"  {it.instruction[:56]:<56} {it.candidates}  gold={it.candidates[it.label]}")


if __name__ == "__main__":
    main()

"""Shared-execution correctness and cost (plan 9.1 and 9.2).

Two things are measured and kept apart.

*Parity* asks whether the shared paths reproduce the independent path's
decision.  We report the max and the quantiles of the logit and probability
differences and the argmax flip rate, and we look separately at the items whose
top two options are close, because those are the ones a small numerical shift
can actually flip.

*Cost* asks what sharing buys.  Reported totals use synchronized wall-clock
intervals after a warm-up; CUDA events provide the internal phase breakdown,
so a speedup can be attributed rather than just asserted:

    preprocess | vision encode | prefix prefill | cache fork | suffix | readout

Two generation baselines are included so the comparison is not against a
strawman: emitting one answer token per question, and emitting all N answers in
one compact sequence.
"""
from __future__ import annotations

import argparse, collections, json, os, random, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.data.schema import read_jsonl
from vdm.models.prompts import prefix_text, suffix_text
from vdm.models.vdm_model import VDM


class Timer:
    """CUDA-event timing; wall time alone hides async launches."""

    def __init__(self):
        self.spans: dict[str, float] = collections.defaultdict(float)

    def __call__(self, name: str):
        return _Span(self, name)


class _Span:
    def __init__(self, t: Timer, name: str):
        self.t, self.name = t, name

    def __enter__(self):
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)
        self.a.record()
        return self

    def __exit__(self, *exc):
        self.b.record()
        torch.cuda.synchronize()
        self.t.spans[self.name] += self.a.elapsed_time(self.b) / 1000.0
        return False


# --------------------------------------------------------------------------- #
# instrumented paths
# --------------------------------------------------------------------------- #
@torch.no_grad()
def timed_independent(m: VDM, img, qs, ctx) -> Timer:
    t = Timer()
    for q in qs:
        with t("preprocess"):
            g = m.prepare_group(img, [q], ctx)
        with t("forward_full"):
            m.run_independent(g)
    return t


@torch.no_grad()
def timed_vision_cache(m: VDM, img, qs, ctx) -> Timer:
    t = Timer()
    with t("preprocess"):
        g = m.prepare_group(img, qs, ctx)
    with t("vision_encode"):
        image_embeds, deepstack = m.encode_vision(g.pixel_values, g.image_grid_thw)
    for i in range(g.n_questions):
        with t("language_forward"):
            full = torch.cat([g.prefix_ids, g.suffix_ids[i]]).to(m.device_str)
            emb, vmask = m._embed(full, image_embeds)
            m.vl_model.language_model(
                inputs_embeds=emb[None],
                position_ids=g.position_ids[i][:, None].to(m.device_str),
                visual_pos_masks=vmask[None], deepstack_visual_embeds=deepstack,
                use_cache=False,
            )
    return t


@torch.no_grad()
def timed_independent_batch(m: VDM, img, qs, ctx) -> Timer:
    """Nothing shared, everything batched: the control that separates the two."""
    t = Timer()
    with t("preprocess"):
        g = m.prepare_group(img, qs, ctx)
    with t("batched_full_forward"):
        m.run_independent_batch(g)
    return t


@torch.no_grad()
def timed_vision_cache_batch(m: VDM, img, qs, ctx) -> Timer:
    t = Timer()
    with t("preprocess"):
        g = m.prepare_group(img, qs, ctx)
    with t("vision_encode"):
        m.encode_vision(g.pixel_values, g.image_grid_thw)
    with t("batched_language_forward"):
        m.run_vision_cache_batch(g)
    return t


@torch.no_grad()
def timed_prefix_share(m: VDM, img, qs, ctx, batched: bool) -> Timer:
    t = Timer()
    with t("preprocess"):
        g = m.prepare_group(img, qs, ctx)
    with t("vision_encode"):
        image_embeds, deepstack = m.encode_vision(g.pixel_values, g.image_grid_thw)
    with t("prefix_prefill"):
        from transformers.cache_utils import DynamicCache
        ids = g.prefix_ids.to(m.device_str)
        emb, vmask = m._embed(ids, image_embeds)
        base = DynamicCache()
        m.vl_model.language_model(
            inputs_embeds=emb[None],
            position_ids=g.position_ids[0][:, : g.prefix_len][:, None].to(m.device_str),
            past_key_values=base, visual_pos_masks=vmask[None],
            deepstack_visual_embeds=deepstack, use_cache=True,
        )
    if batched:
        n, P = g.n_questions, g.prefix_len
        lens = [int(s.shape[0]) for s in g.suffix_ids]
        S = max(lens)
        pad_id = m.tokenizer.pad_token_id or m.tokenizer.eos_token_id
        sids = torch.full((n, S), pad_id, dtype=torch.long)
        pos = torch.zeros(3, n, S, dtype=torch.long)
        attn = torch.zeros(n, P + S, dtype=torch.long)
        attn[:, :P] = 1
        for i, L in enumerate(lens):
            sids[i, S - L :] = g.suffix_ids[i]
            pos[:, i, S - L :] = g.position_ids[i][:, P:]
            attn[i, P + S - L :] = 1
        with t("cache_fork"):
            cache = m._fork(base, n)
        with t("suffix"):
            emb, _ = m._embed(sids.to(m.device_str), None)
            m.vl_model.language_model(
                inputs_embeds=emb, position_ids=pos.to(m.device_str),
                attention_mask=attn.to(m.device_str), past_key_values=cache, use_cache=True,
            )
    else:
        for i in range(g.n_questions):
            with t("cache_fork"):
                cache = m._fork(base, 1)
            with t("suffix"):
                emb, _ = m._embed(g.suffix_ids[i].to(m.device_str), None)
                m.vl_model.language_model(
                    inputs_embeds=emb[None],
                    position_ids=g.position_ids[i][:, g.prefix_len :][:, None].to(m.device_str),
                    past_key_values=cache, use_cache=True,
                )
    return t


@torch.no_grad()
def timed_independent_batch_gen(m: VDM, img, qs, ctx) -> Timer:
    """L5: no reuse, batched, and the answer is *emitted* rather than read.

    Same forward as `independent_batch` plus a full-vocabulary argmax, so the
    pair isolates the cost of the readout itself with no generation-harness
    overhead on either side.
    """
    t = Timer()
    with t("preprocess"):
        g = m.prepare_group(img, qs, ctx)
    with t("batched_full_forward"):
        ids, pos, attn = m._pad_full(g)
        n = g.n_questions
        px = g.pixel_values.to(m.device_str, m.dtype)
        px = px.repeat(n, 1) if px.dim() == 2 else px.repeat(n, *([1] * (px.dim() - 1)))
        out = m.vl_model(
            input_ids=ids.to(m.device_str), pixel_values=px,
            image_grid_thw=g.image_grid_thw.to(m.device_str).repeat(n, 1),
            mm_token_type_ids=(ids == m.image_token_id).to(torch.int32).to(m.device_str),
            position_ids=pos.to(m.device_str), attention_mask=attn.to(m.device_str),
            use_cache=False,
        )
    with t("emit_token"):
        m.lm_head(out.last_hidden_state[:, -1].to(m.dtype)).argmax(-1)
    return t


@torch.no_grad()
def timed_prefix_share_batch_gen(m: VDM, img, qs, ctx) -> Timer:
    """L6: the full shared path, but emitting a token instead of reading a head.

    If this lands on top of `prefix_share_batch`, the serving advantage is the
    shared prefix and the batch, not the absence of a decode step.
    """
    from transformers.cache_utils import DynamicCache
    t = Timer()
    with t("preprocess"):
        g = m.prepare_group(img, qs, ctx)
    with t("vision_encode"):
        image_embeds, deepstack = m.encode_vision(g.pixel_values, g.image_grid_thw)
    with t("prefix_prefill"):
        ids = g.prefix_ids.to(m.device_str)
        emb, vmask = m._embed(ids, image_embeds)
        base = DynamicCache()
        m.vl_model.language_model(
            inputs_embeds=emb[None],
            position_ids=g.position_ids[0][:, : g.prefix_len][:, None].to(m.device_str),
            past_key_values=base, visual_pos_masks=vmask[None],
            deepstack_visual_embeds=deepstack, use_cache=True,
        )
    n, P = g.n_questions, g.prefix_len
    lens = [int(x.shape[0]) for x in g.suffix_ids]
    S = max(lens)
    pad_id = m.tokenizer.pad_token_id or m.tokenizer.eos_token_id
    sids = torch.full((n, S), pad_id, dtype=torch.long)
    pos = torch.zeros(3, n, S, dtype=torch.long)
    attn = torch.zeros(n, P + S, dtype=torch.long)
    attn[:, :P] = 1
    for i, L in enumerate(lens):
        sids[i, S - L:] = g.suffix_ids[i]
        pos[:, i, S - L:] = g.position_ids[i][:, P:]
        attn[i, P + S - L:] = 1
    with t("cache_fork"):
        cache = m._fork(base, n)
    with t("suffix"):
        emb, _ = m._embed(sids.to(m.device_str), None)
        out = m.vl_model.language_model(
            inputs_embeds=emb, position_ids=pos.to(m.device_str),
            attention_mask=attn.to(m.device_str), past_key_values=cache, use_cache=True,
        )
    with t("emit_token"):
        m.lm_head(out.last_hidden_state[:, -1].to(m.dtype)).argmax(-1)
    return t


@torch.no_grad()
def timed_generate(m: VDM, img, qs, ctx, mode: str) -> Timer:
    """Generation baselines.

    one_token_each : the fairest comparison to a direct readout -- one forward
                     and a single emitted token per question.
    compact_joint  : all N answers in one short sequence, i.e. the strongest
                     realistic generative alternative (not a verbose JSON).
    """
    t = Timer()
    if mode == "one_token_each":
        for q in qs:
            with t("preprocess"):
                g = m.prepare_group(img, [q], ctx)
            with t("generate"):
                full = torch.cat([g.prefix_ids, g.suffix_ids[0]])[None].to(m.device_str)
                m.vl.generate(
                    input_ids=full,
                    pixel_values=g.pixel_values.to(m.device_str, m.dtype),
                    image_grid_thw=g.image_grid_thw.to(m.device_str),
                    max_new_tokens=1, do_sample=False,
                )
    else:
        with t("preprocess"):
            lines = [f"{i+1}. {q['instruction']} Options: " +
                     ", ".join(f"{chr(65+j)}. {c}" for j, c in enumerate(q.get("candidates") or ["supported", "contradicted", "not determined"]))
                     for i, q in enumerate(qs)]
            txt = (prefix_text(ctx) + "\n\n" + "\n".join(lines) +
                   "\nAnswer every question with just its number and letter, one per line."
                   "<|im_end|>\n<|im_start|>assistant\n")
            enc = m.processor(text=[txt], images=[img], return_tensors="pt")
        with t("generate"):
            m.vl.generate(
                input_ids=enc["input_ids"].to(m.device_str),
                pixel_values=enc["pixel_values"].to(m.device_str, m.dtype),
                image_grid_thw=enc["image_grid_thw"].to(m.device_str),
                max_new_tokens=6 * len(qs), do_sample=False,
            )
    return t


# --------------------------------------------------------------------------- #
def load_groups(items_path: str, split: str, min_q: int, rng: random.Random):
    items = [it for it in read_jsonl(items_path)
             if it["meta"].get("split") == split and it["intervention"] == "none"]
    by_img: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        by_img[it["image_path"]].append(it)
    groups = [(k, v) for k, v in by_img.items() if len(v) >= min_q]
    rng.shuffle(groups)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=paths.MODEL)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--parity_groups", type=int, default=60)
    ap.add_argument("--parity_n", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    m = VDM(args.model)
    m.processor.image_processor.max_pixels = args.max_pixels
    groups = load_groups(args.items, args.split, max(args.ns), rng)
    print(f"{len(groups)} images with >= {max(args.ns)} questions", flush=True)

    results: dict[str, object] = {"max_pixels": args.max_pixels,
                                  "torch": torch.__version__,
                                  "gpu": torch.cuda.get_device_name(0)}

    # ---------------------------- parity ---------------------------- #
    print("\n=== parity ===", flush=True)
    par_groups = load_groups(args.items, args.split, args.parity_n, random.Random(args.seed + 1))[: args.parity_groups]
    rows = []
    for gi, (img_path, its) in enumerate(par_groups):
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            qs = [{"qtype": it["qtype"], "instruction": it["instruction"],
                   "candidates": it["candidates"]} for it in its[: args.parity_n]]
            g = m.prepare_group(img, qs, "")
            ref = m.run_independent(g)
            for name, fn in (("independent_batch", m.run_independent_batch),
                             ("vision_cache", m.run_vision_cache),
                             ("vision_cache_batch", m.run_vision_cache_batch),
                             ("prefix_share", m.run_prefix_share),
                             ("prefix_share_batch", m.run_prefix_share_batch)):
                o = fn(g)
                for i in range(g.n_questions):
                    k = g.n_options[i]
                    za, zb = ref["lm_option_logits"][i, :k].float(), o["lm_option_logits"][i, :k].float()
                    pa, pb = torch.softmax(za, -1), torch.softmax(zb, -1)
                    srt = torch.sort(pa, descending=True).values
                    rows.append({
                        "path": name,
                        "d_logit": float((za - zb).abs().max()),
                        "d_prob": float((pa - pb).abs().max()),
                        "flip": int(za.argmax() != zb.argmax()),
                        "margin": float(srt[0] - srt[1]) if k > 1 else 1.0,
                    })
        if gi % 20 == 0:
            print(f"  parity group {gi}/{len(par_groups)}", flush=True)

    par: dict[str, dict] = {}
    for name in ("independent_batch", "vision_cache", "vision_cache_batch",
                 "prefix_share", "prefix_share_batch"):
        sub = [r for r in rows if r["path"] == name]
        dl = np.array([r["d_logit"] for r in sub])
        dp = np.array([r["d_prob"] for r in sub])
        fl = np.array([r["flip"] for r in sub])
        mg = np.array([r["margin"] for r in sub])
        near = mg < 0.1
        par[name] = {
            "n": len(sub),
            "d_logit_max": float(dl.max()), "d_logit_p99": float(np.percentile(dl, 99)),
            "d_logit_median": float(np.median(dl)),
            "d_prob_max": float(dp.max()), "d_prob_p99": float(np.percentile(dp, 99)),
            "d_prob_median": float(np.median(dp)),
            "argmax_flip_rate": float(fl.mean()),
            "n_near_boundary": int(near.sum()),
            "argmax_flip_rate_near_boundary": float(fl[near].mean()) if near.any() else None,
        }
        print(f"  {name:20s} max|dlogit|={par[name]['d_logit_max']:.2e} "
              f"max|dprob|={par[name]['d_prob_max']:.2e} flips={par[name]['argmax_flip_rate']:.4f}", flush=True)
    results["parity"] = par

    # ------------------------------ cost ------------------------------ #
    print("\n=== N sweep ===", flush=True)
    sweep = []
    paths = {
        "independent": lambda mm, im, q, c: timed_independent(mm, im, q, c),
        "independent_batch": lambda mm, im, q, c: timed_independent_batch(mm, im, q, c),
        "vision_cache": lambda mm, im, q, c: timed_vision_cache(mm, im, q, c),
        "vision_cache_batch": lambda mm, im, q, c: timed_vision_cache_batch(mm, im, q, c),
        "prefix_share": lambda mm, im, q, c: timed_prefix_share(mm, im, q, c, False),
        "prefix_share_batch": lambda mm, im, q, c: timed_prefix_share(mm, im, q, c, True),
        "independent_batch_gen": lambda mm, im, q, c: timed_independent_batch_gen(mm, im, q, c),
        "prefix_share_batch_gen": lambda mm, im, q, c: timed_prefix_share_batch_gen(mm, im, q, c),
        "gen_one_token_each": lambda mm, im, q, c: timed_generate(mm, im, q, c, "one_token_each"),
        "gen_compact_joint": lambda mm, im, q, c: timed_generate(mm, im, q, c, "compact_joint"),
    }
    for N in args.ns:
        usable = [g for g in groups if len(g[1]) >= N][: args.repeats + args.warmup]
        if len(usable) < args.warmup + 1:
            print(f"  N={N}: not enough images, skipped")
            continue
        for pname, fn in paths.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times, spans_acc = [], collections.defaultdict(list)
            for r, (img_path, its) in enumerate(usable):
                with Image.open(img_path) as im:
                    img = im.convert("RGB")
                    qs = [{"qtype": it["qtype"], "instruction": it["instruction"],
                           "candidates": it["candidates"]} for it in its[:N]]
                    # The reported total is synchronized wall time. Image
                    # decode and record extraction happen above this boundary;
                    # fn includes processor/tokenization and path transfers.
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    t = fn(m, img, qs, "")
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                if r < args.warmup:            # cold runs discarded, reported as warm-only
                    continue
                times.append(dt)
                for k, v in t.spans.items():
                    spans_acc[k].append(v)
            if not times:
                continue
            rec = {
                "N": N, "path": pname,
                "total_s_mean": float(np.mean(times)), "total_s_std": float(np.std(times)),
                "per_question_ms": float(np.mean(times) / N * 1000),
                "peak_gib": float(torch.cuda.max_memory_allocated() / 2**30),
                "phases_s": {k: float(np.mean(v)) for k, v in spans_acc.items()},
                "n_warm_repeats": len(times),
            }
            sweep.append(rec)
            print(f"  N={N:<3} {pname:20s} {rec['total_s_mean']*1000:8.1f} ms "
                  f"({rec['per_question_ms']:6.1f} ms/q)  peak={rec['peak_gib']:.2f} GiB", flush=True)
    results["sweep"] = sweep

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

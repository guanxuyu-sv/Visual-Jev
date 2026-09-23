"""Paired visual evidence interventions (plan 6.4).

The whole point of the pairing is to make *global* image statistics useless as a
sufficiency cue.  For every base judgment we try to emit a matched pair:

  relevant   : the degradation lands on the region the question depends on
  irrelevant : the *same* degradation, of the *same* total area, lands on a
               region verified not to carry the evidence

If a model's answerability score drops for the first and not the second, it has
learned a question-conditioned notion of evidence.  If it drops for both, it has
learned a blur detector -- which plan 13.2 lists as a kill criterion, so the
construction has to be able to expose it.

Every emitted variant carries a `verification` dict with the numbers that were
actually checked, so the audit in the paper reports measured coverage rather
than an assumption.
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
from typing import Sequence

from PIL import Image, ImageFilter

Box = tuple[int, int, int, int]      # x, y, w, h


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def clip_box(b: Box, W: int, H: int) -> Box:
    x, y, w, h = b
    x0, y0 = max(0, min(int(x), W - 1)), max(0, min(int(y), H - 1))
    x1, y1 = max(x0 + 1, min(int(x + w), W)), max(y0 + 1, min(int(y + h), H))
    return (x0, y0, x1 - x0, y1 - y0)


def box_area(b: Box) -> int:
    return max(0, b[2]) * max(0, b[3])


def intersect_area(a: Box, b: Box) -> int:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))


def union_area(boxes: Sequence[Box], W: int, H: int, step: int = 2) -> int:
    """Rasterised union area.  `step` subsamples for speed; exact enough for the
    area-matching tolerance we report."""
    if not boxes:
        return 0
    grid = set()
    for (x, y, w, h) in boxes:
        for gx in range(x, x + w, step):
            for gy in range(y, y + h, step):
                grid.add((gx // step, gy // step))
    return len(grid) * step * step


# --------------------------------------------------------------------------- #
# degradations
# --------------------------------------------------------------------------- #
def _apply_regions(img: Image.Image, boxes: Sequence[Box], mode: str, sigma: float) -> Image.Image:
    """Apply one degradation to a set of regions, leaving the rest untouched."""
    out = img.copy()
    for b in boxes:
        x, y, w, h = b
        if w <= 0 or h <= 0:
            continue
        patch = out.crop((x, y, x + w, y + h))
        if mode == "blur":
            patch = patch.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif mode == "occlude":
            # Mid-grey fill: removes evidence without inserting a new object and
            # without the "black rectangle" cue being trivially separable from
            # natural content statistics.
            patch = Image.new("RGB", patch.size, (128, 128, 128))
        elif mode == "downscale":
            # Shrink then restore the same processing size: destroys fine detail
            # (text, small parts) while keeping the region's layout.  This is an
            # *evidence* degradation, not a token-cost reduction (plan 8.3).
            k = max(1.0, sigma)
            small = patch.resize((max(1, int(w / k)), max(1, int(h / k))), Image.BILINEAR)
            patch = small.resize((w, h), Image.BILINEAR)
        else:
            raise ValueError(mode)
        out.paste(patch, (x, y))
    return out


# --------------------------------------------------------------------------- #
# picking a verified-irrelevant region of matched area
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class IrrelevantPick:
    boxes: list[Box]
    overlap_with_evidence: int
    overlap_with_samename: int
    area: int


def pick_irrelevant_region(
    W: int,
    H: int,
    evidence_boxes: Sequence[Box],
    forbidden_boxes: Sequence[Box],
    target_area: int,
    rng: random.Random,
    tries: int = 400,
    tol: float = 0.15,
) -> IrrelevantPick | None:
    """Find background boxes with total area ~= target_area and zero overlap
    with the evidence or with any object that could substitute for it.

    `forbidden_boxes` should include every object whose name matches a
    referenced object: occluding a *second* bird would change the answer to
    "how many birds", so such a region is not irrelevant (plan 6.4, "verify the
    background really is irrelevant").
    """
    if target_area <= 0:
        return None
    avoid = list(evidence_boxes) + list(forbidden_boxes)
    # Try a single box of matched area with a plausible aspect ratio first, then
    # fall back to two smaller boxes if the image is crowded.
    for n_parts in (1, 2):
        part_area = target_area / n_parts
        for _ in range(tries):
            boxes: list[Box] = []
            ok = True
            for _p in range(n_parts):
                ar = rng.uniform(0.6, 1.7)
                w = int(round(math.sqrt(part_area * ar)))
                h = int(round(part_area / max(1, w)))
                if w < 4 or h < 4 or w >= W or h >= H:
                    ok = False
                    break
                x = rng.randint(0, W - w)
                y = rng.randint(0, H - h)
                cand = (x, y, w, h)
                if any(intersect_area(cand, a) > 0 for a in avoid):
                    ok = False
                    break
                if any(intersect_area(cand, b) > 0 for b in boxes):
                    ok = False
                    break
                boxes.append(cand)
            if not ok or not boxes:
                continue
            area = sum(box_area(b) for b in boxes)
            if abs(area - target_area) / target_area <= tol:
                return IrrelevantPick(
                    boxes=boxes,
                    overlap_with_evidence=sum(
                        intersect_area(b, e) for b in boxes for e in evidence_boxes
                    ),
                    overlap_with_samename=sum(
                        intersect_area(b, f) for b in boxes for f in forbidden_boxes
                    ),
                    area=area,
                )
    return None


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def make_pair(
    src_image: str,
    out_dir: str,
    base_id: str,
    evidence_boxes: Sequence[Box],
    forbidden_boxes: Sequence[Box],
    mode: str = "occlude",
    sigma: float = 12.0,
    pad: float = 0.10,
    rng: random.Random | None = None,
    min_evidence_frac: float = 0.0015,
    max_evidence_frac: float = 0.45,
) -> dict | None:
    """Emit the relevant / irrelevant pair for one base judgment.

    Returns None when the pair cannot be built honestly -- no evidence box, the
    evidence covers most of the image (then "irrelevant of equal area" does not
    exist), or no verified background region was found.  Rejecting is preferable
    to emitting a pair whose control arm is contaminated.
    """
    rng = rng or random.Random(0)
    with Image.open(src_image) as im:
        im = im.convert("RGB")
        W, H = im.size

        ev = [clip_box(b, W, H) for b in evidence_boxes]
        ev = [b for b in ev if box_area(b) > 0]
        if not ev:
            return None
        # A small pad makes the occlusion actually cover the object rather than
        # leaving a readable rim.
        ev_pad = [
            clip_box((b[0] - pad * b[2], b[1] - pad * b[3], b[2] * (1 + 2 * pad), b[3] * (1 + 2 * pad)), W, H)
            for b in ev
        ]
        ev_area = union_area(ev_pad, W, H)
        frac = ev_area / float(W * H)
        if frac < min_evidence_frac or frac > max_evidence_frac:
            return None

        pick = pick_irrelevant_region(W, H, ev_pad, forbidden_boxes, ev_area, rng)
        if pick is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        rel_path = os.path.join(out_dir, f"{base_id}_rel_{mode}.jpg")
        irr_path = os.path.join(out_dir, f"{base_id}_irr_{mode}.jpg")
        _apply_regions(im, ev_pad, mode, sigma).save(rel_path, quality=95)
        _apply_regions(im, pick.boxes, mode, sigma).save(irr_path, quality=95)

    coverage = (ev_area - 0) / max(1, union_area(ev, W, H))
    return {
        "relevant_path": rel_path,
        "irrelevant_path": irr_path,
        "mode": mode,
        "sigma": sigma,
        "verification": {
            # measured, not assumed
            "evidence_area_px": ev_area,
            "irrelevant_area_px": pick.area,
            "area_ratio": pick.area / float(ev_area),
            "evidence_frac_of_image": frac,
            "evidence_coverage_of_boxes": coverage,
            "irrelevant_overlap_evidence_px": pick.overlap_with_evidence,
            "irrelevant_overlap_samename_px": pick.overlap_with_samename,
            "image_wh": [W, H],
            "n_evidence_boxes": len(ev),
            "n_irrelevant_boxes": len(pick.boxes),
        },
        "relevant_boxes": ev_pad,
        "irrelevant_boxes": pick.boxes,
    }


# --------------------------------------------------------------------------- #
# semantics-preserving variants (for the consistency term, plan 7.2)
# --------------------------------------------------------------------------- #
# Rule-based only, on purpose: GQA questions are template generated, so a
# prefix that adds no information cannot introduce new ambiguity.  A teacher
# model paraphrase would need its own logging and audit (plan 6.5) and is not
# needed for this term.
PARAPHRASE_PREFIXES = [
    "Looking at the image, {q}",
    "Based on what is visible in the image, {q}",
    "In this image, {q}",
    "From the image alone, {q}",
]


def paraphrase(question: str, rng: random.Random) -> str:
    q = question.strip()
    tmpl = rng.choice(PARAPHRASE_PREFIXES)
    body = q[0].lower() + q[1:] if q[:1].isupper() and not q[:2].isupper() else q
    return tmpl.format(q=body)


def permute_candidates(
    candidates: Sequence[str], label: int, rng: random.Random
) -> tuple[list[str], int, list[int]]:
    """Return a *different* ordering plus the mapping used to realign outputs."""
    n = len(candidates)
    order = list(range(n))
    for _ in range(20):
        rng.shuffle(order)
        if order != list(range(n)):
            break
    new = [candidates[i] for i in order]
    return new, order.index(label), order

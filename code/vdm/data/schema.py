"""Core record schema for Visual Decision Models (VDM).

One *base judgment* is a (image, shared_context, question) triple with a typed
output space.  Interventions derive *variants* of a base judgment that share
`base_id` and `parent_image_id` so that split isolation can be enforced on the
original image (plan 6.2).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


CLAIM_LABELS = ["supported", "contradicted", "not_determined"]

# Engineering cap on Choice slots (plan 5.1).  Items above this are rejected,
# never silently truncated.
K_MAX = 16


@dataclasses.dataclass
class Item:
    """A single decision instance handed to the model."""

    item_id: str
    base_id: str            # groups all variants of one base judgment
    source: str             # gqa | textvqa | snli_ve | ...
    task_family: str        # gqa_choice | gqa_verify | textvqa_choice | snli_ve_claim
    qtype: str              # "choice" | "claim"
    image_path: str
    parent_image_path: str  # the un-intervened image this was derived from
    parent_image_id: str    # dataset image id, split isolation key
    question_id: str
    shared_context: str     # public text prefix, may be ""
    instruction: str        # the question itself
    candidates: list[str] | None      # Choice only
    label: int | None                 # index into candidates, or CLAIM_LABELS
    answerable: int | None            # 1 = enough evidence, 0 = not
    decision_supervised: int          # 0 => mask the decision loss (plan 7.1)
    intervention: str                 # none | blur_relevant | occlude_relevant | ...
    intervention_chain: list[str]
    meta: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def stable_id(*parts: Any) -> str:
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


def write_jsonl(path: str, items: list[Item]) -> None:
    with open(path, "w") as fh:
        for it in items:
            fh.write(it.to_json() + "\n")


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

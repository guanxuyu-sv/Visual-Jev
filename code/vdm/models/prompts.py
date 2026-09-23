"""Prompt construction with an explicit, token-verified prefix/suffix split.

Everything downstream -- training, the four execution paths, the parity check --
depends on one invariant:

    tokenize(PREFIX + SUFFIX_i)[: len(tokenize(PREFIX))] == tokenize(PREFIX)

i.e. the shared prefix must be a *token-level* prefix of every full sequence.
BPE merges across the boundary would silently break that, so `split_ids`
asserts it rather than assuming it (plan 9.1, "candidate token boundaries ...
must be verified").
"""
from __future__ import annotations

from typing import Sequence

LETTERS = [chr(ord("A") + i) for i in range(26)]

SYSTEM = (
    "You are a visual decision model. You inspect the image and answer each "
    "question by choosing exactly one of the given options. You never explain."
)

CLAIM_OPTIONS = [
    "supported by the image",
    "contradicted by the image",
    "not determined by the image",
]


def prefix_text(shared_context: str = "") -> str:
    """System turn + user turn opener + the image + any public context.

    The image and the shared context are the only things every question in a
    batch has in common, so the prefix ends immediately after them.
    """
    ctx = f"\n{shared_context.strip()}" if shared_context.strip() else ""
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<|vision_start|><|image_pad|><|vision_end|>{ctx}"
    )


def suffix_text(instruction: str, options: Sequence[str], qtype: str = "choice") -> str:
    """Per-question continuation, ending at the readout position.

    The final token is the colon of "Answer:"; the decision is read from the
    hidden state there, and the LM-head baseline reads the next-token logits of
    " A"/" B"/... at the same position.  Both readouts therefore see exactly the
    same context, which is what makes B1 vs. B2 an isolated comparison.
    """
    lines = [f"\n\nQuestion: {instruction.strip()}"]
    if qtype == "claim":
        lines.append("Decide whether the statement above is:")
    lines.append("Options:")
    for i, o in enumerate(options):
        lines.append(f"{LETTERS[i]}. {o}")
    lines.append("Reply with the single letter of the correct option.")
    body = "\n".join(lines)
    return f"{body}<|im_end|>\n<|im_start|>assistant\nAnswer:"


def option_token_strings(n: int) -> list[str]:
    """The strings whose first token the LM-head baseline scores."""
    return [f" {LETTERS[i]}" for i in range(n)]


def split_ids(tokenizer, prefix: str, full: str) -> tuple[list[int], list[int]]:
    """Tokenize prefix and full, verify the prefix property, return both halves."""
    pre = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    fl = tokenizer(full, add_special_tokens=False)["input_ids"]
    if fl[: len(pre)] != pre:
        raise AssertionError(
            "prefix is not a token-level prefix of the full sequence; "
            "BPE merged across the boundary"
        )
    return pre, fl[len(pre) :]

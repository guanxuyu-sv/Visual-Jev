"""Dataset and collator.

The one structural requirement: paired items must land in the same batch, or
the ranking and consistency terms have nothing to compare.  `PairedSampler`
emits whole base judgments (original + its degraded and rephrased variants)
as indivisible blocks, then shuffles the blocks.

Everything else is ordinary right-padded batching.  Training uses independent
samples; the sharing is an inference-time property (plan 5.2).
"""
from __future__ import annotations

import collections
import random
from typing import Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from vdm.models.prompts import CLAIM_OPTIONS, prefix_text, suffix_text

CLAIM_LABELS = ["supported", "contradicted", "not_determined"]


class ItemDataset(Dataset):
    def __init__(self, items: Sequence[dict], with_unknown_slot: bool = False):
        self.items = list(items)
        self.with_unknown_slot = with_unknown_slot

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class PairedSampler(Sampler[list[int]]):
    """Batches built from whole base-judgment blocks."""

    def __init__(self, items: Sequence[dict], batch_size: int, seed: int = 0, drop_last: bool = True):
        self.blocks: list[list[int]] = []
        by_base: dict[str, list[int]] = collections.defaultdict(list)
        for i, it in enumerate(items):
            by_base[it["base_id"]].append(i)
        for _, idxs in by_base.items():
            # A base judgment with more variants than one batch is split; the
            # original is kept in the first block so a pair always has its anchor.
            for j in range(0, len(idxs), batch_size):
                self.blocks.append(idxs[j : j + batch_size])
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        blocks = list(self.blocks)
        rng.shuffle(blocks)
        batch: list[int] = []
        for b in blocks:
            if len(batch) + len(b) > self.batch_size and batch:
                yield batch
                batch = []
            batch.extend(b)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(b) for b in self.blocks)
        return max(1, total // self.batch_size)


class Collator:
    """Tokenize, pad right, and record where each sample's readout sits."""

    def __init__(self, processor, tokenizer, image_token_id: int, k_max: int = 16,
                 with_unknown_slot: bool = False, max_len: int = 1400):
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.k_max = k_max
        self.with_unknown_slot = with_unknown_slot
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict:
        texts, images = [], []
        metas = []
        for it in batch:
            opts = CLAIM_OPTIONS if it["qtype"] == "claim" else list(it["candidates"])
            if self.with_unknown_slot and it["qtype"] == "choice":
                opts = opts + ["none of the above / cannot be determined"]
            txt = prefix_text(it["shared_context"]) + suffix_text(it["instruction"], opts, it["qtype"])
            texts.append(txt)
            with Image.open(it["image_path"]) as im:
                images.append(im.convert("RGB"))
            metas.append((it, len(opts)))

        enc = self.processor(text=texts, images=images, return_tensors="pt",
                             padding=True, padding_side="right")
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        readout = attn.sum(-1) - 1                       # last real token per row

        n = len(batch)
        labels = torch.full((n,), -100, dtype=torch.long)
        ans = torch.full((n,), -1.0)
        dec_mask = torch.zeros(n, dtype=torch.bool)
        n_opts, qtypes = [], []
        for i, (it, k) in enumerate(metas):
            n_opts.append(k)
            qtypes.append(it["qtype"])
            if it["decision_supervised"] and it["label"] is not None:
                labels[i] = it["label"]
                dec_mask[i] = True
            elif self.with_unknown_slot and it["qtype"] == "choice":
                # B4: instead of masking, the removed-evidence item is taught to
                # pick the explicit unknown slot.
                labels[i] = k - 1
                dec_mask[i] = True
            if it["answerable"] is not None:
                ans[i] = float(it["answerable"])

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
            "readout_index": readout,
            "labels": labels,
            "answerable": ans,
            "decision_mask": dec_mask,
            "n_options": n_opts,
            "qtypes": qtypes,
            "base_ids": [it["base_id"] for it, _ in metas],
            "interventions": [it["intervention"] for it, _ in metas],
            # realignment map for permuted-candidate variants; None elsewhere
            "permutations": [it["meta"].get("permutation") for it, _ in metas],
            "item_ids": [it["item_id"] for it, _ in metas],
        }

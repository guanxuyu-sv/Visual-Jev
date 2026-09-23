"""The visual decision model and its four execution paths.

Model
-----
A Qwen3-VL backbone plus three small heads read off the hidden state at the
readout position (the ":" of "Answer:"):

  choice_head  -> K_MAX slot logits, masked to the K options actually present
  claim_head   -> 3 logits (supported / contradicted / not-determined)
  answer_head  -> 1 logit, "is there enough evidence in this observation"

The LM-head readout of the untouched backbone is kept available at the same
position, so the "does a decision head help at all" comparison (B1 vs. B2) does
not confound the readout change with the training change (plan 5.1).

Execution paths
---------------
All four produce the *same* logits by construction; `bench/parity.py` measures
how well that holds numerically.

  independent        image re-encoded and full sequence re-run per question
  vision_cache       image encoded once, full language sequence re-run
  prefix_share       prefix prefilled once, KV forked, suffixes run serially
  prefix_share_batch prefix prefilled once, suffixes run as one padded batch

Qwen3-VL specifics that shape this implementation:
* DeepStack re-injects visual features into *text* layers 0..2, but only at
  image-token positions, which all live in the shared prefix.  The prefix KV
  therefore captures the whole DeepStack contribution and suffix forwards must
  pass `deepstack_visual_embeds=None`.
* M-RoPE position ids are 3-D and image tokens consume a 2-D span, so suffix
  position ids cannot be derived from a token count.  We compute them once for
  the full sequence with `get_rope_index` and slice, which keeps the shared and
  independent paths on identical positions.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from .prompts import (
    CLAIM_OPTIONS,
    LETTERS,
    option_token_strings,
    prefix_text,
    split_ids,
    suffix_text,
)

K_MAX = 16
CLAIM_CLASSES = 3


# --------------------------------------------------------------------------- #
# prepared inputs
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PreparedGroup:
    """One image + shared context, with N questions hanging off it."""

    prefix_ids: torch.Tensor                 # [P]
    suffix_ids: list[torch.Tensor]           # N x [S_i]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: list[torch.Tensor]         # N x [3, P+S_i]
    qtypes: list[str]
    n_options: list[int]
    image_token_id: int

    @property
    def n_questions(self) -> int:
        return len(self.suffix_ids)

    @property
    def prefix_len(self) -> int:
        return int(self.prefix_ids.shape[0])


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class DecisionHeads(nn.Module):
    """Task-typed heads shared across datasets (plan 5.1: one head per *type*,
    never one per dataset)."""

    def __init__(self, hidden: int, k_max: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.choice = nn.Linear(hidden, k_max)
        self.claim = nn.Linear(hidden, CLAIM_CLASSES)
        self.answerable = nn.Linear(hidden, 1)
        for lin in (self.choice, self.claim, self.answerable):
            nn.init.normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.drop(self.norm(h))
        return {
            "choice": self.choice(h),
            "claim": self.claim(h),
            "answerable": self.answerable(h).squeeze(-1),
        }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class VDM(nn.Module):
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        with_heads: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.device_str = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation=attn_implementation
        ).to(device)
        self.backbone.eval()
        hidden = self.config.text_config.hidden_size
        self.heads = DecisionHeads(hidden).to(device=device, dtype=torch.float32) if with_heads else None
        self.image_token_id = self.config.image_token_id
        # only the answer-SFT baseline needs the full vocabulary head output
        self.keep_full_lm_logits = False
        self._option_ids = self._build_option_ids()

    @property
    def option_token_ids(self) -> torch.Tensor:
        """Vocabulary ids of " A", " B", ... -- the answer-SFT targets."""
        return self._option_ids

    # ------------------------- backbone unwrapping ------------------------- #
    @property
    def vl(self):
        """The Qwen3VLForConditionalGeneration, whether or not PEFT wrapped it.

        `get_peft_model` inserts PeftModel -> LoraModel in front of the model,
        so `self.backbone.model` stops meaning what it meant before LoRA was
        attached.  Everything below goes through here instead.
        """
        b = self.backbone
        get_base = getattr(b, "get_base_model", None)
        return get_base() if callable(get_base) else b

    @property
    def vl_model(self):
        """The Qwen3VLModel (vision tower + language tower, no LM head)."""
        return self.vl.model

    @property
    def lm_head(self):
        return self.vl.lm_head

    @property
    def dtype(self):
        return self.vl.dtype

    # ---------------- option token ids for the LM-head readout -------------- #
    def _build_option_ids(self) -> torch.Tensor:
        """First token id of " A", " B", ... -- verified to be single tokens."""
        ids = []
        for s in option_token_strings(K_MAX):
            t = self.tokenizer(s, add_special_tokens=False)["input_ids"]
            if len(t) != 1:
                raise AssertionError(f"option string {s!r} is not a single token: {t}")
            ids.append(t[0])
        return torch.tensor(ids, device=self.device_str)

    # ------------------------------ preparation ---------------------------- #
    def prepare_group(
        self,
        image,
        questions: Sequence[dict],
        shared_context: str = "",
    ) -> PreparedGroup:
        """Tokenize one image with N questions into a prefix + N suffixes.

        `questions` entries need `instruction`, `candidates` (or qtype "claim")
        and `qtype`.
        """
        pre_txt = prefix_text(shared_context)
        # The processor expands <|image_pad|> to the real number of visual
        # tokens; we run it on the prefix text alone so the expansion is shared.
        enc = self.processor(text=[pre_txt], images=[image], return_tensors="pt")
        prefix_ids = enc["input_ids"][0]
        pixel_values = enc["pixel_values"]
        image_grid_thw = enc["image_grid_thw"]

        suffix_ids: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        qtypes: list[str] = []
        n_options: list[int] = []
        for q in questions:
            opts = CLAIM_OPTIONS if q["qtype"] == "claim" else list(q["candidates"])
            if len(opts) > K_MAX:
                raise ValueError(f"{len(opts)} options exceeds K_MAX={K_MAX}; reject, do not truncate")
            suf_txt = suffix_text(q["instruction"], opts, q["qtype"])
            suf = self.tokenizer(suf_txt, add_special_tokens=False)["input_ids"]
            suffix_ids.append(torch.tensor(suf, dtype=torch.long))
            qtypes.append(q["qtype"])
            n_options.append(len(opts))

            full = torch.cat([prefix_ids, suffix_ids[-1]])[None]           # [1, P+S]
            mm = (full == self.image_token_id).to(torch.int32)
            pos, _ = self.vl_model.get_rope_index(
                full.to(self.device_str),
                mm_token_type_ids=mm.to(self.device_str),
                image_grid_thw=image_grid_thw.to(self.device_str),
            )
            position_ids.append(pos[:, 0])                                  # [3, P+S]

        return PreparedGroup(
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            qtypes=qtypes,
            n_options=n_options,
            image_token_id=self.image_token_id,
        )

    # ------------------------------ vision --------------------------------- #
    def encode_vision(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        out = self.vl_model.get_image_features(
            pixel_values.to(self.device_str, self.dtype),
            image_grid_thw.to(self.device_str),
            return_dict=True,
        )
        image_embeds = torch.cat(out.pooler_output, dim=0)
        return image_embeds, out.deepstack_features

    def _embed(self, ids: torch.Tensor, image_embeds: torch.Tensor | None):
        """Token embeddings with visual embeddings scattered into image slots."""
        emb = self.vl_model.get_input_embeddings()(ids)
        mask = ids == self.image_token_id
        if image_embeds is not None and mask.any():
            emb = emb.clone()
            emb[mask] = image_embeds.to(emb.dtype)
        return emb, mask

    # ------------------------------ readout -------------------------------- #
    def _readout(self, hidden_last: torch.Tensor, qtypes: Sequence[str], n_options: Sequence[int]):
        """Turn the readout-position hidden state into logits for each output type."""
        out: dict[str, torch.Tensor] = {}
        lm_logits = self.lm_head(hidden_last.to(self.dtype))
        out["lm_option_logits"] = lm_logits.index_select(-1, self._option_ids).float()
        if self.keep_full_lm_logits:
            # answer-SFT trains the LM head over the whole vocabulary, not over
            # a pre-restricted option set; the restriction belongs at readout.
            out["lm_logits"] = lm_logits
        if self.heads is not None:
            h = self.heads(hidden_last.float())
            out.update({k: v.float() for k, v in h.items()})
        # Mask slots that this item does not actually have.
        n = hidden_last.shape[0]
        slot_mask = torch.zeros(n, K_MAX, dtype=torch.bool, device=hidden_last.device)
        for i, k in enumerate(n_options):
            slot_mask[i, :k] = True
        out["slot_mask"] = slot_mask
        out["qtypes"] = list(qtypes)
        return out

    # =============================== paths ================================= #
    @torch.no_grad()
    def run_independent(self, g: PreparedGroup) -> dict[str, torch.Tensor]:
        """Full repeat: the image is re-encoded for every question."""
        hs = []
        for i in range(g.n_questions):
            full = torch.cat([g.prefix_ids, g.suffix_ids[i]])[None].to(self.device_str)
            mm = (full == self.image_token_id).to(torch.int32)
            out = self.vl_model(
                input_ids=full,
                pixel_values=g.pixel_values.to(self.device_str, self.dtype),
                image_grid_thw=g.image_grid_thw.to(self.device_str),
                mm_token_type_ids=mm,
                position_ids=g.position_ids[i][:, None].to(self.device_str),
                use_cache=False,
            )
            hs.append(out.last_hidden_state[0, -1])
        return self._readout(torch.stack(hs), g.qtypes, g.n_options)

    @torch.no_grad()
    def run_vision_cache(self, g: PreparedGroup) -> dict[str, torch.Tensor]:
        """Vision encoder runs once; the language model still re-reads the prefix."""
        image_embeds, deepstack = self.encode_vision(g.pixel_values, g.image_grid_thw)
        hs = []
        for i in range(g.n_questions):
            full = torch.cat([g.prefix_ids, g.suffix_ids[i]]).to(self.device_str)
            emb, vmask = self._embed(full, image_embeds)
            out = self.vl_model.language_model(
                inputs_embeds=emb[None],
                position_ids=g.position_ids[i][:, None].to(self.device_str),
                visual_pos_masks=vmask[None],
                deepstack_visual_embeds=deepstack,
                use_cache=False,
            )
            hs.append(out.last_hidden_state[0, -1])
        return self._readout(torch.stack(hs), g.qtypes, g.n_options)

    def _pad_full(self, g: PreparedGroup):
        """Left-pad the N full sequences so the readout is the last column."""
        n = g.n_questions
        fulls = [torch.cat([g.prefix_ids, s]) for s in g.suffix_ids]
        lens = [int(f.shape[0]) for f in fulls]
        S = max(lens)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids = torch.full((n, S), pad_id, dtype=torch.long)
        pos = torch.zeros(3, n, S, dtype=torch.long)
        attn = torch.zeros(n, S, dtype=torch.long)
        for i, L in enumerate(lens):
            ids[i, S - L:] = fulls[i]
            pos[:, i, S - L:] = g.position_ids[i]
            attn[i, S - L:] = 1
        return ids, pos, attn

    @torch.no_grad()
    def run_independent_batch(self, g: PreparedGroup) -> dict[str, torch.Tensor]:
        """No sharing at all, but the N questions run as one batch.

        This is the control that separates the two things the shared path does
        at once. Against `run_independent` it isolates batching; against
        `run_prefix_share_batch`, which batches the same way, what remains is
        the effect of sharing the prefix. The image really is re-encoded N
        times here and the prefix really is recomputed on every row.
        """
        ids, pos, attn = self._pad_full(g)
        n = g.n_questions
        px = g.pixel_values.to(self.device_str, self.dtype)
        px = px.repeat(n, 1) if px.dim() == 2 else px.repeat(n, *([1] * (px.dim() - 1)))
        grid = g.image_grid_thw.to(self.device_str).repeat(n, 1)
        out = self.vl_model(
            input_ids=ids.to(self.device_str),
            pixel_values=px,
            image_grid_thw=grid,
            mm_token_type_ids=(ids == self.image_token_id).to(torch.int32).to(self.device_str),
            position_ids=pos.to(self.device_str),
            attention_mask=attn.to(self.device_str),
            use_cache=False,
        )
        return self._readout(out.last_hidden_state[:, -1], g.qtypes, g.n_options)

    @torch.no_grad()
    def run_vision_cache_batch(self, g: PreparedGroup) -> dict[str, torch.Tensor]:
        """Vision encoded once, language side still recomputed per question but
        batched. Isolates batching given visual reuse, without prefix sharing."""
        image_embeds, deepstack = self.encode_vision(g.pixel_values, g.image_grid_thw)
        ids, pos, attn = self._pad_full(g)
        ids = ids.to(self.device_str)
        emb = self.vl_model.get_input_embeddings()(ids)
        vmask = ids == self.image_token_id
        n_img = int(vmask[0].sum())
        emb = emb.clone()
        emb[vmask] = image_embeds[:n_img].repeat(g.n_questions, 1).to(emb.dtype)
        ds = [d[:n_img].repeat(g.n_questions, 1) for d in deepstack] if deepstack else None
        out = self.vl_model.language_model(
            inputs_embeds=emb,
            position_ids=pos.to(self.device_str),
            attention_mask=attn.to(self.device_str),
            visual_pos_masks=vmask,
            deepstack_visual_embeds=ds,
            use_cache=False,
        )
        return self._readout(out.last_hidden_state[:, -1], g.qtypes, g.n_options)

    @torch.no_grad()
    def _prefill_prefix(self, g: PreparedGroup):
        image_embeds, deepstack = self.encode_vision(g.pixel_values, g.image_grid_thw)
        ids = g.prefix_ids.to(self.device_str)
        emb, vmask = self._embed(ids, image_embeds)
        cache = DynamicCache()
        pos = g.position_ids[0][:, :g.prefix_len][:, None].to(self.device_str)
        self.vl_model.language_model(
            inputs_embeds=emb[None],
            position_ids=pos,
            past_key_values=cache,
            visual_pos_masks=vmask[None],
            deepstack_visual_embeds=deepstack,
            use_cache=True,
        )
        return cache

    @staticmethod
    def _fork(cache: DynamicCache, batch: int = 1) -> DynamicCache:
        """Copy the prefix KV so a branch can extend it without mutating it.

        The copy is a real cost of the shared path and the benchmark reports it
        separately rather than folding it into the prefill (plan 5.2).
        """
        new = DynamicCache()
        for layer_idx, layer in enumerate(cache.layers):
            k, v = layer.keys, layer.values
            if batch != 1:
                k = k.expand(batch, -1, -1, -1)
                v = v.expand(batch, -1, -1, -1)
            new.update(k.contiguous().clone(), v.contiguous().clone(), layer_idx)
        return new

    @torch.no_grad()
    def run_prefix_share(self, g: PreparedGroup) -> dict[str, torch.Tensor]:
        """Prefill once, fork the KV per question, run suffixes one at a time."""
        base = self._prefill_prefix(g)
        hs = []
        for i in range(g.n_questions):
            cache = self._fork(base, 1)
            ids = g.suffix_ids[i].to(self.device_str)
            emb, _ = self._embed(ids, None)
            pos = g.position_ids[i][:, g.prefix_len :][:, None].to(self.device_str)
            out = self.vl_model.language_model(
                inputs_embeds=emb[None],
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            )
            hs.append(out.last_hidden_state[0, -1])
        return self._readout(torch.stack(hs), g.qtypes, g.n_options)

    @torch.no_grad()
    def run_prefix_share_batch(self, g: PreparedGroup) -> dict[str, torch.Tensor]:
        """Prefill once, then all suffixes as a single padded batch.

        Suffixes are *left*-padded so that the readout position is the last
        column for every row, and the pad columns are masked out of attention.
        """
        base = self._prefill_prefix(g)
        n = g.n_questions
        lens = [int(s.shape[0]) for s in g.suffix_ids]
        S = max(lens)
        P = g.prefix_len
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        ids = torch.full((n, S), pad_id, dtype=torch.long)
        pos = torch.zeros(3, n, S, dtype=torch.long)
        attn = torch.zeros(n, P + S, dtype=torch.long)
        attn[:, :P] = 1
        for i, L in enumerate(lens):
            ids[i, S - L :] = g.suffix_ids[i]
            pos[:, i, S - L :] = g.position_ids[i][:, P:]
            attn[i, P + S - L :] = 1

        cache = self._fork(base, n)
        emb, _ = self._embed(ids.to(self.device_str), None)
        out = self.vl_model.language_model(
            inputs_embeds=emb,
            position_ids=pos.to(self.device_str),
            attention_mask=attn.to(self.device_str),
            past_key_values=cache,
            use_cache=True,
        )
        return self._readout(out.last_hidden_state[:, -1], g.qtypes, g.n_options)

    # ---------------------------- training path ---------------------------- #
    def forward_train(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        readout_index: torch.Tensor,
        qtypes: Sequence[str],
        n_options: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Independent-sample batching for training (plan 5.2 allows this; the
        sharing is an inference-time property)."""
        out = self.vl_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=(input_ids == self.image_token_id).to(torch.int32),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        h = out.last_hidden_state
        idx = readout_index.to(h.device)
        hidden_last = h[torch.arange(h.shape[0], device=h.device), idx]
        return self._readout(hidden_last, qtypes, n_options)

    # --------------------------- misc utilities ---------------------------- #
    @contextlib.contextmanager
    def trainable(self):
        was = self.backbone.training
        self.backbone.train()
        try:
            yield
        finally:
            self.backbone.train(was)

"""Train one variant of the visual decision model.

Variants (plan 8.1).  They differ only in which loss terms are active and which
items carry a usable label; the backbone, the trainable parameter set, the
optimiser and the step budget are held fixed so the comparison isolates the
training signal rather than the capacity.

  b2  decision CE only, on items that have a valid decision label
  b4  same data as the full method, but the evidence-degraded items are taught
      an explicit "cannot be determined" slot instead of being masked
  b5  b2 + answerability BCE
  m   b5 + sufficiency ranking + semantic consistency
"""
from __future__ import annotations

import argparse, dataclasses, json, math, os, random, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vdm import paths
from vdm.data.schema import read_jsonl
from vdm.models.vdm_model import VDM
from vdm.training import checkpoint
from vdm.training.dataset import Collator, ItemDataset, PairedSampler
from vdm.training.losses import total_loss

VARIANTS = {
    #            lambda_a  lambda_r  lambda_c  unknown_slot  use_masked_items
    "b2": dict(lambda_a=0.0, lambda_r=0.0, lambda_c=0.0, unknown_slot=False, use_masked=False),
    "b4": dict(lambda_a=0.0, lambda_r=0.0, lambda_c=0.0, unknown_slot=True, use_masked=True),
    "b5": dict(lambda_a=1.0, lambda_r=0.0, lambda_c=0.0, unknown_slot=False, use_masked=True),
    "m": dict(lambda_a=1.0, lambda_r=0.5, lambda_c=0.1, unknown_slot=False, use_masked=True),
    # A2: the same data and budget, trained the ordinary way -- next-token CE on
    # the answer label through the LM head. Isolates the decision head from the
    # extra training it sits on top of.
    "a2": dict(lambda_a=0.0, lambda_r=0.0, lambda_c=0.0, unknown_slot=False,
               use_masked=False, readout="lm"),
}


@dataclasses.dataclass
class Cfg:
    lambda_a: float = 1.0
    lambda_r: float = 0.5
    lambda_c: float = 0.1
    rank_margin: float = 1.0
    readout: str = "head"          # "head" = decision head, "lm" = answer SFT
    option_token_ids: object = None


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(args) -> VDM:
    m = VDM(args.model, attn_implementation=args.attn)
    # Vision encoder stays frozen for the first round; the limitation is stated
    # rather than hidden (plan 7.3).
    for p in m.backbone.parameters():
        p.requires_grad_(False)
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # language tower only; the vision tower is excluded on purpose
            exclude_modules=r".*visual.*",
        )
        m.backbone = get_peft_model(m.backbone, lc)
    for p in m.heads.parameters():
        p.requires_grad_(True)
    if args.grad_ckpt:
        m.backbone.gradient_checkpointing_enable()
        m.backbone.enable_input_require_grads()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--train_items", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=paths.MODEL)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=200704)   # 448x448 -> 196 visual tokens
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--profile_only", type=int, default=0)
    # Overrides so a weight sweep does not need a new variant name.
    ap.add_argument("--lambda_a", type=float, default=None)
    ap.add_argument("--lambda_r", type=float, default=None)
    ap.add_argument("--lambda_c", type=float, default=None)
    ap.add_argument("--tag", default=None, help="label recorded in the checkpoint config")
    args = ap.parse_args()

    set_seed(args.seed)
    spec = VARIANTS[args.variant]
    cfg = Cfg(
        lambda_a=spec["lambda_a"] if args.lambda_a is None else args.lambda_a,
        lambda_r=spec["lambda_r"] if args.lambda_r is None else args.lambda_r,
        lambda_c=spec["lambda_c"] if args.lambda_c is None else args.lambda_c,
    )
    cfg.readout = spec.get("readout", "head")
    print(f"lambdas: a={cfg.lambda_a} r={cfg.lambda_r} c={cfg.lambda_c} readout={cfg.readout}", flush=True)

    items = []
    for f in args.train_items:
        items.extend(read_jsonl(f))
    items = [it for it in items if it["meta"].get("split") == "train"]
    if not spec["use_masked"]:
        # A plain CE baseline may only use labels its own definition admits.
        items = [it for it in items if it["decision_supervised"]]
    print(f"variant={args.variant}  train items={len(items)}", flush=True)

    m = build_model(args)
    m.processor.image_processor.max_pixels = args.max_pixels
    cfg.option_token_ids = m.option_token_ids
    if cfg.readout == "lm":
        m.keep_full_lm_logits = True
        # no decision head is trained in this baseline
        for p_ in m.heads.parameters():
            p_.requires_grad_(False)
    trainable = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_tr/1e6:.2f}M", flush=True)

    ds = ItemDataset(items)
    sampler = PairedSampler(items, args.batch_size, seed=args.seed)
    coll = Collator(m.processor, m.tokenizer, m.image_token_id,
                    with_unknown_slot=spec["unknown_slot"])
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=coll,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    head_params = list(m.heads.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for p in trainable if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": head_params, "lr": args.head_lr}],
        weight_decay=0.0, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train_log.jsonl"), "w")
    m.backbone.train()
    m.heads.train()

    step = 0
    t0 = time.time()
    running: dict[str, float] = {}
    epoch = 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for batch in dl:
            if step >= args.steps:
                break
            scale = lr_at(step)
            for g, base in zip(opt.param_groups, (args.lr, args.head_lr)):
                g["lr"] = base * scale

            out = m.forward_train(
                pixel_values=batch["pixel_values"].to(m.device_str, m.backbone.dtype),
                image_grid_thw=batch["image_grid_thw"].to(m.device_str),
                input_ids=batch["input_ids"].to(m.device_str),
                position_ids=None,
                attention_mask=batch["attention_mask"].to(m.device_str),
                readout_index=batch["readout_index"],
                qtypes=batch["qtypes"], n_options=batch["n_options"],
            )
            loss, parts = total_loss(out, batch, cfg)
            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            step += 1
            if step % args.log_every == 0:
                avg = {k: v / args.log_every for k, v in running.items()}
                running = {}
                rec = {"step": step, "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0,
                       "sec_per_step": (time.time() - t0) / step,
                       "peak_gib": torch.cuda.max_memory_allocated() / 2**30, **avg}
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if args.profile_only and step >= args.profile_only:
                print("PROFILE_DONE", flush=True)
                return

    checkpoint.save(m, args.out, cfg={**vars(args), **spec, "trainable_params": n_tr})
    print(f"saved -> {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

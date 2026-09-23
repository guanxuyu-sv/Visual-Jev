"""Save and load exactly what training changed: the heads and the LoRA deltas."""
from __future__ import annotations

import json, os

import torch


def save(model, path: str, cfg: dict | None = None) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(model.heads.state_dict(), os.path.join(path, "heads.pt"))
    peft = getattr(model.backbone, "peft_config", None)
    if peft is not None:
        model.backbone.save_pretrained(os.path.join(path, "lora"))
    if cfg is not None:
        json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=1)


def load_into(model, path: str) -> None:
    heads = os.path.join(path, "heads.pt")
    if os.path.exists(heads):
        sd = torch.load(heads, map_location="cpu")
        model.heads.load_state_dict(sd)
        model.heads.to(model.device_str)
    lora = os.path.join(path, "lora")
    if os.path.isdir(lora):
        from peft import PeftModel
        model.backbone = PeftModel.from_pretrained(model.backbone, lora)
        model.backbone.eval()

"""Score supplied visual choices with the released Hugging Face adapter.

Run from the repository root:
    python code/examples/quickstart.py --image image.jpg \
        --question "What animal is in the image?" \
        --choices cat dog bird other
"""
from __future__ import annotations

import argparse
import os
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to a local image")
    parser.add_argument("--question", help="one question to ask about the image")
    parser.add_argument("--choices", nargs="+", help="2 to 16 candidate answers")
    parser.add_argument("--context", default="", help="public context shared by all questions")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft")
    args = parser.parse_args()
    if (args.question is None) != (args.choices is None):
        parser.error("--question and --choices must be supplied together")
    if args.choices is not None and not 2 <= len(args.choices) <= 16:
        parser.error("--choices requires 2 to 16 answers")

    import torch
    from peft import PeftModel
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vdm.models.vdm_model import VDM

    if args.question is not None:
        questions = [{"qtype": "choice", "instruction": args.question,
                      "candidates": args.choices}]
    else:
        questions = [
            {"qtype": "choice", "instruction": "Is there a person in the image?",
             "candidates": ["yes", "no"]},
            {"qtype": "choice", "instruction": "Is the scene indoors?",
             "candidates": ["yes", "no"]},
        ]

    model = VDM(args.model, with_heads=False)
    model.processor.image_processor.max_pixels = 200704  # paper's visual budget
    model.backbone = PeftModel.from_pretrained(model.backbone, args.adapter).eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    with torch.inference_mode():
        group = model.prepare_group(image, questions, shared_context=args.context)
        run = model.run_independent if len(questions) == 1 else model.run_prefix_share_batch
        output = run(group)

    for i, question in enumerate(questions):
        logits = output["lm_option_logits"][i, :group.n_options[i]]
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        print(question["instruction"])
        for answer, probability in zip(question["candidates"], probs):
            print(f"  {answer}: {probability:.3f}")
        print(f"  prediction: {question['candidates'][max(range(len(probs)), key=probs.__getitem__)]}")


if __name__ == "__main__":
    main()

"""Score supplied visual choices with the released Hugging Face adapter.

Run from the repository root:
    python code/examples/quickstart.py --image image.jpg \
        --question "What animal is in the image?" \
        --choices cat dog bird other
"""
from __future__ import annotations

import argparse
import json
import os
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to a local image")
    parser.add_argument("--question", help="one question to ask about the image")
    parser.add_argument("--choices", nargs="+", help="2 to 16 candidate answers")
    parser.add_argument("--context", default="", help="public context shared by all questions")
    parser.add_argument("--request-file", help="JSON request with shared state and named questions")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft")
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto",
                        help="auto selects CUDA first, then Apple MPS")
    parser.add_argument("--max-pixels", type=int, default=200704,
                        help="image pixel budget; lower it if unified memory is tight")
    args = parser.parse_args()
    if (args.question is None) != (args.choices is None):
        parser.error("--question and --choices must be supplied together")
    if args.request_file and (args.question is not None or args.context):
        parser.error("--request-file cannot be combined with --question, --choices, or --context")
    if args.choices is not None and not 2 <= len(args.choices) <= 16:
        parser.error("--choices requires 2 to 16 answers")

    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from peft import PeftModel
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vdm.models.vdm_model import VDM

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            parser.error("no supported accelerator found; this example requires CUDA or Apple MPS")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no available CUDA device")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested, but Apple MPS is not available in this PyTorch installation")
    dtype = torch.float16 if device == "mps" else torch.bfloat16

    question_ids = []
    if args.request_file:
        try:
            with open(args.request_file, encoding="utf-8") as request_stream:
                request = json.load(request_stream)
            request_questions = request["questions"]
            if not isinstance(request_questions, dict) or not request_questions:
                raise ValueError("'questions' must be a non-empty object keyed by question ID")
            shared_context = request.get("state", "")
            if not isinstance(shared_context, str):
                shared_context = json.dumps(shared_context, ensure_ascii=False)
            questions = []
            for question_id, item in request_questions.items():
                if not isinstance(item, dict):
                    raise ValueError(f"question {question_id!r}: each question must be an object")
                if item.get("type") != "choice":
                    raise ValueError(f"question {question_id!r}: only type 'choice' is supported")
                criteria = item.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
                    raise ValueError(f"question {question_id!r}: criteria must contain 2 to 16 choices")
                instruction = item.get("instructions")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(f"question {question_id!r}: 'instructions' must be a non-empty string")
                candidates = []
                for option_id, description in criteria.items():
                    if description is None or description == "":
                        candidates.append(option_id)
                    elif isinstance(description, str):
                        candidates.append(f"{option_id}: {description}")
                    else:
                        raise ValueError(f"question {question_id!r}: choice descriptions must be strings or null")
                question_ids.append(list(criteria))
                questions.append({"qtype": "choice", "instruction": instruction,
                                  "candidates": candidates})
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"invalid --request-file: {exc}")
    elif args.question is not None:
        questions = [{"qtype": "choice", "instruction": args.question,
                      "candidates": args.choices}]
        question_ids = [args.choices]
        shared_context = args.context
    else:
        questions = [
            {"qtype": "choice", "instruction": "Is there a person in the image?",
             "candidates": ["yes", "no"]},
            {"qtype": "choice", "instruction": "Is the scene indoors?",
             "candidates": ["yes", "no"]},
        ]
        question_ids = [["yes", "no"], ["yes", "no"]]
        shared_context = args.context

    model = VDM(args.model, device=device, dtype=dtype, with_heads=False)
    print(f"Using {device} with {dtype}")
    model.processor.image_processor.max_pixels = args.max_pixels
    model.backbone = PeftModel.from_pretrained(model.backbone, args.adapter).eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    with torch.inference_mode():
        group = model.prepare_group(image, questions, shared_context=shared_context)
        run = model.run_independent if len(questions) == 1 else model.run_prefix_share_batch
        output = run(group)

    for i, question in enumerate(questions):
        logits = output["lm_option_logits"][i, :group.n_options[i]]
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        print(f"[{i + 1}] {question['instruction']}")
        for answer, probability in zip(question_ids[i], probs):
            print(f"  {answer}: {probability:.3f}")
        print(f"  prediction: {question_ids[i][max(range(len(probs)), key=probs.__getitem__)]}")


if __name__ == "__main__":
    main()

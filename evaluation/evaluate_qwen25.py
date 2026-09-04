
# cd /root/autodl-tmp/repos/Automingo-Seeing-the-Unseen

# python evaluate_qwen25.py \
#   --model /root/autodl-tmp/outputs/qwen2.5vl-automingo-lora-v2 \
#   --base_model /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
#   --dataset_dir /root/autodl-tmp/data/Automingo_dataset/data \
#   --mode both \
#   --max_samples 1055 \
#   --output results_lora_v2_1055_both.json





# -*- coding: utf-8 -*-
"""Lightweight Qwen2.5-VL evaluation entrypoint for the Automingo repo.

Official behavior kept byte-for-byte (libs/models.py Qwen branch + evaluate.py):
NCAP_SYSTEM_PROMPT, image_1..image_5 in temporal order, system + user chat
messages via apply_chat_template, AutoProcessor, temperature=0.2, top_p=0.9,
do_sample=True, max_new_tokens=200, the official decode/post-processing, the
MCQ option shuffle + GT index, and Lingo-Judge scoring. Scoring and adapter
merging come straight from the official libs.

Deliberate differences vs. the official repo:
  * validation data is read directly from local validation-*.parquet
    (no HF datasets / load_dataset / Arrow cache / DatasetWrapper)
  * Qwen2.5-VL only (no LLaVA/Qwen3-VL/OpenAI/Gemini/Anthropic paths)
  * convenience additions (NOT in the official repo):
      - --mode both (runs the MCQ and Lingo prompts per sample)
      - --base_model override for adapters whose recorded base path is
        unreachable on this machine
      - "lingo_metrics" aggregate in the output JSON

Usage (on the GPU server):
  python evaluate_qwen25.py --model /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
      --dataset_dir /root/autodl-tmp/data/Automingo_dataset/data
"""

from __future__ import annotations
import random
import re
import argparse
import datetime
import gc
import glob
import io
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional for this script
    tqdm = lambda iterable, **kwargs: iterable

# Official Automingo utilities — never re-implemented in this file.
# _evaluate_via_judge loads wayveai/Lingo-Judge per call (official behavior).
from libs.calculate_accuracy import calculate_accuracy
from libs.utils import _evaluate_via_judge, _to_pil
from libs.peft_merge import ensure_merged_if_lora_adapter


# ---- constants ----

# libs/models.py: NCAP_SYSTEM_PROMPT = open("system_prompt.txt").read().strip()
_NCAP_FILE = Path(__file__).resolve().parent / "system_prompt.txt"
NCAP_SYSTEM_PROMPT = _NCAP_FILE.read_text(encoding="utf-8").strip()

# libs/datasets.py DatasetWrapper defaults
IMAGE_COLS: Tuple[str, ...] = tuple(f"image_{i}" for i in range(1, 6))
DEFAULT_PAD_SIZE: Tuple[int, int] = (224, 224)

DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET_DIR = "/root/autodl-tmp/data/Automingo_dataset/data"


# libs/datasets.py DatasetWrapper._pad_to_five (a method; not importable standalone)
def _pad_to_five(images: List[Any]) -> List[Any]:
    if len(images) >= 5:
        return images[:5]
    size = None
    for im in images:
        if hasattr(im, "size"):
            size = im.size
            break
    if size is None:
        size = DEFAULT_PAD_SIZE
    return images + [Image.new("RGB", size, color=(0, 0, 0))] * (5 - len(images))


# evaluate.py: options shuffle + GT index + prompt
def build_mcq_prompt(question: str, distractors: List[str], completion: str) -> Tuple[str, int]:
    multi_choice = list(distractors)
    multi_choice.append(completion)
    random.shuffle(multi_choice)
    gt_index = multi_choice.index(completion) + 1
    prompt = (question + "\n\nOptions:\n"
              + "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(multi_choice))
              + "\n\nAnswer (just the option number):")
    return prompt, gt_index










def normalize_mcq_answer(text: str) -> str:
    """Normalize common MCQ response formats before official scoring."""
    text = (text or "").strip()

    # LoRA may output:
    # "answer: 4\nreasoning: ..."
    match = re.search(r"\banswer\s*:\s*([1-4])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    # Preserve already-correct official format:
    # "4"
    if re.fullmatch(r"[1-4]", text):
        return text

    # Unknown format: leave unchanged so official scorer
    # can still mark it as Not Processed.
    return text

















    
# ---- local parquet reader (the one data-layer simplification) ----

def load_validation_parquet(dataset_dir: str, max_samples: Optional[int] = None) -> Tuple[pa.Table, int]:
    """Read validation-*.parquet under dataset_dir (or a direct file/glob) with
    pyarrow — no HF datasets, no download, no Arrow cache, train never touched."""
    if os.path.isfile(dataset_dir) or any(ch in dataset_dir for ch in "*?["):
        files = sorted(glob.glob(dataset_dir))
    else:
        files = sorted(glob.glob(os.path.join(dataset_dir, "validation-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No validation-*.parquet files found under '{dataset_dir}'.")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    n_eval = len(table) if max_samples is None else min(len(table), int(max_samples))
    print(f"Loaded {len(table)} validation rows from {len(files)} parquet file(s); evaluating {n_eval}.")
    return table, n_eval


def _row_as_dict(table: pa.Table, idx: int) -> Dict[str, Any]:
    return {col: table.column(col)[idx].as_py() for col in table.column_names}


# Mirror of libs/datasets.py DatasetWrapper._preprocess for a single row.
def _preprocess_row(row: Dict[str, Any]) -> Tuple[str, List[str], str, List[Image.Image]]:
    question = (row.get("question") or "").strip()
    answer = (row.get("ground_truth_answer") or "").strip()
    reasoning = (row.get("ground_truth_reasoning") or "").strip()
    distractors = [(row.get(f"distractor_{i}") or "").strip() for i in (1, 2, 3)]
    if reasoning:
        completion = f"{answer}, {reasoning}" if answer else f"Reasoning:\n{reasoning}"
    else:
        completion = answer

    images: List[Any] = []
    for col in IMAGE_COLS:
        raw = row.get(col)
        if raw is None or raw == "":
            continue
        if isinstance(raw, dict):  # evaluate.py dict branch: {"image"|"bytes"|"path": ...}
            if "image" in raw:
                images.append(raw["image"])
            elif "bytes" in raw:
                images.append(Image.open(io.BytesIO(raw["bytes"])).convert("RGB"))
            elif "path" in raw:
                images.append(Image.open(raw["path"]).convert("RGB"))
            else:
                raise ValueError(f"Unknown image dict format: {raw.keys()}")
        else:
            images.append(raw)

    return question, distractors, completion, _pad_to_five(images)


# ---- LoRA merge (official libs.peft_merge, plus our base-path override) ----

def resolve_model_path(model_id: str, base_override: Optional[str] = None) -> str:
    """Official ensure_merged_if_lora_adapter(model_id, hf_dtype="bfloat16").

    base_override is our only extension: it temporarily re-points the adapter's
    adapter_config.json at a local base path (training-time paths are often
    unreachable on the eval server) and restores the file afterwards.
    """
    cfg_path = Path(model_id) / "adapter_config.json"
    if base_override and cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("base_model_name_or_path") != base_override:
            bak = cfg_path.with_suffix(".json.bak")
            cfg_path.replace(bak)
            cfg["base_model_name_or_path"] = base_override
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            try:
                return ensure_merged_if_lora_adapter(model_id, hf_dtype="bfloat16")
            finally:
                bak.replace(cfg_path)
    return ensure_merged_if_lora_adapter(model_id, hf_dtype="bfloat16")


# ---- official Qwen2.5-VL wrapper (libs/models.py HuggingFaceVLM, Qwen path only) ----

class Qwen25VLModel:
    def __init__(self, model_id: str, device: str = "cuda", trust_remote_code: bool = False):
        from transformers import AutoProcessor, AutoModelForImageTextToText, logging
        logging.set_verbosity_error()
        #self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)


        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=784,
            max_pixels=50176,
            trust_remote_code=trust_remote_code,
        )


        
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     trust_remote_code=trust_remote_code,
        #     token=os.getenv("HF_TOKEN"),
        # ).eval().to(device)
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     trust_remote_code=trust_remote_code,
        #     token=os.getenv("HF_TOKEN"),
        # ).eval()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
            token=os.getenv("HF_TOKEN"),
        ).eval()

        self.device = device

        

    def generate(
            self,
            prompt: str,
            images: List[Image.Image],
            max_new_tokens: int = 256,  # official default; evaluate.py passes 200
            temperature: float = 0.2,
            top_p: float = 0.9,
            do_sample: bool = True,
    ) -> str:
        pil_images = [_to_pil(im) for im in images]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": NCAP_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": im} for im in pil_images]
                                        + [{"type": "text", "text": prompt}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        out = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if prompt in out:  # official post-processing
            out = out.split(prompt, 1)[-1].strip()
        out = out.lower().split("assistant")[-1].strip()
        return out.strip()


# ---- entrypoint ----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL Automingo validation evaluation (official logic, local parquet input).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=None,
                        help="Model directory or HF repo id. A PEFT LoRA adapter directory "
                             "(adapter_model.safetensors + adapter_config.json) is merged first "
                             "via the official libs.peft_merge. Defaults to --base_model.")
    parser.add_argument("--base_model", default=None,
                        help="Optional local base model path. Used (1) as the model when --model "
                             "is omitted, and (2) to override the adapter config's "
                             "base_model_name_or_path when --model is an adapter whose recorded "
                             "base path is unreachable on this machine.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR,
                        help="Directory with validation-*.parquet files (or a direct file/glob).")
    parser.add_argument("--mode", choices=["mcq", "lingo", "both"], default="mcq",
                        help="mcq: official multi-choice path (settings.multi_choice_prompting: true). "
                             "lingo: official Lingo-Judge path. "
                             "both: convenience mode, NOT in the official repo (runs both prompts).")
    parser.add_argument("--max_samples", type=int, default=1055,
                        help="Validation rows to evaluate (official: select(range(1055))). 0 = all rows.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: vlm_results_<timestamp>_<mode>.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model or args.base_model or DEFAULT_BASE_MODEL
    base_override = args.base_model if (args.model and args.base_model) else None
    merged_id = resolve_model_path(model_path, base_override)
    print(f"Evaluating model: {merged_id}" + (" (merged from LoRA adapter)" if merged_id != model_path else ""))

    table, n_eval = load_validation_parquet(args.dataset_dir, args.max_samples or None)

    # Official evaluate.py: VLM(..., hf_dtype="bfloat16") -> HuggingFaceVLM(..., device="cuda")
    vlm = Qwen25VLModel(model_id=merged_id, device="cuda", trust_remote_code=False)

    model_name = os.path.basename(merged_id.rstrip("/\\")) or merged_id
    results = {"model": model_name, "model_id": merged_id, "provider": "hf",
               "mode": args.mode, "samples": []}

    gt_indexes: List[int] = []
    answers: List[str] = []

    for sample_idx in tqdm(range(n_eval), desc=f"Evaluating {model_name}"):
        question, distractors, completion, images = _preprocess_row(_row_as_dict(table, sample_idx))

        if args.mode in ("mcq", "both"):
            prompt, gt_index = build_mcq_prompt(question, distractors, completion)
        else:  # lingo: raw question (official multi_choice_prompting=False path)
            prompt, gt_index = question, None

        rec: Dict[str, Any] = {"sample_idx": sample_idx, "prompt": prompt, "gt_index": gt_index}
        try:
            # if args.mode in ("mcq", "both"):
            #     rec["answer"] = vlm.generate(prompt, images=images, max_new_tokens=200)
            #     gt_indexes.append(gt_index)
            #     answers.append(rec["answer"])

            if args.mode in ("mcq", "both"):
                rec["answer"] = vlm.generate(
                    prompt,
                    images=images,
                    max_new_tokens=200
                )
            
                rec["parsed_answer"] = normalize_mcq_answer(rec["answer"])
            
                gt_indexes.append(gt_index)
                answers.append(rec["parsed_answer"])



            
            if args.mode in ("lingo", "both"):
                answer_lingo = vlm.generate(question, images=images, max_new_tokens=200)
                _, score = _evaluate_via_judge(question, completion, answer_lingo)
                rec["answer" if args.mode == "lingo" else "answer_lingo"] = answer_lingo
                rec["lingo_judge"] = score

            if args.mode == "mcq":  # official: lingo_judge=0.0 when the judge is unused
                rec["lingo_judge"] = 0.0

        # except Exception as e:
        #     print(f"Sample {sample_idx} failed: {e}")
        #     rec["error"] = str(e)
        #     rec["lingo_judge"] = 0.0


        except Exception as e:
            print(f"Sample {sample_idx} failed: {e}")
            rec["error"] = str(e)
            rec["lingo_judge"] = 0.0
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


                

        results["samples"].append(rec)

    if args.mode in ("mcq", "both"):
        final_accuracy, not_processed = calculate_accuracy(gt_indexes, answers, model_name)
        print(f"Final accuracy for {model_name}: {100 * final_accuracy:.2f} %. "
              f"Not processed: {not_processed} out of {len(gt_indexes)}")
        results.update(final_accuracy=final_accuracy, not_processed=not_processed,
                       num_samples=len(gt_indexes))

    if args.mode in ("lingo", "both"):  # aggregate is our addition (official lingo mode has none)
        judge_scores = [s["lingo_judge"] for s in results["samples"]
                        if "error" not in s and s.get("lingo_judge") is not None]
        results["lingo_metrics"] = {
            "num_judged": len(judge_scores),
            "agreement_rate": (sum(1 for s in judge_scores if s > 0.5) / len(judge_scores))
                              if judge_scores else 0.0,
            "mean_score": (sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0,
        }
        print(f"Lingo-Judge: {len(judge_scores)} judged, "
              f"agreement {results['lingo_metrics']['agreement_rate'] * 100:.2f} %")

    # official per-model cleanup
    del vlm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"vlm_results_{timestamp}_{args.mode}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

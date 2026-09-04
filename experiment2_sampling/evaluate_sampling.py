# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from evaluate_qwen25 import (
    Qwen25VLModel,
    resolve_model_path,
    NCAP_SYSTEM_PROMPT,
)

from sampling_strategies import (
    decode_candidate_frames,
    apply_sampling_strategy,
)


# ============================================================
# Default paths
# ============================================================

# DEFAULT_MODEL = (
#     "/root/autodl-tmp/outputs/"
#     "qwen2.5vl-automingo-lora-v2/checkpoint-814"
# )
DEFAULT_MODEL = (
    "/root/autodl-tmp/models/experiment2/"
    "qwen2.5vl-automingo-merged"
)

DEFAULT_BASE_MODEL = (
    "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
)

DEFAULT_DATASET = (
    "/root/autodl-tmp/datasets/VRU-Accident/"
    "experiment2/sampling_eval_200.jsonl"
)

DEFAULT_DATASET_ROOT = (
    "/root/autodl-tmp/datasets/VRU-Accident"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# VRU-Accident prompt
# ============================================================

def clean_option(option: str) -> str:
    """
    VRU-Accident annotation already stores options as:

        A. sunny day
        B. cloudy day
        C. rainy evening
        D. stormy night

    Remove the existing option label so that we can reconstruct
    a clean A/B/C/D prompt exactly once.
    """
    option = (option or "").strip()

    option = re.sub(
        r"^[A-Da-d]\s*[\.\:\)]\s*",
        "",
        option,
    )

    return option.strip()


def build_vru_mcq_prompt(
    question: str,
    options: List[str],
) -> str:
    """
    Build a deterministic A/B/C/D multiple-choice prompt.

    Important:
    VRU-Accident preserves the original option ordering, so
    we must NOT shuffle the options.
    """
    cleaned_options = [
        clean_option(option)
        for option in options
    ]

    option_text = "\n".join(
        f"{chr(ord('A') + i)}. {option}"
        for i, option in enumerate(cleaned_options)
    )

    return (
        f"{question}\n\n"
        f"Options:\n"
        f"{option_text}\n\n"
        f"Answer (just the option letter):"
    )


# ============================================================
# Answer normalization
# ============================================================

def normalize_answer(text: str) -> str:
    """
    Normalize common Qwen outputs to A/B/C/D.

    Supported examples:
        A
        B.
        Answer: C
        The correct answer is D
        1 / 2 / 3 / 4
    """
    text = (text or "").strip()

    # Remove obvious whitespace/newline noise.
    text = text.strip()

    # Explicit "answer: X"
    match = re.search(
        r"\banswer\s*[:\-]?\s*([ABCD])\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    # Standalone A/B/C/D
    match = re.search(
        r"\b([ABCD])\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    # Fallback: 1/2/3/4 -> A/B/C/D
    match = re.search(
        r"\b([1-4])\b",
        text,
    )

    if match:
        return chr(
            ord("A") + int(match.group(1)) - 1
        )

    return text.upper()


# ============================================================
# Visual token measurement
# ============================================================

def count_visual_tokens(
    processor: Any,
    inputs: Dict[str, Any],
) -> int:
    """
    Measure actual Qwen2.5-VL visual tokens from image_grid_thw.

    image_grid_thw contains the temporal/spatial visual grid.
    The final visual token count is:

        prod(T, H, W) / merge_size^2

    summed over all input images.
    """
    grid = inputs.get("image_grid_thw")

    if grid is None:
        raise RuntimeError(
            "image_grid_thw is missing from processor output; "
            "cannot measure visual tokens reliably."
        )

    if not torch.is_tensor(grid):
        grid = torch.as_tensor(grid)

    grid = grid.detach().cpu()

    if grid.ndim == 1:
        grid = grid.unsqueeze(0)

    merge_size = getattr(
        processor.image_processor,
        "merge_size",
        2,
    )

    per_image_tokens = (
        grid.to(torch.long).prod(dim=1)
        // (int(merge_size) ** 2)
    )

    return int(per_image_tokens.sum().item())


# ============================================================
# Qwen2.5-VL inference
# ============================================================

def generate_with_metrics(
    vlm: Qwen25VLModel,
    prompt: str,
    images: List[Image.Image],
) -> Tuple[str, int, float]:
    """
    Reuse Experiment 1's Qwen2.5-VL inference pipeline.

    Returns:
        raw_answer
        visual_tokens
        inference_latency_ms
    """
    pil_images = [
        image.convert("RGB")
        for image in images
    ]

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": NCAP_SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": (
                [
                    {
                        "type": "image",
                        "image": image,
                    }
                    for image in pil_images
                ]
                + [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            ),
        },
    ]

    text = vlm.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = vlm.processor(
        text=[text],
        images=pil_images,
        return_tensors="pt",
    )

    # Measure actual visual token count before moving tensors.
    visual_tokens = count_visual_tokens(
        vlm.processor,
        inputs,
    )

    inputs = {
        key: (
            value.to(vlm.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    # Synchronize CUDA so latency measures actual generation time.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = vlm.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    out = vlm.processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    # Same post-processing logic as Experiment 1.
    if prompt in out:
        out = out.split(
            prompt,
            1,
        )[-1].strip()

    out = (
        out.lower()
        .split("assistant")[-1]
        .strip()
    )

    return (
        out.strip(),
        visual_tokens,
        latency_ms,
    )


# ============================================================
# Dataset loading
# ============================================================

def load_records(
    path: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def group_by_video(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[
        str,
        List[Dict[str, Any]]
    ] = defaultdict(list)

    for record in records:
        grouped[record["video_path"]].append(
            record
        )

    return dict(grouped)


# ============================================================
# Video path resolution
# ============================================================

def resolve_video_path(
    video_path: str,
    dataset_root: str,
) -> Path:
    """
    Resolve paths such as:

        ./VRU_videos/CAP_DATA/VRU_150.mp4

    against:

        /root/autodl-tmp/datasets/VRU-Accident
    """
    path = Path(video_path)

    if path.is_absolute():
        resolved = path
    else:
        resolved = (
            Path(dataset_root)
            / path
        )

    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Video file not found: {resolved}"
        )

    return resolved


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "LoRA adapter directory or merged model "
            "directory."
        ),
    )

    parser.add_argument(
        "--base_model",
        default=DEFAULT_BASE_MODEL,
        help=(
            "Local Qwen2.5-VL base model used when "
            "resolving the LoRA adapter."
        ),
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Experiment 2 JSONL evaluation file.",
    )

    parser.add_argument(
        "--dataset_root",
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root directory of VRU-Accident. "
            "Relative video_path values are resolved here."
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "uniform",
            "dense",
            "event-aware",
        ],
        required=True,
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        choices=[4, 8, 16],
        required=True,
    )

    parser.add_argument(
        "--candidate_frames",
        type=int,
        default=64,
        help=(
            "Number of uniformly decoded candidate "
            "frames before applying the sampling strategy."
        ),
    )

    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="0 = evaluate all videos.",
    )

    parser.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="0 = evaluate all questions.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # --------------------------------------------------------
    # Load evaluation records
    # --------------------------------------------------------

    records = load_records(
        args.dataset
    )

    if args.max_questions > 0:
        records = records[
            :args.max_questions
        ]

    grouped = group_by_video(records)

    video_paths = list(
        grouped.keys()
    )

    if args.max_videos > 0:
        video_paths = video_paths[
            :args.max_videos
        ]

    total_requested_qa = sum(
        len(grouped[path])
        for path in video_paths
    )

    # --------------------------------------------------------
    # Resolve model
    # --------------------------------------------------------

    model_path = resolve_model_path(
        args.model,
        args.base_model,
    )

    print(
        f"Model: {model_path}"
    )
    print(
        f"Strategy: {args.strategy}"
    )
    print(
        f"Frame budget: {args.num_frames}"
    )
    print(
        f"Candidate frames: "
        f"{args.candidate_frames}"
    )
    print(
        f"Videos: {len(video_paths)}"
    )
    print(
        f"QA: {total_requested_qa}"
    )
    print(
        f"Dataset root: "
        f"{args.dataset_root}"
    )

    # --------------------------------------------------------
    # Load Qwen2.5-VL + Automingo LoRA
    # --------------------------------------------------------

    vlm = Qwen25VLModel(
        model_id=model_path,
        device="cuda",
        trust_remote_code=False,
    )

    # --------------------------------------------------------
    # Result containers
    # --------------------------------------------------------

    all_results: List[
        Dict[str, Any]
    ] = []

    correct = 0
    evaluated = 0
    failed = 0

    visual_token_values: List[int] = []
    latency_values: List[float] = []
    sampling_latency_values: List[float] = []

    source_stats: Dict[
        str,
        Dict[str, int]
    ] = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    category_stats: Dict[
        str,
        Dict[str, int]
    ] = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    # --------------------------------------------------------
    # Video-level loop
    # --------------------------------------------------------

    for raw_video_path in tqdm(
        video_paths,
        desc=(
            f"{args.strategy}-"
            f"{args.num_frames}"
        ),
    ):

        # Resolve relative path correctly.
        try:
            video_path = resolve_video_path(
                raw_video_path,
                args.dataset_root,
            )
        except Exception as exc:
            print(
                f"[VIDEO ERROR] "
                f"{raw_video_path}: {exc}"
            )

            failed += len(
                grouped[raw_video_path]
            )

            continue

        # ----------------------------------------------------
        # Decode candidate frames + sampling
        # ----------------------------------------------------

        try:
            start_sampling = time.perf_counter()

            (
                candidate_frames,
                candidate_indices,
                video_meta,
            ) = decode_candidate_frames(
                str(video_path),
                num_candidates=args.candidate_frames,
            )

            (
                selected_frames,
                sampling_meta,
            ) = apply_sampling_strategy(
                candidate_frames,
                args.strategy,
                args.num_frames,
            )

            sampling_ms = (
                time.perf_counter()
                - start_sampling
            ) * 1000.0

        except Exception as exc:
            print(
                f"[SAMPLING ERROR] "
                f"{video_path}: {exc}"
            )

            failed += len(
                grouped[raw_video_path]
            )

            continue

        # Sampling strategy returns indices into candidate_frames.
        selected_candidate_indices = (
            sampling_meta["candidate_indices"]
        )

        # Convert candidate-frame indices into
        # original video frame indices.
        selected_indices = [
            candidate_indices[i]
            for i in selected_candidate_indices
        ]

        # ----------------------------------------------------
        # QA-level evaluation
        # ----------------------------------------------------

        for record in grouped[
            raw_video_path
        ]:

            prompt = build_vru_mcq_prompt(
                record["question"],
                record["options"],
            )

            result: Dict[str, Any] = {
                "source": record["source"],
                "video_path": record["video_path"],
                "category": record["category"],
                "question": record["question"],
                "options": record["options"],
                "ground_truth": record["ground_truth"],

                "strategy": args.strategy,
                "num_frames": args.num_frames,
                "candidate_frames": args.candidate_frames,

                "selected_frame_indices": (
                    selected_indices
                ),

                "total_video_frames": (
                    video_meta["total_frames"]
                ),
                "fps": video_meta["fps"],
                "duration_sec": (
                    video_meta["duration_sec"]
                ),

                "sampling_latency_ms": (
                    sampling_ms
                ),
            }

            try:
                (
                    answer,
                    visual_tokens,
                    latency_ms,
                ) = generate_with_metrics(
                    vlm,
                    prompt,
                    selected_frames,
                )

                parsed_answer = normalize_answer(
                    answer
                )

                ground_truth = (
                    record["ground_truth"]
                    .strip()
                    .upper()
                )

                is_correct = (
                    parsed_answer
                    == ground_truth
                )

                result.update(
                    {
                        "answer": answer,
                        "parsed_answer": parsed_answer,
                        "correct": is_correct,
                        "visual_tokens": (
                            visual_tokens
                        ),
                        "inference_latency_ms": (
                            latency_ms
                        ),
                    }
                )

                evaluated += 1

                if is_correct:
                    correct += 1

                # Per-source statistics.
                source = record["source"]

                source_stats[source][
                    "total"
                ] += 1

                if is_correct:
                    source_stats[source][
                        "correct"
                    ] += 1

                # Per-category statistics.
                category = record["category"]

                category_stats[category][
                    "total"
                ] += 1

                if is_correct:
                    category_stats[category][
                        "correct"
                    ] += 1

                # Core metrics.
                visual_token_values.append(
                    visual_tokens
                )

                latency_values.append(
                    latency_ms
                )

                sampling_latency_values.append(
                    sampling_ms
                )

            except Exception as exc:
                print(
                    f"[QA ERROR] "
                    f"{raw_video_path} / "
                    f"{record['category']}: {exc}"
                )

                failed += 1

                result.update(
                    {
                        "error": str(exc),
                        "correct": False,
                    }
                )

            all_results.append(result)

    # ========================================================
    # Aggregate metrics
    # ========================================================

    accuracy = (
        correct / evaluated
        if evaluated > 0
        else 0.0
    )

    mean_visual_tokens = (
        sum(visual_token_values)
        / len(visual_token_values)
        if visual_token_values
        else 0.0
    )

    mean_latency = (
        sum(latency_values)
        / len(latency_values)
        if latency_values
        else 0.0
    )

    if latency_values:
        sorted_latencies = sorted(
            latency_values
        )

        p50_index = min(
            len(sorted_latencies) - 1,
            int(0.50 * len(sorted_latencies)),
        )

        p95_index = min(
            len(sorted_latencies) - 1,
            int(0.95 * len(sorted_latencies)),
        )

        p50_latency = (
            sorted_latencies[p50_index]
        )

        p95_latency = (
            sorted_latencies[p95_index]
        )
    else:
        p50_latency = 0.0
        p95_latency = 0.0

    mean_sampling_latency = (
        sum(sampling_latency_values)
        / len(sampling_latency_values)
        if sampling_latency_values
        else 0.0
    )

    # ========================================================
    # Per-source metrics
    # ========================================================

    source_metrics: Dict[
        str,
        Dict[str, Any]
    ] = {}

    for source, stat in source_stats.items():
        total = stat["total"]

        source_metrics[source] = {
            "accuracy": (
                stat["correct"] / total
                if total
                else 0.0
            ),
            "correct": stat["correct"],
            "total": total,
        }

    # ========================================================
    # Per-category metrics
    # ========================================================

    category_metrics: Dict[
        str,
        Dict[str, Any]
    ] = {}

    for category, stat in category_stats.items():
        total = stat["total"]

        category_metrics[category] = {
            "accuracy": (
                stat["correct"] / total
                if total
                else 0.0
            ),
            "correct": stat["correct"],
            "total": total,
        }

    # ========================================================
    # Final result object
    # ========================================================

    results = {
        "experiment": (
            "experiment2_frame_sampling"
        ),

        "model": str(model_path),

        "dataset": {
            "name": "VRU-Accident",
            "evaluation_file": args.dataset,
            "dataset_root": args.dataset_root,
            "num_videos": len(video_paths),
            "num_questions": evaluated,
        },

        "sampling": {
            "strategy": args.strategy,
            "num_frames": args.num_frames,
            "candidate_frames": (
                args.candidate_frames
            ),
        },

        # ====================================================
        # Three core metrics
        # ====================================================

        "metrics": {
            "vqa_accuracy": accuracy,

            "visual_tokens": {
                "mean": mean_visual_tokens,
            },

            "inference_latency_ms": {
                "mean": mean_latency,
                "p50": p50_latency,
                "p95": p95_latency,
            },

            # Secondary diagnostic.
            "sampling_latency_ms": {
                "mean": mean_sampling_latency,
            },
        },

        "statistics": {
            "correct": correct,
            "evaluated": evaluated,
            "failed": failed,
        },

        "by_source": source_metrics,

        "by_category": category_metrics,

        "samples": all_results,
    }

    # ========================================================
    # Save result
    # ========================================================

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Console summary
    # ========================================================

    print(
        "\n========== RESULT =========="
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Visual Tokens: "
        f"{mean_visual_tokens:.1f}"
    )

    print(
        f"Inference Latency: "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"Latency P50: "
        f"{p50_latency:.2f} ms"
    )

    print(
        f"Latency P95: "
        f"{p95_latency:.2f} ms"
    )

    print(
        f"Sampling Latency: "
        f"{mean_sampling_latency:.2f} ms"
    )

    print(
        f"Evaluated: "
        f"{evaluated}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"\nResults saved to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
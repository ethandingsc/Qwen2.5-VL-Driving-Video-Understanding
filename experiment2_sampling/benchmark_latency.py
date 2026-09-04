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
import statistics
import time
from typing import Any, Dict, List

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

STRATEGIES = [
    "uniform",
    "dense",
    "event-aware",
]

FRAME_BUDGETS = [
    4,
    8,
    16,
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def load_records(path: str) -> List[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def get_unique_videos(
    records: List[Dict[str, Any]],
) -> List[str]:

    seen = set()
    videos = []

    for record in records:
        video_path = record["video_path"]

        if video_path not in seen:
            seen.add(video_path)
            videos.append(video_path)

    return videos


def resolve_video_path(
    video_path: str,
    dataset_root: str,
) -> Path:

    path = Path(video_path)

    if not path.is_absolute():
        path = Path(dataset_root) / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {path}"
        )

    return path


# ============================================================
# Fixed latency prompt
# ============================================================

LATENCY_PROMPT = (
    "Choose the best answer from A, B, C, or D. "
    "Reply with only one option letter. "
    "Do not provide any explanation.\n\n"
    "Question: What best describes the driving scene?\n\n"
    "Options:\n"
    "A. Normal driving scene\n"
    "B. Potential traffic interaction\n"
    "C. Safety-critical traffic event\n"
    "D. Cannot determine\n\n"
    "Answer:"
)


# ============================================================
# Prepare multimodal inputs
# ============================================================

def prepare_inputs(
    vlm: Qwen25VLModel,
    images: List[Image.Image],
) -> Dict[str, Any]:

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
                        "text": LATENCY_PROMPT,
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

    inputs = {
        key: (
            value.to(vlm.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    return inputs


# ============================================================
# Controlled model latency
# ============================================================

def benchmark_generate(
    vlm: Qwen25VLModel,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:

    input_length = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = vlm.model.generate(
            **inputs,

            # Critical:
            # keep decoding short and deterministic.
            max_new_tokens=2,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    generated_ids = output_ids[
        :,
        input_length:
    ]

    generated_tokens = int(
        generated_ids.shape[1]
    )

    answer = vlm.processor.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    return {
        "latency_ms": latency_ms,
        "generated_tokens": generated_tokens,
        "answer": answer,
    }


# ============================================================
# Statistics
# ============================================================

def percentile(
    values: List[float],
    q: float,
) -> float:

    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = round(
        (len(sorted_values) - 1) * q
    )

    return float(
        sorted_values[index]
    )


def summarize(
    values: List[float],
) -> Dict[str, float]:

    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "std": 0.0,
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "std": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


# ============================================================
# Warm-up
# ============================================================

def run_warmup(
    vlm: Qwen25VLModel,
    frames: List[Image.Image],
    warmup_runs: int,
) -> None:

    print(
        f"\nRunning {warmup_runs} warm-up inference(s)..."
    )

    inputs = prepare_inputs(
        vlm,
        frames,
    )

    for _ in range(warmup_runs):
        benchmark_generate(
            vlm,
            inputs,
        )

    print("Warm-up finished.\n")


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
    )

    parser.add_argument(
        "--base_model",
        default=DEFAULT_BASE_MODEL,
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--dataset_root",
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--candidate_frames",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_videos",
        type=int,
        default=50,
        help="Number of unique videos used for latency benchmark.",
    )

    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default=(
            "experiment2_sampling/results/"
            "latency_benchmark.json"
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # ========================================================
    # Load dataset
    # ========================================================

    records = load_records(
        args.dataset
    )

    video_paths = get_unique_videos(
        records
    )

    # Fixed deterministic subset.
    video_paths = video_paths[
        :args.num_videos
    ]

    if not video_paths:
        raise RuntimeError(
            "No videos found for benchmark."
        )

    print(
        "\n========== LATENCY BENCHMARK =========="
    )

    print(
        f"Videos: {len(video_paths)}"
    )

    print(
        f"Candidate frames: "
        f"{args.candidate_frames}"
    )

    print(
        "Configurations: 9"
    )

    print(
        "Generation: deterministic / "
        "max_new_tokens=2"
    )

    # ========================================================
    # Load model ONCE
    # ========================================================

    model_path = resolve_model_path(
        args.model,
        args.base_model,
    )

    print(
        f"\nLoading model:\n{model_path}"
    )

    vlm = Qwen25VLModel(
        model_id=model_path,
        device="cuda",
        trust_remote_code=False,
    )

    print("\nModel loaded.")

    # ========================================================
    # Decode first video for warm-up
    # ========================================================

    first_video = resolve_video_path(
        video_paths[0],
        args.dataset_root,
    )

    (
        warmup_candidates,
        _,
        _,
    ) = decode_candidate_frames(
        str(first_video),
        num_candidates=args.candidate_frames,
    )

    (
        warmup_frames,
        _,
    ) = apply_sampling_strategy(
        warmup_candidates,
        "uniform",
        8,
    )

    run_warmup(
        vlm,
        warmup_frames,
        args.warmup_runs,
    )

    # ========================================================
    # Result containers
    # ========================================================

    configuration_results = {
        strategy: {
            str(num_frames): []
            for num_frames in FRAME_BUDGETS
        }
        for strategy in STRATEGIES
    }

    failures = []

    # ========================================================
    # Benchmark
    # ========================================================

    for raw_video_path in tqdm(
        video_paths,
        desc="Latency benchmark",
    ):

        try:
            video_path = resolve_video_path(
                raw_video_path,
                args.dataset_root,
            )

            (
                candidate_frames,
                _,
                _,
            ) = decode_candidate_frames(
                str(video_path),
                num_candidates=args.candidate_frames,
            )

        except Exception as exc:
            print(
                f"\n[VIDEO ERROR] "
                f"{raw_video_path}: {exc}"
            )

            failures.append(
                {
                    "video_path": raw_video_path,
                    "error": str(exc),
                }
            )

            continue

        # ----------------------------------------------------
        # Randomize configuration order per video.
        # Prevent systematic runtime-order bias.
        # ----------------------------------------------------

        configurations = [
            (strategy, num_frames)
            for strategy in STRATEGIES
            for num_frames in FRAME_BUDGETS
        ]

        random.shuffle(
            configurations
        )

        for strategy, num_frames in configurations:

            try:
                (
                    selected_frames,
                    _,
                ) = apply_sampling_strategy(
                    candidate_frames,
                    strategy,
                    num_frames,
                )

                # IMPORTANT:
                # Processor runs BEFORE timing.
                #
                # Therefore primary latency measures:
                #
                # model.generate()
                #
                # only.
                inputs = prepare_inputs(
                    vlm,
                    selected_frames,
                )

                result = benchmark_generate(
                    vlm,
                    inputs,
                )

                configuration_results[
                    strategy
                ][
                    str(num_frames)
                ].append(
                    {
                        "video_path": raw_video_path,
                        "latency_ms": (
                            result["latency_ms"]
                        ),
                        "generated_tokens": (
                            result[
                                "generated_tokens"
                            ]
                        ),
                        "answer": (
                            result["answer"]
                        ),
                    }
                )

            except Exception as exc:
                print(
                    f"\n[BENCHMARK ERROR] "
                    f"{raw_video_path} / "
                    f"{strategy} / "
                    f"{num_frames}F: {exc}"
                )

                failures.append(
                    {
                        "video_path": raw_video_path,
                        "strategy": strategy,
                        "num_frames": num_frames,
                        "error": str(exc),
                    }
                )

    # ========================================================
    # Aggregate
    # ========================================================

    summary = {}

    print(
        "\n\n========== RESULTS =========="
    )

    for strategy in STRATEGIES:

        summary[strategy] = {}

        print(
            f"\n--- {strategy.upper()} ---"
        )

        for num_frames in FRAME_BUDGETS:

            samples = (
                configuration_results[
                    strategy
                ][
                    str(num_frames)
                ]
            )

            latencies = [
                sample["latency_ms"]
                for sample in samples
            ]

            generated_tokens = [
                sample["generated_tokens"]
                for sample in samples
            ]

            stats = summarize(
                latencies
            )

            token_counts = sorted(
                set(generated_tokens)
            )

            summary[
                strategy
            ][
                str(num_frames)
            ] = {
                "num_samples": len(samples),

                "latency_ms": stats,

                "generated_token_counts": (
                    token_counts
                ),
            }

            print(
                f"{num_frames:2d} Frames | "
                f"N={len(samples):3d} | "
                f"Median={stats['median']:.2f} ms | "
                f"Mean={stats['mean']:.2f} ms | "
                f"P95={stats['p95']:.2f} ms | "
                f"Std={stats['std']:.2f} ms | "
                f"Generated Tokens={token_counts}"
            )

    # ========================================================
    # Save
    # ========================================================

    final_result = {
        "experiment": (
            "experiment2_latency_benchmark"
        ),

        "model": str(model_path),

        "protocol": {
            "num_videos": len(video_paths),
            "candidate_frames": (
                args.candidate_frames
            ),
            "strategies": STRATEGIES,
            "frame_budgets": FRAME_BUDGETS,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,

            "generation": {
                "max_new_tokens": 2,
                "do_sample": False,
            },

            "timing_scope": (
                "model.generate only"
            ),

            "processor_included": False,
            "sampling_included": False,
        },

        "summary": summary,

        "samples": configuration_results,

        "failures": failures,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )

    print(
        f"\nFailures: {len(failures)}"
    )


if __name__ == "__main__":
    main()
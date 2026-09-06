# -*- coding: utf-8 -*-

import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import requests


# ============================================================
# Configuration
# ============================================================

SGLANG_API_URL = os.getenv(
    "SGLANG_API_URL",
    "http://127.0.0.1:30000/v1/chat/completions",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "/root/autodl-tmp/models/experiment2/qwen2.5vl-automingo-merged",
)

CANDIDATE_FRAMES = 64


# ============================================================
# Video decoding
# ============================================================

def decode_candidate_frames(
    video_path: str,
    num_candidates: int = 64,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video.")

    num_candidates = min(
        num_candidates,
        total_frames,
    )

    indices = [
        round(
            i * (total_frames - 1)
            / max(num_candidates - 1, 1)
        )
        for i in range(num_candidates)
    ]

    frames = []

    for idx in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx,
        )

        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            (idx, frame)
        )

    cap.release()

    return frames


# ============================================================
# Sampling
# ============================================================

def uniform_sampling(
    candidates,
    k,
):
    n = len(candidates)

    indices = [
        round(
            i * (n - 1)
            / max(k - 1, 1)
        )
        for i in range(k)
    ]

    return [
        candidates[i]
        for i in indices
    ]


def frame_difference(
    frame1,
    frame2,
):
    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_RGB2GRAY,
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_RGB2GRAY,
    )

    gray1 = cv2.resize(
        gray1,
        (64, 36),
    )

    gray2 = cv2.resize(
        gray2,
        (64, 36),
    )

    diff = cv2.absdiff(
        gray1,
        gray2,
    )

    return float(diff.mean())


def event_aware_sampling(
    candidates,
    k,
):
    if len(candidates) <= k:
        return candidates

    scores = [0.0]

    for i in range(
        1,
        len(candidates),
    ):
        score = frame_difference(
            candidates[i - 1][1],
            candidates[i][1],
        )

        scores.append(score)

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    # Lightweight temporal spacing.
    min_gap = max(
        1,
        len(candidates) // (k * 2),
    )

    selected = []

    for idx in ranked:
        if all(
            abs(idx - old) >= min_gap
            for old in selected
        ):
            selected.append(idx)

        if len(selected) == k:
            break

    # Fallback.
    if len(selected) < k:
        for idx in ranked:
            if idx not in selected:
                selected.append(idx)

            if len(selected) == k:
                break

    selected.sort()

    return [
        candidates[i]
        for i in selected
    ]


def dense_sampling(
    candidates,
    k,
):
    if len(candidates) <= k:
        return candidates

    scores = []

    for i in range(
        1,
        len(candidates),
    ):
        score = frame_difference(
            candidates[i - 1][1],
            candidates[i][1],
        )

        scores.append(score)

    center = (
        max(
            range(len(scores)),
            key=lambda i: scores[i],
        )
        + 1
    )

    window_size = max(
        k * 2,
        len(candidates) // 4,
    )

    start = max(
        0,
        center - window_size // 2,
    )

    end = min(
        len(candidates),
        start + window_size,
    )

    start = max(
        0,
        end - window_size,
    )

    window = candidates[
        start:end
    ]

    return uniform_sampling(
        window,
        k,
    )


def sample_frames(
    candidates,
    strategy,
    num_frames,
):
    if strategy == "Uniform":
        return uniform_sampling(
            candidates,
            num_frames,
        )

    if strategy == "Dense":
        return dense_sampling(
            candidates,
            num_frames,
        )

    if strategy == "Event-aware":
        return event_aware_sampling(
            candidates,
            num_frames,
        )

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# Save selected frames temporarily
# ============================================================

def save_frames(selected):
    temp_dir = tempfile.mkdtemp(
        prefix="qwen_vru_"
    )

    paths = []

    for i, (frame_idx, frame) in enumerate(
        selected
    ):
        path = Path(temp_dir) / (
            f"frame_{i:02d}_"
            f"{frame_idx}.jpg"
        )

        bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR,
        )

        cv2.imwrite(
            str(path),
            bgr,
        )

        paths.append(
            str(path)
        )

    return paths


# ============================================================
# SGLang request
# ============================================================

def call_sglang(
    frame_paths,
    question,
):
    content = []

    # NOTE:
    # This assumes the SGLang OpenAI-compatible endpoint
    # accepts local file URLs accessible by the server.
    for path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "file://"
                        + str(
                            Path(path).resolve()
                        )
                    )
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": question,
        }
    )

    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],

        "temperature": 0.2,
        "max_tokens": 256,
    }

    start = time.perf_counter()

    response = requests.post(
        SGLANG_API_URL,
        json=payload,
        timeout=300,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    response.raise_for_status()

    data = response.json()

    answer = (
        data["choices"][0]
        ["message"]["content"]
    )

    return (
        answer,
        latency_ms,
    )


# ============================================================
# Gradio inference
# ============================================================

def analyze_video(
    video_path,
    strategy,
    num_frames,
    question,
):
    if not video_path:
        raise gr.Error(
            "Please upload a driving video."
        )

    if not question.strip():
        raise gr.Error(
            "Please enter a question."
        )

    # Decode shared candidate pool.
    candidates = decode_candidate_frames(
        video_path,
        CANDIDATE_FRAMES,
    )

    # Sampling.
    selected = sample_frames(
        candidates,
        strategy,
        int(num_frames),
    )

    selected_indices = [
        item[0]
        for item in selected
    ]

    # Gallery images.
    gallery = [
        (
            frame,
            f"Frame {frame_idx}",
        )
        for frame_idx, frame in selected
    ]

    # Temporary image files for API.
    frame_paths = save_frames(
        selected
    )

    answer, latency_ms = call_sglang(
        frame_paths,
        question,
    )

    info = (
        f"Strategy: {strategy}\n"
        f"Frame Budget: {num_frames}\n"
        f"Selected Frames: {selected_indices}\n"
        f"API Latency: {latency_ms:.2f} ms"
    )

    return (
        gallery,
        answer,
        info,
    )


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Driving Video VLM",
) as demo:

    gr.Markdown(
        """
# Driving Video Multimodal Understanding

**Qwen2.5-VL-7B + Driving-domain LoRA**

Upload a driving video, select a frame sampling strategy,
and ask a question about the scene.
"""
    )

    with gr.Row():

        with gr.Column():

            # video = gr.Video(
            #     label="Driving Video",
            # )

            video = gr.File(
                label="Driving Video",
                file_types=[".mp4", ".avi", ".mov", ".mkv"],
                type="filepath",
            )

            strategy = gr.Dropdown(
                choices=[
                    "Uniform",
                    "Dense",
                    "Event-aware",
                ],
                value="Event-aware",
                label="Sampling Strategy",
            )

            num_frames = gr.Radio(
                choices=[
                    4,
                    8,
                    16,
                ],
                value=8,
                label="Frame Budget",
            )

            question = gr.Textbox(
                label="Question",
                placeholder=(
                    "What happened in this "
                    "driving scene?"
                ),
                lines=3,
            )

            run_button = gr.Button(
                "Analyze Video",
                variant="primary",
            )

        with gr.Column():

            answer = gr.Textbox(
                label="Model Answer",
                lines=8,
            )

            metrics = gr.Textbox(
                label="Inference Information",
                lines=5,
            )

    gallery = gr.Gallery(
        label="Selected Frames",
        columns=4,
        height="auto",
    )

    run_button.click(
        fn=analyze_video,
        inputs=[
            video,
            strategy,
            num_frames,
            question,
        ],
        outputs=[
            gallery,
            answer,
            metrics,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
    )
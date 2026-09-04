# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================

DEFAULT_CANDIDATE_FRAMES = 64

# Small resolution used for visual-change calculation.
DIFF_WIDTH = 64
DIFF_HEIGHT = 36


# ============================================================
# Video decoding
# ============================================================

def decode_candidate_frames(
    video_path: str,
    num_candidates: int = DEFAULT_CANDIDATE_FRAMES,
) -> Tuple[List[Image.Image], List[int], Dict[str, Any]]:
    """
    Decode a fixed number of candidate frames uniformly
    from the complete video.

    This creates a common candidate pool that is shared by
    all sampling strategies.

    Args:
        video_path:
            Path to the input video.

        num_candidates:
            Number of candidate frames to decode.

    Returns:
        frames:
            Candidate frames as RGB PIL Images.

        frame_indices:
            Original frame indices corresponding to `frames`.

        metadata:
            Basic video information.
    """

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be greater than 0"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if total_frames <= 0:
        cap.release()

        raise ValueError(
            f"Invalid frame count: {video_path}"
        )

    if fps <= 0:
        fps = 30.0

    num_candidates = min(
        int(num_candidates),
        total_frames,
    )

    # Uniformly generate candidate positions over
    # the entire original video.
    candidate_indices = np.linspace(
        0,
        total_frames - 1,
        num=num_candidates,
        dtype=int,
    )

    # Remove duplicates caused by integer rounding.
    candidate_indices = np.unique(
        candidate_indices
    )

    frames: List[Image.Image] = []
    valid_indices: List[int] = []

    # Decode candidate frames.
    for frame_index in candidate_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = cap.read()

        if not success:
            continue

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            Image.fromarray(frame)
        )

        valid_indices.append(
            int(frame_index)
        )

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded from: {video_path}"
        )

    metadata = {
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": (
            total_frames / fps
        ),
        "num_candidates": len(frames),
    }

    return (
        frames,
        valid_indices,
        metadata,
    )


# ============================================================
# Uniform Sampling
# ============================================================

def uniform_sampling(
    num_frames_available: int,
    num_frames: int,
) -> List[int]:
    """
    Uniformly sample K frames over the complete
    candidate-frame sequence.

    Example:

        64 candidates, K=4
        -> [0, 21, 42, 63]

    This is the primary baseline.
    """

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be greater than 0"
        )

    if num_frames_available <= 0:
        raise ValueError(
            "No available candidate frames"
        )

    # If the requested budget is greater than the
    # candidate pool, simply use all candidates.
    if num_frames >= num_frames_available:
        return list(
            range(num_frames_available)
        )

    indices = np.linspace(
        0,
        num_frames_available - 1,
        num=num_frames,
        dtype=int,
    )

    return sorted(
        set(
            int(index)
            for index in indices
        )
    )


# ============================================================
# Dense Sampling
# ============================================================

def _motion_score(
    frame1: Image.Image,
    frame2: Image.Image,
) -> float:
    """
    Compute a simple visual motion score between
    two adjacent frames.

    The score is the mean absolute grayscale
    difference after resizing to a small resolution.
    """

    image1 = np.asarray(
        frame1
        .convert("L")
        .resize(
            (DIFF_WIDTH, DIFF_HEIGHT)
        ),
        dtype=np.float32,
    )

    image2 = np.asarray(
        frame2
        .convert("L")
        .resize(
            (DIFF_WIDTH, DIFF_HEIGHT)
        ),
        dtype=np.float32,
    )

    return float(
        np.mean(
            np.abs(
                image1 - image2
            )
        )
    )


def calculate_motion_scores(
    frames: List[Image.Image],
) -> List[float]:
    """
    Calculate temporal motion scores for all
    candidate frames.

    The motion score at index i represents the
    visual difference between frame i-1 and i.

    The first frame has score 0.
    """

    n = len(frames)

    if n == 0:
        raise ValueError(
            "No frames available"
        )

    scores = np.zeros(
        n,
        dtype=np.float32,
    )

    for i in range(1, n):
        scores[i] = _motion_score(
            frames[i - 1],
            frames[i],
        )

    return scores.tolist()


def dense_sampling(
    frames: List[Image.Image],
    num_frames: int,
    window_ratio: float = 0.25,
) -> Tuple[List[int], List[float]]:
    """
    Dense temporal sampling based on the most active
    continuous temporal region.

    Procedure:

        1. Calculate frame-difference scores.
        2. Build a fixed-size temporal window.
        3. Find the window with the largest total
           visual-change score.
        4. Uniformly sample K frames inside this window.

    The method therefore represents a simple
    "dense local temporal sampling" baseline.

    It does NOT directly select individual high-score
    frames, which keeps it conceptually different
    from Event-aware sampling.

    Args:
        frames:
            Candidate frames.

        num_frames:
            Final frame budget.

        window_ratio:
            Fraction of the candidate timeline used
            as the dense temporal window.

    Returns:
        selected_indices:
            Indices into `frames`.

        motion_scores:
            Motion score for every candidate frame.
    """

    n = len(frames)

    if n == 0:
        raise ValueError(
            "No frames available"
        )

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be greater than 0"
        )

    if not (
        0 < window_ratio <= 1
    ):
        raise ValueError(
            "window_ratio must be in (0, 1]"
        )

    if num_frames >= n:
        scores = calculate_motion_scores(
            frames
        )

        return (
            list(range(n)),
            scores,
        )

    motion_scores = calculate_motion_scores(
        frames
    )

    # Number of candidate frames covered by the
    # dense temporal window.
    window_size = max(
        num_frames,
        int(
            round(
                n * window_ratio
            )
        ),
    )

    window_size = min(
        window_size,
        n,
    )

    # Prefix sum allows fast calculation of
    # motion energy for every possible window.
    scores = np.asarray(
        motion_scores,
        dtype=np.float32,
    )

    prefix = np.concatenate(
        [
            np.zeros(
                1,
                dtype=np.float32,
            ),
            np.cumsum(scores),
        ]
    )

    best_start = 0
    best_score = -float("inf")

    for start in range(
        0,
        n - window_size + 1,
    ):
        end = (
            start
            + window_size
        )

        window_score = float(
            prefix[end]
            - prefix[start]
        )

        if window_score > best_score:
            best_score = window_score
            best_start = start

    best_end = (
        best_start
        + window_size
        - 1
    )

    # Uniformly sample within the selected
    # high-motion temporal window.
    selected = np.linspace(
        best_start,
        best_end,
        num=num_frames,
        dtype=int,
    )

    selected = sorted(
        set(
            int(index)
            for index in selected
        )
    )

    # Extremely unlikely fallback caused by integer
    # rounding / pathological small windows.
    if len(selected) < num_frames:

        for index in range(
            best_start,
            best_end + 1,
        ):
            if index not in selected:
                selected.append(index)

            if len(selected) >= num_frames:
                break

        selected = sorted(
            selected[:num_frames]
        )

    return (
        selected,
        motion_scores,
    )


# ============================================================
# Event-aware Sampling
# ============================================================

def event_aware_sampling(
    frames: List[Image.Image],
    num_frames: int,
) -> Tuple[List[int], List[float]]:
    """
    Event-aware sampling based on frame-difference scores.

    Procedure:

        1. Calculate visual change between adjacent
           candidate frames.
        2. Rank candidate frames by visual-change score.
        3. Greedily select high-score frames.
        4. Apply a temporal spacing constraint so that
           all selected frames do not collapse into one
           extremely small temporal region.
        5. Restore chronological order.

    Returns:
        selected_indices:
            Selected candidate-frame indices.

        change_scores:
            Visual-change score of every candidate frame.
    """

    n = len(frames)

    if n == 0:
        raise ValueError(
            "No frames available"
        )

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be greater than 0"
        )

    motion_scores = calculate_motion_scores(
        frames
    )

    if num_frames >= n:
        return (
            list(range(n)),
            motion_scores,
        )

    scores = np.asarray(
        motion_scores,
        dtype=np.float32,
    )

    # The minimum temporal gap prevents all selected
    # frames from concentrating around one abrupt change.
    min_gap = max(
        1,
        n // (
            2 * num_frames
        ),
    )

    ranked_indices = sorted(
        range(n),
        key=lambda index: float(
            scores[index]
        ),
        reverse=True,
    )

    selected: List[int] = []

    for index in ranked_indices:

        if all(
            abs(index - selected_index)
            >= min_gap
            for selected_index in selected
        ):
            selected.append(index)

        if len(selected) >= num_frames:
            break

    # Fallback in case the temporal-spacing
    # constraint is too restrictive.
    if len(selected) < num_frames:

        for index in ranked_indices:
            if index not in selected:
                selected.append(index)

            if len(selected) >= num_frames:
                break

    selected = sorted(
        selected[:num_frames]
    )

    return (
        selected,
        motion_scores,
    )


# ============================================================
# Unified interface
# ============================================================

def apply_sampling_strategy(
    frames: List[Image.Image],
    strategy: str,
    num_frames: int,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Apply one sampling strategy to a shared
    candidate-frame pool.

    Supported strategies:

        uniform
        dense
        event-aware

    Returns:
        selected_frames:
            PIL images sent to Qwen2.5-VL.

        metadata:
            Sampling information and diagnostic scores.
    """

    strategy = strategy.lower().strip()

    if strategy == "uniform":

        indices = uniform_sampling(
            len(frames),
            num_frames,
        )

        extra: Dict[str, Any] = {}

    elif strategy == "dense":

        indices, motion_scores = dense_sampling(
            frames,
            num_frames,
        )

        extra = {
            "motion_scores": motion_scores,
        }

    elif strategy in {
        "event-aware",
        "event_aware",
    }:

        indices, change_scores = (
            event_aware_sampling(
                frames,
                num_frames,
            )
        )

        extra = {
            "change_scores": change_scores,
        }

    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Expected one of: "
            f"uniform, dense, event-aware."
        )

    selected_frames = [
        frames[index]
        for index in indices
    ]

    metadata = {
        "strategy": strategy,
        "num_frames": len(
            selected_frames
        ),
        "candidate_indices": indices,
        **extra,
    }

    return (
        selected_frames,
        metadata,
    )
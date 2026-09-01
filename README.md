# Qwen2.5-VL for Driving Video Temporal Understanding

> Parameter-efficient fine-tuning of Qwen2.5-VL-7B for multi-frame driving scene understanding and safety-critical reasoning.

## Overview

This project explores **multi-frame visual question answering and temporal reasoning for safety-critical driving scenarios** using **Qwen2.5-VL-7B-Instruct**.

The model is adapted to the driving domain through **LoRA supervised fine-tuning (SFT)** on the Automingo dataset. Each sample contains five temporally ordered driving frames together with a question, answer, and reasoning annotation.

Rather than replacing real-time perception systems such as object detection or BEV-based models, this project focuses on **offline analysis of recorded driving data**, including scene understanding, event reasoning, and safety-related question answering.

The project currently covers:

- Automingo → Qwen2.5-VL multimodal SFT data conversion
- LoRA fine-tuning with the official Qwen training framework
- Controlled Base vs. LoRA evaluation
- MCQ accuracy and reasoning-quality evaluation
- Future exploration of frame-sampling efficiency and inference deployment

---

## Motivation

Traditional driving perception systems are highly effective at tasks such as object detection, lane detection, and spatial localization. However, offline driving-data analysis often requires higher-level understanding across multiple frames, including:

- Temporal event relationships
- Safety-critical scene understanding
- Risk interpretation
- Reasoning about dynamic changes

This project investigates whether domain-specific LoRA fine-tuning can improve Qwen2.5-VL's ability to understand and reason about these multi-frame driving scenarios.

---

## Pipeline

```text
Automingo Dataset
        │
        ▼
Parquet Parsing
        │
        ▼
5-Frame Temporal Samples
        │
        ▼
Qwen Multimodal SFT Format
        │
        ▼
Qwen2.5-VL-7B-Instruct
        │
        ▼
LoRA Fine-Tuning
        │
        ▼
Base vs. LoRA Evaluation
        │
        ├── MCQ Accuracy
        └── Lingo-Judge
```

The original Automingo images are stored as binary data inside Parquet files. A custom preprocessing pipeline extracts the five temporal frames, saves them as PNG files, and converts the question, answer, and reasoning annotations into the multimodal conversation format expected by Qwen2.5-VL.

---

## Dataset

Experiments are based on the **Automingo** safety-critical driving dataset.

```text
Training samples:     3,256
Validation samples:   1,055
Frames per sample:    5
```

Each training sample contains temporally ordered driving images together with a question and supervised **Answer + Reasoning** output, allowing the model to learn both answer prediction and driving-scene reasoning.

---

## Fine-Tuning

The project uses the **official Qwen VL fine-tuning implementation** with PEFT LoRA rather than introducing an additional third-party training framework.

Current configuration:

```text
Model              Qwen2.5-VL-7B-Instruct
Training           LoRA SFT
Epochs             2
Learning Rate      5e-5
Effective Batch    8

LoRA Rank          64
LoRA Alpha         128
LoRA Dropout       0.05

Target Modules     Attention + FFN
Vision Encoder     Frozen
Precision          BF16
FlashAttention2    Enabled
```

Training is designed to run on a **single RTX 4090 24GB GPU**. The final run required approximately **1 hour 18 minutes** for 814 optimizer steps.

---

## Results

Base and LoRA models are evaluated on the same **1,055-sample Automingo validation set** using identical five-frame inputs, system prompts, generation settings, and scoring procedures.

| Model | MCQ Accuracy | Lingo-Judge Agreement |
|---|---:|---:|
| Qwen2.5-VL-7B Base | 76.40% | 61.71% |
| **Qwen2.5-VL-7B + LoRA** | **81.62%** | **73.74%** |
| **Improvement** | **+5.22 pp** | **+12.03 pp** |

Domain-specific LoRA fine-tuning improves both multiple-choice accuracy and the agreement between generated **Answer + Reasoning** responses and reference annotations.

---

## Evaluation

Evaluation follows the **Automingo official protocol** as closely as possible.

For Qwen2.5-VL, a lightweight evaluation entry point is implemented to read the local Parquet validation shards directly while preserving:

- Five-frame temporal ordering
- Official NCAP system prompt
- Identical generation parameters
- Official MCQ construction
- Official accuracy calculation
- Lingo-Judge evaluation

This avoids unnecessary Hugging Face Arrow cache generation while keeping Base and LoRA evaluation directly comparable.

---

## Roadmap

The next stage focuses on the trade-off between **temporal information, visual token cost, and inference latency**.

### Frame Sampling

Planned experiments include:

- Uniform sampling
- Dense sampling around dynamic events
- Event-aware sampling based on inter-frame changes

The goal is not simply to maximize accuracy, but to investigate whether similar performance can be maintained with fewer visual tokens and lower latency, or whether better accuracy can be achieved under the same token budget.

### Deployment

Planned deployment pipeline:

```text
LoRA
  │
  ▼
Merge
  │
  ▼
SGLang
  │
  ▼
OpenAI-Compatible API
  │
  ▼
Gradio Demo
```

The final demo is planned to support:

- Driving video upload
- Sampling-strategy selection
- Sampled-frame visualization
- Driving-scene question answering
- Answer + Reasoning generation
- Inference latency and frame statistics

---

## Tech Stack

`Python` · `PyTorch` · `Transformers` · `Qwen2.5-VL` · `PEFT/LoRA` · `FlashAttention2` · `Hugging Face` · `PyArrow` · `SGLang`

---

## Project Status

**In Progress**

- [x] Automingo data preprocessing
- [x] Qwen multimodal SFT conversion
- [x] LoRA fine-tuning
- [x] Base model evaluation
- [x] LoRA model evaluation
- [x] MCQ / Lingo-Judge comparison
- [ ] Frame sampling ablation
- [ ] Visual token / latency analysis
- [ ] SGLang deployment
- [ ] Interactive demo
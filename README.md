# Qwen2.5-VL for Driving Video Temporal Understanding


> Parameter-efficient fine-tuning and frame-sampling analysis of Qwen2.5-VL-7B for multi-frame driving scene understanding and safety-critical reasoning.

---

## 🎯 Overview

This project adapts **Qwen2.5-VL-7B-Instruct** to safety-critical driving scenarios through **LoRA fine-tuning**, and investigates how different **video frame sampling strategies** affect multimodal reasoning performance.

The system supports multi-frame visual question answering over offline driving footage, with a focus on:

- Dynamic scene understanding
- Vulnerable road user (VRU) behavior
- Accident and near-accident analysis
- Temporal event relationships
- Safety-critical reasoning

**Target use case:** offline analysis of recorded driving data, such as fleet dashcam logs, accident replay, and hard-case mining.

> This project is designed for offline video understanding and is **not intended for real-time vehicle control**.

### Research Questions

1. Can domain-specific LoRA fine-tuning improve a general-purpose VLM on safety-critical driving QA?
2. How much does **frame selection** matter under a limited visual-token budget?
3. Is concentrating frames around high-motion events actually better than preserving global temporal context?

---

## 📊 Key Results

### Experiment 1 — Domain LoRA Fine-Tuning

Evaluation on the **Automingo** driving QA validation set:

| Model | MCQ Accuracy | Lingo-Judge Agreement |
|:---|:---:|:---:|
| Qwen2.5-VL-7B Base | 76.40% | 61.71% |
| **Qwen2.5-VL-7B + LoRA** | **81.62%** | **73.74%** |
| **Improvement** | **+5.22 pp** | **+12.03 pp** |

Domain-specific LoRA fine-tuning improves both answer accuracy and agreement with the reasoning-quality evaluator, with a particularly large gain on **Lingo-Judge (+12.03 pp)**.

### Experiment 2 — Frame Sampling Ablation

To study the trade-off between temporal coverage and visual input cost, three sampling strategies were evaluated on **1,200 QA pairs from 200 VRU-Accident videos**.

| Frames | Uniform | Dense | Event-aware |
|:---:|:---:|:---:|:---:|
| **4** | 59.00% | 58.67% | **60.17%** |
| **8** | 60.08% | 59.17% | **60.25%** |
| **16** | **62.17%** | 59.58% | 62.00% |

### Main Observations

- Increasing the frame budget generally improves performance, but also substantially increases visual input cost.
- **Event-aware sampling performs best under constrained 4–8 frame budgets.**
- At 16 frames, simple **uniform sampling reaches the highest accuracy (62.17%)**.
- Surprisingly, **dense high-motion sampling consistently underperforms uniform sampling**.

This last result became one of the most interesting findings of the project.

---

## 🔬 Counterintuitive Finding: More "Important" Frames Are Not Always Better

A natural assumption for accident-video understanding is:

> **Focus more frames around the moment where the largest visual change occurs.**

The experiments suggest otherwise.

Dense sampling intentionally allocates more frames to high-motion regions. However, its accuracy remains below uniform sampling across all tested frame budgets:

| Frames | Uniform | Dense | Difference |
|:---:|:---:|:---:|:---:|
| 4 | 59.00% | 58.67% | -0.33 pp |
| 8 | 60.08% | 59.17% | -0.91 pp |
| 16 | 62.17% | 59.58% | **-2.59 pp** |

### Why might this happen?

High inter-frame pixel change does not necessarily correspond to the most useful semantic information.

Around an accident, high-motion regions may contain:

- Motion blur
- Camera shake
- Abrupt ego-vehicle movement
- Visually redundant collision frames

Meanwhile, concentrating too many samples around the collision can reduce coverage of the **pre-event context** needed to answer questions such as:

- Where did the pedestrian come from?
- Was the vehicle already approaching the pedestrian?
- What happened immediately before the collision?
- How did the relative positions of the road users change?

The results therefore suggest that for driving-video reasoning:

> **Temporal coverage can be more valuable than simply concentrating visual tokens around the highest-motion moment.**

This also explains why event-aware sampling is most useful when the frame budget is small: it attempts to preserve global context while selectively allocating limited frames to potentially informative events.

---

## 🏗️ Pipeline

```text
Automingo Dataset (Parquet)
        │
        │ binary images → decode → PNG
        ▼
5-Frame Temporal Samples
        │
        │ multi-image conversation format
        ▼
Qwen Multimodal SFT Dataset
        │
        ▼
Qwen2.5-VL-7B-Instruct
        +
       LoRA
        │
        │ Vision Encoder Frozen
        │ Attention + FFN LoRA
        ▼
Domain-Adapted Driving VLM
        │
        ├── Automingo Evaluation
        │     ├── MCQ Accuracy
        │     └── Lingo-Judge
        │
        └── VRU-Accident Evaluation
              └── Frame Sampling Ablation
                    ├── Uniform
                    ├── Dense
                    └── Event-aware
```

---

## 🧪 Fine-Tuning Configuration

| Configuration | Value |
|:---|:---|
| **Base Model** | Qwen2.5-VL-7B-Instruct |
| **Training Method** | LoRA SFT |
| **Epochs** | 2 |
| **Learning Rate** | 5e-5 |
| **Effective Batch Size** | 8 (BS=1, Grad Accum=8) |
| **LoRA Rank / Alpha** | r=64, alpha=128 |
| **LoRA Targets** | q/k/v/o_proj + up/down/gate_proj |
| **Vision Encoder** | Frozen |
| **Precision** | BF16 |
| **Attention** | FlashAttention2 |
| **Memory Optimization** | Gradient Checkpointing |
| **Hardware** | NVIDIA RTX 4090 24GB |
| **Training Time** | ~1h 18min |
| **Training Steps** | 814 |

The training configuration was designed to make domain adaptation feasible on a **single consumer 24GB GPU** while preserving the pretrained visual representation.

---

## 📂 Datasets

| Dataset | Purpose | Samples | Frames | Description |
|:---|:---|:---:|:---:|:---|
| **Automingo Train** | LoRA SFT | 3,256 | 5 | Safety-critical driving QA with answer and reasoning |
| **Automingo Val** | Base vs. LoRA evaluation | 1,055 | 5 | In-domain driving QA |
| **VRU-Accident** | Sampling ablation | 200 videos / 1,200 QA | 4 / 8 / 16 | Real-world accident and VRU interaction videos |

The VRU-Accident evaluation subset contains videos from multiple sources, including **DADA-2000, DoTA, CAP_DATA, and manually curated samples**, with six QA pairs per video.

---

## 🚀 Deployment

The fine-tuned model is deployed as a complete video-to-answer inference pipeline:

```text
Video Upload
    ↓
OpenCV Decode
    ↓
Frame Sampling
    ↓
Qwen2.5-VL Input Construction
    ↓
SGLang OpenAI-Compatible API
    ↓
Answer + Reasoning
    ↓
Gradio UI
```

### Deployment Stack

- **Inference backend:** SGLang
- **API:** OpenAI-compatible HTTP endpoint
- **Frontend:** Gradio
- **Frame strategies:** Uniform / Dense / Event-aware
- **Frame budgets:** 4 / 8 / 16
- **Output:** Answer + reasoning + selected-frame visualization
- **GPU:** Single RTX 4090 24GB

The frontend is kept stateless and communicates with the SGLang inference server through HTTP, separating model serving from application logic.

---

## 🛠️ Tech Stack

`Python 3.11` · `PyTorch 2.6` · `Transformers 4.57.6` · `Qwen2.5-VL` · `PEFT 0.20.0` · `LoRA` · `FlashAttention2` · `DeepSpeed 0.17.1` · `SGLang 0.5.18` · `Gradio` · `OpenCV`

---

## 📁 Repository Structure

```text
Qwen2.5-VL-Driving-Video-Understanding/
│
├── assets/
│   # Figures and media used in the README
│
├── deployment/
│   └── app.py
│       # Gradio demo with SGLang API backend
│
├── evaluation/
│   └── evaluate_qwen25.py
│       # Base / LoRA evaluation on Automingo
│
├── experiment2_sampling/
│   ├── sampling_strategies.py
│   │   # Uniform / Dense / Event-aware frame sampling
│   │
│   ├── evaluate_sampling.py
│   │   # Frame sampling evaluation on VRU-Accident
│   │
│   ├── benchmark_latency.py
│   │   # End-to-end inference latency benchmark
│   │
│   └── results/
│       ├── uniform_*.json
│       ├── dense_*.json
│       ├── event-aware_*.json
│       └── latency_*.json
│           # Accuracy and latency experiment results
│
├── training/
│   ├── automingo_7b_lora.sh
│   │   # Qwen2.5-VL-7B LoRA training configuration
│   │
│   ├── finetune_sweep.py
│   │   # Fine-tuning experiment utilities
│   │
│   ├── results_base_1055_both.json
│   └── results_lora_v2_1055_both.json
│       # Base vs. LoRA evaluation results
│
└── README.md
```
---


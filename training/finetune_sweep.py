"""
finetune_sweep.py

W&B sweep-ready Qwen3-VL + LoRA finetuning script for Automingo-style datasets.

How to use:
1) Create a sweep once:
   python finetune_sweep.py --sweep --project <project> --entity <entity>
   -> prints a sweep id like: entity/project/xxxxxxx
2) Put that sweep id into finetune_sweep.sh.
3) Launch agents with finetune_sweep.sh locally or via the cluster scheduler.

Single run without a sweep:
   python finetune_sweep.py

Notes:
- The shell launcher is the intended agent entrypoint for this project.
- Dataset/model/cache settings are expected to come from CLI args or environment variables.
- During sweeps, push_to_hub stays off unless explicitly enabled.
"""

import argparse
import os
import random

import torch
import wandb
from datasets import Image, load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer
from transformers import Qwen3VLForConditionalGeneration


# Sweep config
# Chosen to be "wise" for 4-bit + LoRA finetuning:
# - learning_rate: log-uniform in a safe LoRA range (5e-5 .. 5e-4)
# - warmup_ratio: small but non-trivial (2% .. 10%)
# - weight_decay: modest regularization (0 .. 0.1)
# - LoRA rank/alpha/dropout: common performant choices; rank capped to keep VRAM friendly
# - batch/grad_accum: explores throughput vs stability
# - max_steps: controls compute budget for sweeps (useful when dataset is large)
SWEEP_CONFIG = {
    "program": "finetune_sweep.py",
    "method": "bayes",
    "metric": {"name": "eval/loss_best", "goal": "minimize"},
    "parameters": {
        # Optimization
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4},
        "warmup_ratio": {"distribution": "uniform", "min": 0.02, "max": 0.10},
        # "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.10},

        # LoRA
        "lora_r": {"values": [8, 16, 32]},
        #"lora_alpha": {"values": [16, 32, 64]},
        #"lora_dropout": {"values": [0.0, 0.05, 0.10]},

        # Throughput / stability - first sweep lora_r+warmup_ratio+learning_rate at max capacity
        #"per_device_train_batch_size": {"values": [2, 4, 8]},
        #"gradient_accumulation_steps": {"values": [4, 8, 16]},
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, hf_token: str = None, compute_dtype: torch.dtype = torch.float16):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        token=hf_token,
        quantization_config=quant_config,
        # Force non-quantized modules to the requested precision. Qwen configs often default
        # to bf16, which can leak a bf16 path into fp16 training and break AMP unscale.
        torch_dtype=compute_dtype,
    )
    return model


def build_peft_config(cfg) -> LoraConfig:
    # You may need to update target_modules for different architectures.
    target_modules = ["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"]

    return LoraConfig(
        r=int(cfg["lora_r"]),
        # lora_alpha=int(cfg["lora_alpha"]),
        # lora_dropout=float(cfg["lora_dropout"]),
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def expand_to_single_image_rows(batch):
    image_cols = [f"image_{i}" for i in range(1, 6)]
    new_images = []
    new_prompts = []
    new_completions = []

    batch_size = len(batch["question"])
    answers = batch.get("ground_truth_answer")
    reasonings = batch.get("ground_truth_reasoning")

    for i in range(batch_size):
        question = (batch["question"][i] or "").strip()
        answer = ((answers[i] if answers is not None else "") or "").strip()
        reasoning = ((reasonings[i] if reasonings is not None else "") or "").strip()

        if reasoning:
            completion_text = f"{answer}\n\nReasoning:\n{reasoning}" if answer else f"Reasoning:\n{reasoning}"
        else:
            completion_text = answer

        for col in image_cols:
            if col not in batch:
                continue
            img = batch[col][i]
            if img is None or img == "":
                continue

            new_images.append([img])
            new_prompts.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ])
            new_completions.append([
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": completion_text}
                    ],
                }
            ])

    return {
        "images": new_images,
        "prompt": new_prompts,
        "completion": new_completions,
    }


def expand_to_multi_image_rows(batch):
    image_cols = [f"image_{i}" for i in range(1, 6)]
    new_images = []
    new_prompts = []
    new_completions = []

    batch_size = len(batch["question"])
    answers = batch.get("ground_truth_answer")
    reasonings = batch.get("ground_truth_reasoning")

    for i in range(batch_size):
        question = (batch["question"][i] or "").strip()
        answer = ((answers[i] if answers is not None else "") or "").strip()
        reasoning = ((reasonings[i] if reasonings is not None else "") or "").strip()

        if reasoning:
            completion_text = f"{answer}\n\nReasoning:\n{reasoning}" if answer else f"Reasoning:\n{reasoning}"
        else:
            completion_text = answer

        images_for_sample = []
        for col in image_cols:
            if col not in batch:
                continue
            img = batch[col][i]
            if img is None or img == "":
                continue
            images_for_sample.append(img)

        if not images_for_sample:
            continue

        content = [{"type": "image"} for _ in images_for_sample]
        content.append({"type": "text", "text": question})

        new_images.append(images_for_sample)
        new_prompts.append([
            {
                "role": "user",
                "content": content,
            }
        ])
        new_completions.append([
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": completion_text}
                ],
            }
        ])

    return {
        "images": new_images,
        "prompt": new_prompts,
        "completion": new_completions,
    }


def prepare_dataset(dataset, dataset_format: str, split_name: str, image_payload_mode: str):
    image_cols = [f"image_{i}" for i in range(1, 6)]
    has_automingo_cols = "question" in dataset.column_names and any(col in dataset.column_names for col in image_cols)
    map_batch_size = max(1, int(os.getenv("MAP_BATCH_SIZE", "8")))
    map_writer_batch_size = max(1, int(os.getenv("MAP_WRITER_BATCH_SIZE", str(map_batch_size))))

    if dataset_format == "raw":
        return dataset

    if dataset_format == "automingo" and not has_automingo_cols:
        raise ValueError(
            f"Split '{split_name}' does not look like Automingo format "
            f"(missing 'question' and image columns image_1..image_5)."
        )

    if dataset_format == "automingo" or (dataset_format == "auto" and has_automingo_cols):
        if image_payload_mode == "single":
            expand_fn = expand_to_single_image_rows
        elif image_payload_mode == "multi":
            expand_fn = expand_to_multi_image_rows
        else:
            raise ValueError(f"Unsupported image_payload_mode: {image_payload_mode}")

        for col in image_cols:
            if col in dataset.column_names:
                dataset = dataset.cast_column(col, Image())
        dataset = dataset.map(
            expand_fn,
            batched=True,
            batch_size=map_batch_size,
            writer_batch_size=map_writer_batch_size,
            desc=f"Automingo preprocess ({split_name}, {image_payload_mode})",
            remove_columns=dataset.column_names,
        )
        print(
            f"Applied Automingo preprocessing for split '{split_name}' "
            f"with image_payload_mode='{image_payload_mode}': {dataset}"
        )
    return dataset


def _extract_first_text(messages, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if role and message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        return str(text).strip()
    return ""


def log_eval_sample_to_wandb(
    model,
    processor,
    eval_dataset,
    seed: int,
    image_payload_mode: str,
    step: int = None,
) -> None:
    if len(eval_dataset) == 0:
        print("Skipping eval sample logging: eval dataset is empty.")
        return

    sample_index = int(seed) % len(eval_dataset)
    sample = eval_dataset[sample_index]

    prompt_messages = sample.get("prompt") or []
    completion_messages = sample.get("completion") or []
    images = sample.get("images") or []

    if not isinstance(images, list):
        images = [images]
    if len(images) == 0:
        print(f"Skipping eval sample logging: sample {sample_index} has no images.")
        return

    question_text = _extract_first_text(prompt_messages, "user")
    ground_truth = _extract_first_text(completion_messages, "assistant")

    if image_payload_mode == "multi":
        image_idx = 2 if len(images) > 2 else len(images) // 2
    else:
        image_idx = 0
    image_idx = min(image_idx, len(images) - 1)

    prediction_text = ""
    generation_error = ""
    try:
        if hasattr(processor, "apply_chat_template"):
            input_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = processor(text=[input_text], images=images, return_tensors="pt")
        else:
            model_inputs = processor(text=[question_text], images=images, return_tensors="pt")

        try:
            target_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(target_device) if torch.is_tensor(value) else value
                for key, value in model_inputs.items()
            }
        except StopIteration:
            pass

        model.eval()
        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        if "input_ids" in model_inputs and output_ids.shape[-1] > model_inputs["input_ids"].shape[-1]:
            generated_ids = output_ids[:, model_inputs["input_ids"].shape[-1]:]
        else:
            generated_ids = output_ids

        prediction_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        prediction_text = prediction_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    except Exception as exc:
        generation_error = f"{type(exc).__name__}: {exc}"
        prediction_text = f"[generation_failed] {generation_error}"

    table = wandb.Table(
        columns=[
            "sample_index",
            "image_payload_mode",
            "image_idx",
            "image_count",
            "question",
            "ground_truth",
            "prediction",
            "generation_error",
        ]
    )
    table.add_data(
        sample_index,
        image_payload_mode,
        image_idx,
        len(images),
        question_text,
        ground_truth,
        prediction_text,
        generation_error,
    )

    log_payload = {"eval_sample/table": table}
    try:
        log_payload["eval_sample/middle_image"] = wandb.Image(
            images[image_idx],
            caption=f"sample={sample_index}, idx={image_idx}, count={len(images)}",
        )
    except Exception:
        pass

    if step is None:
        wandb.log(log_payload)
    else:
        wandb.log(log_payload, step=int(step))
    step_label = "final" if step is None else str(int(step))
    print(
        f"Logged eval sample to W&B at global_step={step_label} "
        f"(sample_index={sample_index}, image_idx={image_idx}, image_count={len(images)})."
    )


class EvalSampleLoggingCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, seed: int, image_payload_mode: str):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.seed = seed
        self.image_payload_mode = image_payload_mode

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if self.processor is None or wandb.run is None:
            return control
        target_model = model if model is not None else kwargs.get("model")
        if target_model is None:
            return control
        try:
            log_eval_sample_to_wandb(
                target_model,
                self.processor,
                self.eval_dataset,
                seed=self.seed,
                image_payload_mode=self.image_payload_mode,
                step=state.global_step,
            )
        except Exception as exc:
            print(f"Skipping eval sample logging at step {state.global_step}: {type(exc).__name__}: {exc}")
        return control

class SweepBestEvalLossCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if wandb.run is None or not metrics:
            return control

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control

        eval_loss = float(eval_loss)
        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss

        # This is the sweep metric. It's monotonic non-increasing, so the last value is the global minimum.
        wandb.log({"eval/loss_best": self.best_eval_loss}, step=int(state.global_step))
        wandb.run.summary["eval/loss_best"] = self.best_eval_loss

        return control

def main():
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    default_precision = os.getenv("PRECISION", "fp16").strip().lower()
    if default_precision not in {"fp16", "bf16"}:
        default_precision = "fp16"

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Create a W&B sweep and print the sweep id.")
    parser.add_argument("--project", type=str, default=os.getenv("WANDB_PROJECT", "qwen3vl-sft"),
                        help="W&B project name.")
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY", None),
                        help="W&B entity (team/user). Optional.")
    parser.add_argument("--dataset", type=str, default=os.getenv("DATASET", "trl-lib/llava-instruct-mix"),
                        help="HF dataset name.")
    parser.add_argument("--train_split", type=str, default=os.getenv("TRAIN_SPLIT", "train[:10%]"),
                        help="HF dataset split expression for training.")
    parser.add_argument("--eval_split", type=str, default=os.getenv("EVAL_SPLIT", "train[10%:11%]"),
                        help="HF dataset split expression for evaluation.")
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
                        help="HF model id.")
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "Qwen3-VL-8B-Instruct-Automingo"),
                        help="Base output directory.")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"),
                        help="HF token for gated/private datasets or models.")
    parser.add_argument("--dataset_format", type=str, choices=["auto", "raw", "automingo"],
                        default=os.getenv("DATASET_FORMAT", "auto"),
                        help="Dataset schema handling. auto=detect Automingo and map; raw=use as-is.")
    parser.add_argument("--image_payload_mode", type=str, choices=["single", "multi"],
                        default=os.getenv("IMAGE_PAYLOAD_MODE", "single"),
                        help="Automingo image mapping: single=one sample per image, multi=one sample with all images.")
    parser.add_argument("--debug_mode", action="store_true",
                        help="Use short smoke-test schedule (max_steps=10, logging_steps=2, eval_steps=2).")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16"], default=default_precision,
                        help="Mixed precision mode for training.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Enable pushing to hub (NOT recommended during sweeps).")
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE", "2e-4")),
                        help="Default LR for non-swept runs or non-swept parameters.")
    parser.add_argument("--warmup_ratio", type=float, default=float(os.getenv("WARMUP_RATIO", "0.03")),
                        help="Default warmup ratio for non-swept runs or non-swept parameters.")
    parser.add_argument("--weight_decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.01")),
                        help="Default weight decay for non-swept runs or non-swept parameters.")
    parser.add_argument("--lora_r", type=int, default=int(os.getenv("LORA_R", "32")),
                        help="Default LoRA rank for non-swept runs or non-swept parameters.")
    parser.add_argument("--lora_alpha", type=int, default=int(os.getenv("LORA_ALPHA", "32")),
                        help="Default LoRA alpha for non-swept runs or non-swept parameters.")
    parser.add_argument("--lora_dropout", type=float, default=float(os.getenv("LORA_DROPOUT", "0.0")),
                        help="Default LoRA dropout for non-swept runs or non-swept parameters.")
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "8")),
                        help="Default per-device train batch size for non-swept runs or non-swept parameters.")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")),
                        help="Default gradient accumulation steps for non-swept runs or non-swept parameters.")
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("MAX_STEPS", "160")),
                        help="Default max steps for non-swept runs or non-swept parameters.")
    parser.add_argument("--logging_steps", type=int, default=int(os.getenv("LOGGING_STEPS", "5")),
                        help="Default logging interval for non-swept runs or non-swept parameters.")
    parser.add_argument("--eval_steps", type=int, default=int(os.getenv("EVAL_STEPS", "20")),
                        help="Default eval/save interval for non-swept runs or non-swept parameters.")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")),
                        help="Random seed.")
    args = parser.parse_args()

    if not args.debug_mode and _env_flag("DEBUG_MODE", default=False):
        args.debug_mode = True
    if not args.push_to_hub and _env_flag("PUSH_TO_HUB", default=False):
        args.push_to_hub = True

    if args.sweep:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project, entity=args.entity)
        print("Created sweep:", sweep_id)
        return

    # W&B run (works for sweep agent and single runs)
    run = wandb.init(project=args.project, entity=args.entity, config={
        # Reasonable single-run defaults (will be overwritten by sweep agent when used)
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "seed": args.seed,
        "image_payload_mode": args.image_payload_mode,
    })
    wandb.define_metric("eval/loss_best", summary="min")
    wandb.define_metric("eval/mean_token_accuracy", summary="max")
    
    cfg = dict(run.config)

    set_seed(int(cfg["seed"]))

    # Data
    train_dataset = load_dataset(args.dataset, split=args.train_split, token=args.hf_token)
    eval_dataset = load_dataset(args.dataset, split=args.eval_split, token=args.hf_token)

    eval_limit = int(os.getenv("EVAL_LIMIT", "0"))
    if eval_limit > 0:
        eval_dataset = eval_dataset.shuffle(seed=int(cfg["seed"])).select(
            range(min(eval_limit, len(eval_dataset)))
        )

    train_dataset = prepare_dataset(
        train_dataset,
        args.dataset_format,
        args.train_split,
        args.image_payload_mode,
    )
    eval_dataset = prepare_dataset(
        eval_dataset,
        args.dataset_format,
        args.eval_split,
        args.image_payload_mode,
    )
    
    # Model + LoRA
    if args.precision == "bf16":
        if not torch.cuda.is_available():
            raise ValueError("bf16 precision requested, but CUDA is not available.")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 precision requested, but this GPU does not report bf16 support.")
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    model = build_model(args.model_name, hf_token=args.hf_token, compute_dtype=compute_dtype)
    processor = None
    try:
        min_pixels = int(os.getenv("MIN_PIXELS", str(224 * 224)))
        max_pixels = int(os.getenv("MAX_PIXELS", str(512 * 28 * 28)))

        processor = AutoProcessor.from_pretrained(
            args.model_name,
            token=args.hf_token,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        print(f"Processor pixel limits: min_pixels={min_pixels}, max_pixels={max_pixels}")
    except Exception as exc:
        print(f"Could not load processor for eval sample logging: {type(exc).__name__}: {exc}")

    peft_config = build_peft_config(cfg)

    # Always key checkpoints by immutable run id to avoid collisions.
    run_id = wandb.run.id
    run_name = wandb.run.name or run_id
    safe_run_name = str(run_name).replace("/", "_").replace(" ", "_")
    output_dir = os.path.join(args.output_dir, f"{safe_run_name}_{run_id}")
    print(f"W&B run '{run_name}' (id={run_id}) -> output_dir={output_dir}")

    # Trainer args
    if args.debug_mode:
        max_steps = 10
        logging_steps = 2
        eval_steps = 2
    else:
        max_steps = int(cfg["max_steps"])
        logging_steps = int(cfg["logging_steps"])
        eval_steps = int(cfg["eval_steps"])
        
    eval_steps = min(eval_steps, max_steps)
    
    training_args = SFTConfig(
        # Core training
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        max_grad_norm=1.0,
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        optim="adamw_8bit",

        # Sequence handling for VLMs
        max_length=None,

        # Logging / eval
        output_dir=output_dir,
        logging_steps=logging_steps,
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Practical defaults for sweeps
        save_total_limit=2,

        # Hub
        push_to_hub=bool(args.push_to_hub),
        hub_token=args.hf_token if args.push_to_hub else None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    if processor is not None:
        trainer_kwargs["processing_class"] = processor

    trainer = SFTTrainer(**trainer_kwargs)
    log_eval_samples = os.getenv("LOG_EVAL_SAMPLES", "0").strip().lower() in {"1", "true", "yes", "on"}
    if log_eval_samples:
        trainer.add_callback(
            EvalSampleLoggingCallback(
                processor=processor,
                eval_dataset=eval_dataset,
                seed=int(cfg["seed"]),
                image_payload_mode=args.image_payload_mode,
            )
        )

    trainer.add_callback(SweepBestEvalLossCallback())
    # Optional GPU stats (kept from your original script)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    train_result = trainer.train()

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        print(f"Peak reserved memory = {used_memory} GB ({used_percentage} %).")

    # Save final model for this run
    trainer.save_model(output_dir)

    # For sweeps, it’s usually better to only push the *best* run manually afterwards.
    if args.push_to_hub:
        trainer.push_to_hub(dataset_name=args.dataset)

    wandb.finish()


if __name__ == "__main__":
    main()

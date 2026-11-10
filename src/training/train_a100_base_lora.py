import os
import re
import argparse
import shutil
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

#config

MODEL_NAME = "google/flan-t5-base"
TASK_PREFIX = "transcribe phonemes to text: "

SPECIAL_TOKENS = {"additional_special_tokens": ["<SOS>", "<EOS>", "<space>", "|"]}

MAX_INPUT_LEN = 128
MAX_LABEL_LEN = 128


TRAIN_BS = 8
EVAL_BS = 8
GRAD_ACCUM = 4

EPOCHS = 15
LR = 2e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

LORA_R = 48
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

SEED = 42
LOGGING_STEPS = 50


#utiil UTILS


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_dataset_split(csv_path: str, test_size: float, seed: int) -> DatasetDict:
    df = pd.read_csv(csv_path)

    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must contain columns: source,target")

    df = df[["source", "target"]].dropna()
    df["source"] = df["source"].apply(clean_text)
    df["target"] = df["target"].apply(clean_text)

    df = df[df["source"].str.len() > 0]
    df = df[df["target"].str.len() > 0]

    ds = Dataset.from_pandas(df.reset_index(drop=True)).shuffle(seed=seed)
    split = ds.train_test_split(test_size=test_size, seed=seed)

    return DatasetDict(train=split["train"], val=split["test"])

def make_preprocess_fn(tokenizer: AutoTokenizer):
    def preprocess(batch: Dict[str, List[str]]):
        inputs = [TASK_PREFIX + s for s in batch["source"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=MAX_LABEL_LEN,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        return model_inputs
    return preprocess

def count_trainable_parameters(model) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params
    return {
        "model_name": MODEL_NAME,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": trainable_pct,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": ["q", "k", "v", "o"],
        "label_smoothing_factor": LABEL_SMOOTHING,
        "max_input_len": MAX_INPUT_LEN,
        "max_label_len": MAX_LABEL_LEN,
        "train_batch_size_per_device": TRAIN_BS,
        "eval_batch_size_per_device": EVAL_BS,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_train_batch_size": TRAIN_BS * GRAD_ACCUM,
        "bf16": True,
        "gradient_checkpointing": True,
    }

def audit_label_padding(dataset, tokenizer: AutoTokenizer, split_name: str) -> Dict[str, Any]:
    total_label_positions = 0
    ignored_label_positions = 0
    remaining_pad_token_ids = 0
    non_ignored_label_positions = 0

    for row in dataset:
        labels = row["labels"]
        total_label_positions += len(labels)
        ignored_label_positions += sum(1 for x in labels if x == -100)
        remaining_pad_token_ids += sum(1 for x in labels if x == tokenizer.pad_token_id)
        non_ignored_label_positions += sum(1 for x in labels if x != -100)

    ignored_pct = 100.0 * ignored_label_positions / total_label_positions if total_label_positions else 0.0
    audit = {
        "split": split_name,
        "total_examples": len(dataset),
        "total_label_positions": total_label_positions,
        "non_ignored_label_positions": non_ignored_label_positions,
        "ignored_label_positions_minus_100": ignored_label_positions,
        "ignored_label_percentage": ignored_pct,
        "remaining_tokenizer_pad_token_ids_in_labels": remaining_pad_token_ids,
        "padding_is_ignored_by_loss": remaining_pad_token_ids == 0,
        "ignore_index": -100,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
    }
    return audit

def audit_first_train_batch(trainer, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    total = labels.numel()
    ignored = (labels == -100).sum().item()
    remaining_pad = (labels == tokenizer.pad_token_id).sum().item()
    return {
        "batch_label_shape": list(labels.shape),
        "total_label_positions_in_batch": int(total),
        "ignored_label_positions_minus_100_in_batch": int(ignored),
        "ignored_label_percentage_in_batch": 100.0 * ignored / total if total else 0.0,
        "remaining_tokenizer_pad_token_ids_in_batch_labels": int(remaining_pad),
        "padding_is_ignored_by_loss_in_batch": remaining_pad == 0,
        "ignore_index": -100,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
    }

def save_trainer_log_history(output_dir: str, trainer) -> None:
    log_history = trainer.state.log_history
    save_json(os.path.join(output_dir, "reviewer_logs", "trainer_log_history.json"), log_history)

    if log_history:
        df = pd.DataFrame(log_history)
        df.to_csv(os.path.join(output_dir, "reviewer_logs", "trainer_log_history.csv"), index=False)

        train_loss_df = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
        eval_loss_df = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
        if not train_loss_df.empty:
            train_loss_df.to_csv(os.path.join(output_dir, "reviewer_logs", "training_loss_steps.csv"), index=False)
        if not eval_loss_df.empty:
            eval_loss_df.to_csv(os.path.join(output_dir, "reviewer_logs", "validation_loss_epochs.csv"), index=False)

        
        summary = {
            "training_loss_logged_every_n_steps": LOGGING_STEPS,
            "num_training_loss_log_points": int(len(train_loss_df)),
            "num_validation_loss_log_points": int(len(eval_loss_df)),
        }
        if not train_loss_df.empty:
            summary["last_logged_training_loss"] = float(train_loss_df.iloc[-1]["loss"])
            if "epoch" in train_loss_df.columns:
                summary["last_training_loss_epoch"] = float(train_loss_df.iloc[-1]["epoch"])
        if not eval_loss_df.empty:
            summary["last_validation_loss"] = float(eval_loss_df.iloc[-1]["eval_loss"])
            if "epoch" in eval_loss_df.columns:
                summary["last_validation_loss_epoch"] = float(eval_loss_df.iloc[-1]["epoch"])
        if "last_logged_training_loss" in summary and "last_validation_loss" in summary:
            summary["last_train_minus_validation_loss_gap"] = (
                summary["last_logged_training_loss"] - summary["last_validation_loss"]
            )
        save_json(os.path.join(output_dir, "reviewer_logs", "loss_curve_summary.json"), summary)

def compute_metrics_builder(tokenizer: AutoTokenizer):
    wer_metric = evaluate.load("wer")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = (labels != -100) * labels + (labels == -100) * tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("\n===== EVAL SAMPLES =====")
        for i in range(min(5, len(decoded_preds))):
            print("PRED:", decoded_preds[i])
            print("REF :", decoded_labels[i])
            print("-" * 40)
        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer}
    return compute_metrics


#MODEL

def create_model_and_tokenizer():
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Silence tied weights warning
    model.config.tie_word_embeddings = False

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "k", "v", "o"],
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("\n===== TRAINABLE PARAMETERS =====")
    model.print_trainable_parameters()
    print("================================\n")

    model.generation_config.num_beams = 5
    model.generation_config.max_length = MAX_LABEL_LEN
    model.generation_config.early_stopping = True

    return model, tokenizer

# ============================================================
#Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="train.csv")
    parser.add_argument("--output_dir", type=str, default="flan_t5_base_lora_a100")
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--merge_final",
        action="store_true",
        help="Optional: merge LoRA into the base model at the end. Disabled by default for diagnostic/profiling runs to save time and memory.",
    )
    args = parser.parse_args()

    if args.fresh and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    reviewer_log_dir = os.path.join(args.output_dir, "reviewer_logs")
    os.makedirs(reviewer_log_dir, exist_ok=True)

    set_seed(args.seed)

    ds = load_dataset_split(args.csv, args.test_size, args.seed)

    model, tokenizer = create_model_and_tokenizer()
    preprocess_fn = make_preprocess_fn(tokenizer)

    print("Preprocessing dataset...")
    tokenized_train = ds["train"].map(preprocess_fn, batched=True, remove_columns=ds["train"].column_names)
    tokenized_val = ds["val"].map(preprocess_fn, batched=True, remove_columns=ds["val"].column_names)

    # Reviewer-facing parameter and padding audits.
    parameter_metrics = count_trainable_parameters(model)
    save_json(os.path.join(reviewer_log_dir, "trainable_parameter_count.json"), parameter_metrics)
    print("\n===== PARAMETER JSON =====")
    print(json.dumps(parameter_metrics, indent=2))

    padding_audit = {
        "train": audit_label_padding(tokenized_train, tokenizer, "train"),
        "validation": audit_label_padding(tokenized_val, tokenizer, "validation"),
    }
    save_json(os.path.join(reviewer_log_dir, "padding_audit.json"), padding_audit)
    print("\n===== PADDING AUDIT =====")
    print(json.dumps(padding_audit, indent=2))

    loss_description = {
        "purpose": "Reviewer-facing description of the reported training and validation loss computation.",
        "model": MODEL_NAME,
        "label_smoothing_factor": LABEL_SMOOTHING,
        "training_loss_logging": f"Training loss is logged every {LOGGING_STEPS} optimisation steps by Hugging Face Seq2SeqTrainer.",
        "validation_loss_logging": "Validation loss is computed once per epoch over the full validation split.",
        "loss_reduction": "Sequence-to-sequence token-level cross-entropy/label-smoothed cross-entropy averaged over non-ignored target-label positions.",
        "ignored_label_index": -100,
        "padding_handling": "During preprocessing, target-side tokenizer pad_token_id values are replaced by -100, so padding positions are ignored by the loss.",
        "dropout_mode_difference": "Dropout is active during training mode and disabled during evaluation mode.",
        "extra_regularisation_in_reported_loss": "No additional regularisation term is manually added to the reported loss. Weight decay is used by the optimizer for parameter updates and is not added as a separate reported loss term.",
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": LOGGING_STEPS,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
    }
    save_json(os.path.join(reviewer_log_dir, "loss_computation_description.json"), loss_description)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,

        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        label_smoothing_factor=LABEL_SMOOTHING,
        weight_decay=WEIGHT_DECAY,

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,

        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LEN,
        generation_num_beams=5,

        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        skip_memory_metrics=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding="max_length",
            label_pad_token_id=-100,
        ),
        compute_metrics=compute_metrics_builder(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Batch-level confirmation from the actual Trainer dataloader.
    batch_audit = audit_first_train_batch(trainer, tokenizer)
    save_json(os.path.join(reviewer_log_dir, "first_train_batch_padding_audit.json"), batch_audit)
    print("\n===== FIRST TRAIN BATCH PADDING AUDIT =====")
    print(json.dumps(batch_audit, indent=2))

    print("Starting Training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        memory_before_train = {
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated_before_train_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "memory_reserved_before_train_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        }
    else:
        memory_before_train = {"gpu_name": None, "cuda_available": False}

    train_start_time = time.perf_counter()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_wall_clock_seconds = time.perf_counter() - train_start_time

    train_metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
    reviewer_training_metrics = {
        "run_type": "flan_t5_base_lora_15epoch_ls_0_1",
        "epochs_requested": EPOCHS,
        "label_smoothing_factor": LABEL_SMOOTHING,
        "max_input_len": MAX_INPUT_LEN,
        "max_label_len": MAX_LABEL_LEN,
        "train_wall_clock_seconds_measured_by_script": train_wall_clock_seconds,
        "train_wall_clock_hours_measured_by_script": train_wall_clock_seconds / 3600,
        "trainer_returned_metrics": train_metrics,
        "parameter_metrics": parameter_metrics,
        "memory_before_train": memory_before_train,
    }

    if torch.cuda.is_available():
        reviewer_training_metrics["peak_gpu_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reviewer_training_metrics["peak_gpu_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024 ** 3)
        reviewer_training_metrics["cuda_max_memory_note"] = (
            "Peak memory was reset immediately before trainer.train(); therefore this is the peak during the training/evaluation call, "
            "with the model already loaded. It includes the model-resident memory plus activations/optimizer/training allocations observed after reset."
        )

    save_json(os.path.join(reviewer_log_dir, "training_efficiency_metrics.json"), reviewer_training_metrics)
    print("\n===== TRAINING EFFICIENCY METRICS =====")
    print(json.dumps(reviewer_training_metrics, indent=2))

    trainer.save_state()
    save_trainer_log_history(args.output_dir, trainer)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if trainer.is_world_process_zero() and args.merge_final:
        print("Merging LoRA and saving final model...")
        merged_model = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "final_merged_model")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
    else:
        print("Skipping final LoRA merge. Use --merge_final if you need a merged checkpoint.")

    print(" Training complete.")
    print(f"Reviewer logs saved in: {reviewer_log_dir}")

if __name__ == "__main__":
    main()

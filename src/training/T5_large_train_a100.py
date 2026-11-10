import os
import re
import argparse
import shutil
import gc
import json
import time
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


#Configaration

MODEL_NAME = "google/flan-t5-large"
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

#Utils

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
        raise ValueError("CSV must contain: source,target")

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

#Model

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

    #Silence tied weights warning
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


#Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="train.csv")
    parser.add_argument("--output_dir", type=str, default="flan_a100")
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.fresh and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    set_seed(args.seed)

    ds = load_dataset_split(args.csv, args.test_size, args.seed)

    model, tokenizer = create_model_and_tokenizer()
    preprocess_fn = make_preprocess_fn(tokenizer)

    print("Preprocessing dataset...")
    tokenized_train = ds["train"].map(preprocess_fn, batched=True, remove_columns=ds["train"].column_names)
    tokenized_val = ds["val"].map(preprocess_fn, batched=True, remove_columns=ds["val"].column_names)

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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,

        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length"),
        
        compute_metrics=compute_metrics_builder(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting Training...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Start training
    trainer.train()

    # Measure and save peak GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()

        gpu_metrics = {
            "peak_gpu_memory_allocated_gb": (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            ),
            "peak_gpu_memory_reserved_gb": (
                torch.cuda.max_memory_reserved() / (1024 ** 3)
            ),
        }

        print(
            f"Peak GPU Memory Allocated: "
            f"{gpu_metrics['peak_gpu_memory_allocated_gb']:.4f} GB"
        )

        print(
            f"Peak GPU Memory Reserved: "
            f"{gpu_metrics['peak_gpu_memory_reserved_gb']:.4f} GB"
        )

        save_json(
            os.path.join(args.output_dir, "gpu_peak_memory.json"),
            gpu_metrics
        )

    # Save model as before
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if trainer.is_world_process_zero():
        print("Merging LoRA and saving final model...")
        merged_model = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "final_merged_model")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    print(" Training complete.")

if __name__ == "__main__":
    main()
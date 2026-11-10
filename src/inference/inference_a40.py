import os
import torch
import pandas as pd
import time
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import argparse

#Config



parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_csv", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--metrics_file", type=str, required=True)

args = parser.parse_args()

MODEL_PATH = args.model_path
TEST_CSV = args.test_csv
OUTPUT_FILE = args.output_file
METRICS_FILE = args.metrics_file

OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)

if OUTPUT_DIR:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
MAX_LEN = 128

PROMPT = "transcribe phonemes to text: "


#Load model

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32
    ).to(DEVICE)

    model.eval()
    return tokenizer, model


def batch_inference():
    tokenizer, model = load_model()

    print("Reading test CSV...")
    df = pd.read_csv(TEST_CSV)

    if "source" not in df.columns:
        raise ValueError("CSV must contain 'source' column.")

    inputs = [PROMPT + str(x) for x in df["source"].tolist()]
    predictions = []

    print("Running inference...")

    # Reset GPU memory counter after model loading
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Start timing only the true test-set generation loop
    start_time = time.time()

    for i in tqdm(range(0, len(inputs), BATCH_SIZE)):
        batch = inputs[i:i + BATCH_SIZE]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=MAX_LEN,
                num_beams=5,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    # Wait for all CUDA operations to finish before stopping timer
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    runtime_seconds = end_time - start_time
    num_samples = len(inputs)
    samples_per_second = num_samples / runtime_seconds
    ms_per_sample = 1000.0 * runtime_seconds / num_samples

    peak_gpu_memory_gb = None
    if DEVICE == "cuda":
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    df["model_prediction"] = predictions
    df.to_csv(OUTPUT_FILE, index=False)

    metrics = {
        "evaluation_file": TEST_CSV,
        "number_of_samples": num_samples,
        "model_path": MODEL_PATH,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LEN,
        "num_beams": 5,
        "runtime_seconds": runtime_seconds,
        "samples_per_second": samples_per_second,
        "ms_per_sample": ms_per_sample,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "device": DEVICE
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved predictions to:", OUTPUT_FILE)
    print("Saved latency metrics to:", METRICS_FILE)

    print("\n===== Test Inference Latency =====")
    print(f"Number of samples: {num_samples}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    print(f"Throughput: {samples_per_second:.4f} samples/s")
    print(f"Latency: {ms_per_sample:.4f} ms/sample")

    if peak_gpu_memory_gb is not None:
        print(f"Peak GPU memory allocated: {peak_gpu_memory_gb:.4f} GB")


if __name__ == "__main__":
    batch_inference()
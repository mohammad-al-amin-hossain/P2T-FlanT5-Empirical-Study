# P2T-FlanT5-Empirical-Study

**Research code and experimental results** accompanying the manuscript:

*An Empirical Study of Phoneme-to-Text Reconstruction for Automated Lip-Reading Using an LLM: Model Scale, Input Representation, and Noise Sensitivity*

## Overview

This study investigates phoneme-to-text (P2T) reconstruction using pretrained Flan-T5 encoder–decoder models fine-tuned with Low-Rank Adaptation (LoRA).

The experiments examine how model scale, phoneme input representation and synthetic input corruption influence sentence reconstruction accuracy and computational requirements.

Phoneme sequences are generated from the **Lip Reading Sentences 2 (LRS2)** reference transcripts using ARPAbet representations with explicit word-boundary information. The research focuses on empirical characterisation of the P2T decoding stage.

The evaluation uses a **1,080-sample duplicate-controlled LRS2 test subset**.

## Research Objectives

The study addresses three research questions:

1. **Model scale:** How do Flan-T5 Base, Large and XL compare in reconstruction accuracy and computational requirements?
2. **Input representation:** How does reconstruction performance differ across three progressively simplified phoneme representations?
3. **Noise sensitivity:** How does reconstruction accuracy change under increasing levels of phoneme substitution and deletion errors?

## Experimental Configurations

| Experiment | Configuration |
|---|---|
| Model-scale comparison | Flan-T5 Base, Large and XL with LoRA |
| V1 — Stress + Tokens | ARPAbet stress markers, special start/end tokens and word-boundary tokens |
| V2 — Stress + Pipe | ARPAbet stress markers and pipe (`\|`) word boundaries |
| V3 — No Stress + Pipe | Simplified ARPAbet phonemes without stress markers, retaining pipe boundaries |
| Noise sensitivity | Clean inputs and approximately 10%, 20% and 30% synthetic phoneme corruption |
| Computational analysis | NVIDIA A100 training and NVIDIA A40 inference |
| Error analysis | Character-level confusion patterns and representative sentence reconstruction examples |

## Main Experimental Results

All results below correspond to the 1,080-sample duplicate-controlled test subset.

| Experiment | WER (%) | CER (%) | BLEU-4 | Exact Match (%) |
|---|---:|---:|---:|---:|
| Flan-T5 Base | 7.20 | 6.08 | 0.8571 | 68.61 |
| Flan-T5 Large / V3 | 3.45 | 2.60 | 0.9316 | 83.80 |
| Flan-T5 XL | 2.05 | 1.39 | 0.9619 | 90.83 |
| Large + 10% noise | 21.69 | 13.56 | 0.5872 | 31.30 |
| Large + 20% noise | 38.99 | 25.40 | 0.3298 | 10.46 |
| Large + 30% noise | 54.97 | 37.36 | 0.1810 | 3.15 |
| Large — V1 Stress + Tokens | 4.54 | 3.56 | 0.9151 | 81.48 |
| Large — V2 Stress + Pipe | 3.72 | 2.87 | 0.9266 | 83.43 |

**WER:** Word Error Rate; **CER:** Character Error Rate; **BLEU-4:** four-gram BLEU score; **Exact Match:** normalised sentence-level exact-match accuracy.

The noisy-input conditions use recorded injected-operation rates of 9.97%, 20.00% and 29.98%, respectively, with word-boundary markers preserved.

The complete aggregate results are available in [`results/summary/`](results/summary/).

## Computational Requirements

Training was conducted on an **NVIDIA A100 80 GB GPU** using BF16 precision. Inference was evaluated separately on an **NVIDIA A40 GPU** using FP32 precision.

| Model | A100 Training Time | Peak Training Memory | A40 Inference Time | Peak Inference Memory |
|---|---:|---:|---:|---:|
| Flan-T5 Base | 12.56 h | 1.55 GB | 19.14 ms/sample | 4.53 GB |
| Flan-T5 Large | 14.78 h | 3.95 GB | 33.82 ms/sample | 12.40 GB |
| Flan-T5 XL | 27.92 h | 10.27 GB | 69.60 ms/sample | 29.27 GB |

Memory values represent peak allocated CUDA memory. Inference times are averages over batched evaluation, rather than isolated single-request latency.

## Repository Structure

```text
P2T-FlanT5-Empirical-Study/
├── README.md
├── src/
│   ├── training/
│   │   ├── train_a100_base_lora.py
│   │   ├── T5_large_train_a100.py
│   │   └── train_a100_XL.py
│   ├── inference/
│   │   └── inference_a40.py
│   └── evaluation/
│       ├── evaluation.py
│       └── requirements-a40.txt
├── models/
│   ├── T5_Base/
│   ├── T5_Large/
│   ├── T5_XL/
│   ├── T5_Large_Stress_Tokens/
│   └── T5_Large_Stress_Pipe/
├── results/
│   ├── summary/
│   └── evaluations/
├── docs/
├── hpc/
└── data/
```

### Source Code

The `src/` directory contains Python scripts for model training, inference and evaluation.

- **Training:** Flan-T5 Base, Large and XL with LoRA on NVIDIA A100.
- **Inference:** Batch-based text generation from phoneme sequences.
- **Evaluation:** WER, CER, BLEU-4, sentence-level exact match and additional error-analysis metrics.

### Model Configurations

The `models/` directory contains model configuration files, generation settings and tokenizer metadata for the five evaluated model configurations.

### Experimental Results

The `results/` directory contains aggregate evaluation outputs, including:

- Model-scale comparison.
- Phoneme representation comparison.
- Noise-sensitivity measurements.
- A100 training requirements.
- A40 inference requirements.
- Per-experiment aggregate metrics.

## Software Environment

The experiments use Python and the following principal libraries:

- PyTorch
- Hugging Face Transformers
- PEFT
- Hugging Face Datasets
- Hugging Face Evaluate
- NumPy and Pandas

The A40 evaluation dependencies are listed in [`src/evaluation/requirements-a40.txt`](src/evaluation/requirements-a40.txt).

## Dataset and Model Availability

The study uses the **Lip Reading Sentences 2 (LRS2)** dataset.

Dataset access is available to eligible researchers through the [official Oxford LRS2 website](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html), subject to the applicable data-sharing agreement.

The original LRS2-derived datasets and individual predictions are not redistributed due to data-use restrictions. Trained model weights are excluded because of their large size.

The repository provides the available source code, model configuration metadata and aggregate experimental results.

## Manuscript

**Title:** *An Empirical Study of Phoneme-to-Text Reconstruction for Automated Lip-Reading Using an LLM: Model Scale, Input Representation, and Noise Sensitivity*

**Research area:** Phoneme-to-text reconstruction, visual speech recognition, natural language processing and pretrained encoder–decoder language models.

For experimental methodology, detailed results and discussion, please refer to the accompanying manuscript.

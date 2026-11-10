# P2T-FlanT5-Empirical-Study

**Research repository (under curation)** for the manuscript *An Empirical Study of Phoneme-to-Text Reconstruction for Automated Lip-Reading Using an LLM: Model Scale, Input Representation, and Noise Sensitivity*.

This study **empirically characterises** the behaviour of existing Flan-T5 encoder–decoder models adapted using standard LoRA. It evaluates reconstruction from **reference-derived ARPAbet phoneme sequences with known word boundaries**. It does **not** propose a new P2T algorithm, measure LoRA against full fine-tuning, isolate causal effects of all preprocessing choices, or report end-to-end visual lip-reading accuracy.

> **Release status:** Work in progress. The repository documents experiments and aggregate results, but the complete data-to-model-to-output workflow is **not yet independently reproduction-tested**. Some training scripts have been repaired for publication and should not be represented as byte-identical historical executions. Trained weights are not included.

## Experiments

The reported reconstruction metrics use a **1,080-sample duplicate-controlled LRS2 test subset**. Experiments comprise:

| Study | Configurations |
| --- | --- |
| Model scale | Flan-T5 Base, Large and XL, each adapted with LoRA |
| Input representation | V1: stress + special tokens; V2: stress + pipe boundaries; V3: no stress + pipe boundaries (main Large model) |
| Input corruption | Clean input and approximately 10%, 20%, 30% synthetic phoneme substitution/deletion |
| Computational cost | **NVIDIA A100** training measurements; **NVIDIA A40** inference measurements, reported separately |
| Error analysis | Character-level and representative sentence-level reconstruction patterns |

### Selected results

| Condition | WER (%) | CER (%) | BLEU-4 | Exact match (%) |
| --- | ---: | ---: | ---: | ---: |
| Flan-T5 Base (clean) | 7.20 | 6.08 | 0.8571 | 68.61 |
| Flan-T5 Large / V3 (clean) | 3.45 | 2.60 | 0.9316 | 83.80 |
| Flan-T5 XL (clean) | 2.05 | 1.39 | 0.9619 | 90.83 |
| Flan-T5 Large, ~10% phoneme corruption | 21.69 | 13.56 | 0.5872 | 31.30 |
| Flan-T5 Large, V1 stress + tokens | 4.54 | 3.56 | 0.9151 | 81.48 |
| Flan-T5 Large, V2 stress + pipe | 3.72 | 2.87 | 0.9266 | 83.43 |

The V1 row uses the **corrected 1,080-sample evaluation package**. An earlier V1 A40 archive contains inconsistent predictions and erroneous aggregate results; the exact reason for the earlier discrepancy has not yet been established. Do not substitute the old V1 summary. See the curation notes in `docs/`.

## Repository contents

```text
P2T-FlanT5-Empirical-Study/
├── README.md
├── src/
│   ├── training/       # A100 Base, Large and XL training scripts
│   ├── inference/      # A40 inference_a40.py
│   └── evaluation/     # Evaluation script and A40 requirements
├── models/
│   ├── T5_Base/        # Configuration/tokenizer metadata
│   ├── T5_Large/
│   ├── T5_XL/
│   ├── T5_Large_Stress_Tokens/   # V1
│   └── T5_Large_Stress_Pipe/     # V2
├── hpc/                # SLURM submission script (requires local adaptation)
├── results/
│   └── summary/        # Curated aggregate CSV tables
├── docs/               # Curation and usage notes
└── data/               # Data access information only in the Git repository
```

The local research workspace may additionally contain `data/` training/test CSVs, `results/T5_*/` row-level evaluation outputs and `.log`/`.out` execution logs. **Those local-only materials are intentionally excluded from the tracked release** unless permissions and provenance are checked. Directory contents visible on a local computer do not necessarily appear on GitHub.

## Model and script provenance

Five trained model configurations are represented by small configuration and tokenizer files under `models/`; **merged model weights and intermediate checkpoints are not published here**. The main Large model is also the V3 no-stress/pipe condition.

The `src/training/` directory includes scripts for Base, Large and XL. The uploaded Base and XL script copies required structural corrections and cannot currently be established as the exact files executed in the historical runs. These repaired copies must be treated as **documented implementations**, not proof of exact training-run reproduction. Review checkpoint metadata and original logs before making stronger provenance claims.

`src/inference/inference_a40.py`, `src/evaluation/evaluation.py` and `hpc/run_all_evaluations.slurm` document the A40 workflow. The SLURM script contains environment- and path-specific assumptions and may need editing before reuse. Refer to `docs/how_to_use.md` for current guidance.

## Data and responsible sharing

LRS2 is subject to a data-sharing agreement. Apply for dataset access through the [official Oxford LRS2 page](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html). This repository is not intended to redistribute LRS2 videos, transcripts, phoneme–sentence training/test pairs, or row-level outputs derived from them without permission.

The aggregate tables in `results/summary/` are intended to provide a compact, openly inspectable record of reported measurements. Dataset curation, exact-overlap exclusion, training provenance and the V1 result require the limitations described in the paper and project documentation.

## Reuse and current limitations

The scripts require compatible Python, PyTorch, Transformers, PEFT, Datasets and Evaluate installations, as well as access to the relevant data and saved checkpoints. The A100 training and A40 inference jobs require CUDA-capable systems; the SLURM configuration is HPC-specific. Merely downloading this repository is **not sufficient to reproduce the saved predictions**, since licensed inputs and trained weights are not distributed here.

The main evaluation is decoder-stage reconstruction with correct reference-derived phonemes and explicit boundaries, not lip-reading from video. The noise tests introduce synthetic errors at evaluation time and do not reproduce all errors of a visual recogniser. Preprocessing comparisons are descriptive rather than factorial or causal ablations.

## Citation and availability

Manuscript citation, repository release tag and (if created) archived DOI will be added after final verification and co-author review. No software licence has yet been assigned; do not assume permission to reuse the code under an open-source licence until one is selected. Third-party dataset and model licences are separate.

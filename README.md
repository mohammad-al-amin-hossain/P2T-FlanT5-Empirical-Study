# Empirical Characterisation of Phoneme-to-Text Reconstruction

Repository materials for the manuscript **An Empirical Study of Phoneme-to-Text Reconstruction for Automated Lip-Reading Using an LLM: Model Scale, Input Representation, and Noise Sensitivity**.

**Status:** repository starter for curation, not yet a complete reproducibility release. Scripts and files must be inspected and added from the authors' original records. Do not claim that the repository contains a complete training-to-results pipeline until it has been tested.

## Scope

This is an empirical study of decoder-stage phoneme-to-text (P2T) reconstruction using reference-derived ARPAbet phonemes and known word boundaries. Flan-T5 models are adapted with standard LoRA. The study characterises model-scale differences, input representation, synthetic phoneme corruption, computational requirements, and prediction errors. It does **not** introduce a new decoding algorithm, establish causal effects of individual preprocessing choices, or evaluate end-to-end lip-reading.

## Repository layout

- `src/training/`: original A100 training scripts (add verified originals for Base, Large, and XL)
- `src/inference/`: A40 direct-inference code for the 1,080-sample controlled subset
- `src/evaluation/`: metric calculations and evaluation code
- `src/analysis/`: verified audit and aggregate-analysis scripts
- `hpc/`: SLURM scripts, after removing machine-specific paths
- `results/summary/`: curated **aggregate** CSV metrics, suitable for public sharing after review
- `results/figures/`: final non-data-revealing paper figures, after verifying their source
- `data/`: access instructions only; raw LRS2 text/video and dataset-derived rows not redistributed
- `models/`: model access notes only, pending checkpoint licensing / hosting review
- `docs/`: methodology, curation and replication guidance

## Results and hardware distinction

Results summaries in `results/summary/` record the manuscript's 1,080-sample accuracy evaluations. **A100 training** measurements and **A40 inference** measurements are separate; training memory is not inference memory.

The corrected stress-plus-special-tokens (V1) result is **WER 4.54%, CER 3.56%, BLEU-4 0.9151**. An earlier A40 ZIP contained an incorrect V1 run (WER 75.97%) and must not be used as the final public summary. See `docs/curation_checklist.md`.

## Data access

LRS2 is supplied for non-commercial academic research under a BBC Research & Development data-sharing agreement. Obtain the dataset from the official [Oxford LRS2 page](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html). See `data/README.md` before sharing any dataset-derived file.

## Reproduction status

The original A100 Large training script is retained. Other model training scripts and direct A40 inference/evaluation scripts were identified in the author's directory tree, but should be copied and tested here before public release. Aggregate CSVs do not themselves enable reproduction of model inference without the permitted input data and relevant checkpoints.

## Citation and license

Bibliographic citation, release DOI, software license and final code availability language will be added after the manuscript and repository are approved. No blanket license is applied to third-party models or LRS2 materials.

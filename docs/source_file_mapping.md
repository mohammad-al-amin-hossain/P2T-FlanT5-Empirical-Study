# Original folders -> proposed repository paths

| From the author's tree | Curated destination | Release note |
|---|---|---|
| `T5_base/train_a100_base_lora.py` | `src/training/` | Review imports, paths, and reproducibility |
| `T5_large/T5_large_train_a100.py` | `src/training/` | Main Large A100 training script |
| `T5_XL/train_a100_XL.py` | `src/training/` | XL A100 training script |
| `evaluation.py`, `inference.py` (root and Large) | `src/evaluation/`, `src/inference/` | Compare versions before choosing names |
| `P2T_A40_Evaluation_Results/scripts/evaluation.py` | `src/evaluation/` | A40 evaluation; verify model representation |
| `P2T_A40_Evaluation_Results/scripts/inference.py` | `src/inference/` | A40 inference |
| `P2T_A40_Evaluation_Results/scripts/*.slurm` | `hpc/` | Strip machine-specific settings |
| `P2T_A40_Evaluation_Results/summary_metrics.csv` | `results/summary/` | **Do not copy unchanged**: older V1 score may be wrong |
| `P2T_A40_Evaluation_Results/summary_inference_metrics.csv` | `results/summary/` | Verify against JSON and corrected V1 metadata |
| `*/evaluation_metrics.json`, `new_inference_metrics*.json` | `results/summary/` (curated summaries) | May contain machine paths; review first |
| `*/char_confusion_matrix.png` and paper PNGs | `results/figures/` | Use only final verified figure versions |
| `T5_large/train.csv`, `test.csv`, `stress_*/train_*.csv`, noise CSVs, predictions | **Do not publish by default** | LRS2-derived records; check licence |
| `*/final_merged_model/model.safetensors` | **Do not publish in Git** | Model weight size/licence; decide separately |
| `*.log`, SLURM `.out`/`.err` | **Do not publish by default** | May expose absolute paths/identifiers |

The original file trees were supplied by the author; this document is a staging guide, not proof the individual scripts or checkpoints have all been validated for publication.

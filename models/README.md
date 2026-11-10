# Model checkpoints

Do not place `model.safetensors` or full `final_merged_model` directories in Git history. These can be multiple GB and may be subject to upstream model licences and compute/storage constraints. Record checkpoint provenance, tokenizer requirements, generation parameters, checksums, and a download location only when sharing has been approved. If model weights are unavailable, make that limitation explicit rather than claiming full inference reproducibility.

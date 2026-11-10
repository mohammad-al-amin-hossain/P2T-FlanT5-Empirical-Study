# Pre-publication file review

- [ ] Decide repository visibility (keep private until scientific and licensing checks are complete).
- [ ] Review scripts for credentials, machine-specific paths, university usernames, and hard-coded dataset locations.
- [ ] Keep A100 training and A40 inference measurements clearly labelled and separate.
- [ ] Verify each model's script, checkpoint, tokenizer and test input representation correspond to the published run.
- [ ] Use **corrected V1** (WER 4.54%, CER 3.56%, BLEU-4 0.9151); exclude the older V1 result (WER 75.97%).
- [ ] Check V1 `<space>` / `<Space>` token spelling and preserve provenance; never silently edit recorded inputs.
- [ ] Exclude row-level LRS2 transcript/phoneme/prediction CSVs until redistribution is authorised.
- [ ] Regenerate paper figure 3 from the intended row-normalised 1,080-sample matrix, not the raw-count plot.
- [ ] Exclude `model.safetensors` until licence and hosting are agreed.
- [ ] Test a clean clone, document dependencies, then publish a versioned release.
- [ ] Obtain consent/approval from co-authors where appropriate, add code license and citation metadata.
- [ ] Optionally archive a release on Zenodo and add the real DOI.

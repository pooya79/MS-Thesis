# Thesis Writing and Finalization Plan

Current state: the IranSeda acquisition-to-dataset pipeline, including
transcription and refinement, is complete for the selected 250-hour subset.
Whisper-medium training and evaluation are also complete. The only remaining
experimental work is final Whisper-small training on the controlled mixture
with IranSeda and its fixed evaluation.

The agreed end-to-end narrative, claim boundaries, research questions, and
chapter mapping are recorded in [`FINAL_STORY.md`](FINAL_STORY.md).

Use `TODO:` placeholders for unknown values. Do not present planned work as
completed work.

## Phase A — Completed thesis and pipeline groundwork

### 1. Fix the final thesis story

- [x] State the problem, research questions, and expected contribution in one page.
- [x] Add the new contribution: creating a Persian ASR dataset from IranSeda long audio.
- [x] Separate completed work, work in progress, and optional work if resources permit.

### 2. Finish background and related work

- [x] Complete the existing ASR, Whisper, speech degradation, and fusion background.
- [x] Add short sections on web-sourced speech, long-audio segmentation, pseudo-labeling, and LLM transcript correction.
- [x] Add citations about data quality, train/test leakage, and weakly supervised ASR.

### 3. Update the introduction

- [X] Explain why additional Persian speech data is needed.
- [X] Update the objectives, research questions, contributions, and chapter outline.
- [X] Phrase the new-data experiment as an evaluation to be completed, without predicting its result.

### 4. Write the stable parts of the method chapter

- [X] Describe IranSeda scraping, source validation, and metadata collection.
- [X] Describe segmentation rules, audio format, and rejection rules.
- [X] Describe the completed Whisper transcription process and saved outputs.
- [X] Describe the completed LLM-refinement method, including strict rules against paraphrasing or adding words.
- [X] Describe quality filtering, source-level dataset splitting, training, and evaluation procedures.
- [X] Add one end-to-end pipeline diagram and distinguish the pending final training/evaluation stage.

### 5. Prepare and update the results chapter

- [x] Write the evaluation protocol, datasets, metrics, and comparison rules.
- [x] Add completed Whisper-medium aggregate and per-dataset results.
- [ ] Compare the existing small Whisper model with the same model trained using the added IranSeda data.
- [x] Put `TODO:` in the unknown final Whisper-small result cells instead of estimating values.
- [x] Draft observations, errors, limitations, and answers to research questions 1–4.

### 6. Prepare reproducibility material

- [ ] Keep a table of tool/model versions and important settings.
- [ ] Save the Whisper model name, decoding settings, LLM model and prompt, filters, split seed, and training configuration.
- [x] Record enough provenance to trace each final segment to its IranSeda source.

## Phase B — Completed data preparation and teacher evaluation

### 7. After transcription finishes

- [x] Complete IranSeda segmentation and transcription for the selected subset.
- [x] Preserve transcription outputs and provenance.
- [x] Freeze the transcription settings before LLM refinement.

### 8. After LLM refinement finishes

- [x] Complete constrained LLM refinement and quality filtering.
- [x] Preserve the refinement configuration and outputs.
- [x] Create leakage-safe dataset files without placing segments from one source in different splits.
- [x] Finalize the selected 250-hour IranSeda dataset.

### 9. After Whisper-medium training and evaluation

- [x] Complete Persian Whisper-medium training.
- [x] Evaluate on the same 21,848 eligible examples as the single-stream Whisper-small models.
- [x] Record aggregate WER/CER of 14.36%/6.68% and all per-dataset scores.

## Phase C — Remaining experiment

### 10. After final Whisper-small training finishes

- [ ] Record the exact data mixture, seed, epochs, learning rate, batch size, hardware, and runtime.
- [ ] Preserve checkpoints, logs, and the selected-checkpoint rule outside the thesis repository when large.
- [ ] Do not change the test set or evaluation rules after seeing results.

### 11. After final Whisper-small evaluation finishes

- [ ] Fill the prepared tables with WER/CER and dataset-specific results.
- [ ] Compare against the existing small-Whisper baseline under the same evaluation conditions.
- [ ] Describe improvements, regressions, and negative results without overstating them.
- [ ] Replace the `TODO:` results and answer research questions 5–6 using the final evidence.

## Phase D — Final thesis pass

### 12. Update conclusion and abstracts

- [ ] Revise contributions, limitations, future work, and conclusion from the final results.
- [ ] Write the Persian and English abstracts last and include only final numbers.
- [ ] Discuss resource limits, pseudo-label errors, LLM risks, source bias, copyright/ethics, and generalization.

### 13. Check and submit

- [ ] Replace every `TODO:` and remove text describing abandoned experiments.
- [ ] Cross-check all counts, hours, settings, tables, figures, citations, and terminology.
- [ ] Build LaTeX cleanly and fix missing references, warnings, overflows, and figure quality.
- [ ] Proofread the final PDF and archive the submitted source, PDF, configs, and result manifests.

## Minimum acceptable result if resources remain limited

The new-data experiment is still useful if it contains one controlled comparison:
the existing small Whisper baseline versus the same model trained with the added
IranSeda dataset, evaluated on the same fixed test sets with WER and CER.

# Thesis Writing and Finalization Plan

Current state: IranSeda transcription is in progress. LLM refinement, final
dataset creation, small-Whisper training, and evaluation are still pending.

The agreed end-to-end narrative, claim boundaries, research questions, and
chapter mapping are recorded in [`FINAL_STORY.md`](FINAL_STORY.md).

Use `TODO:` placeholders for unknown values. Do not present planned work as
completed work.

## Phase A — Write now while transcription is running

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
- [X] Describe the current Whisper transcription process and saved outputs.
- [X] Draft the planned LLM-refinement method, including strict rules against paraphrasing or adding words.
- [X] Draft quality filtering, source-level dataset splitting, training, and evaluation procedures.
- [X] Add one end-to-end pipeline diagram and clearly mark any stage not yet executed.

### 5. Prepare the results chapter as a template

- [ ] Write the evaluation protocol, datasets, metrics, and comparison rules now.
- [ ] Create empty tables for dataset statistics, transcript audit, and ASR results.
- [ ] Compare the existing small Whisper model with the same model trained using the added IranSeda data.
- [ ] Put `TODO:` in every unknown table cell instead of estimating values.
- [ ] Draft headings for observations, errors, limitations, and answers to research questions.

### 6. Prepare reproducibility material

- [ ] Keep a table of tool/model versions and important settings.
- [ ] Save the Whisper model name, decoding settings, LLM model and prompt, filters, split seed, and training configuration.
- [ ] Record enough provenance to trace each final segment to its IranSeda source.

## Phase B — Fill in facts as each run finishes

### 7. After transcription finishes

- [ ] Add transcription duration, failures, segment count, and total audio hours.
- [ ] Manually inspect a small, fixed sample and record common transcription errors.
- [ ] Freeze the transcription settings before LLM refinement.

### 8. After LLM refinement finishes

- [ ] Report the model, prompt, settings, failures, and number of changed/rejected transcripts.
- [ ] Audit a fixed sample before and after refinement to check that meaning was not changed.
- [ ] Create final train/dev/test files without placing segments from one source in different splits.
- [ ] Fill the final dataset-statistics table.

### 9. After small-Whisper training finishes

- [ ] Record the exact data mixture, seed, epochs, learning rate, batch size, hardware, and runtime.
- [ ] Preserve checkpoints, logs, and the selected-checkpoint rule outside the thesis repository when large.
- [ ] Do not change the test set or evaluation rules after seeing results.

### 10. After evaluation finishes

- [ ] Fill the prepared tables with WER/CER and dataset-specific results.
- [ ] Compare against the existing small-Whisper baseline under the same evaluation conditions.
- [ ] Describe improvements, regressions, and negative results without overstating them.
- [ ] Answer each research question using the final evidence.

## Phase C — Final thesis pass

### 11. Update conclusion and abstracts

- [ ] Revise contributions, limitations, future work, and conclusion from the final results.
- [ ] Write the Persian and English abstracts last and include only final numbers.
- [ ] Discuss resource limits, pseudo-label errors, LLM risks, source bias, copyright/ethics, and generalization.

### 12. Check and submit

- [ ] Replace every `TODO:` and remove text describing abandoned experiments.
- [ ] Cross-check all counts, hours, settings, tables, figures, citations, and terminology.
- [ ] Build LaTeX cleanly and fix missing references, warnings, overflows, and figure quality.
- [ ] Proofread the final PDF and archive the submitted source, PDF, configs, and result manifests.

## Minimum acceptable result if resources remain limited

The new-data experiment is still useful if it contains one controlled comparison:
the existing small Whisper baseline versus the same model trained with the added
IranSeda dataset, evaluated on the same fixed test sets with WER and CER.

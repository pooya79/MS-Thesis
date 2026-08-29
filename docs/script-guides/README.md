# Script Guides

Every maintained Python script exposes `--help`. Use the help output before running a script with custom paths.

## Guide index

- [External ASR evaluation](external-asr-evaluation.md): OpenRouter, ElevenLabs Scribe v2, and Ivira Avanegar evaluation and rescoring.
- [Dataset management](dataset-management.md): mixed test datasets, duration summaries, FLAC conversion, Hugging Face publication, transcript normalization, and long-audio concatenation.
- [IranSeda scripts](iranseda-scripts.md): audiobook discovery, inspection, downloading, verification, radio archive discovery, and metadata utilities.
- [Dataset download and preparation](dataset-download-and-preparation.md): Common Voice, FLEURS, PerSets, and Persian evaluation datasets.
- [Speech degradation](speech-degradation.md): degradation assets, degraded and noise-only datasets, demos, manifests, and validation.
- [ASR training and evaluation](asr-training-and-evaluation.md): Whisper and FastConformer training and local evaluation workflows.
- [Enhancement and fusion](enhancement-and-fusion.md): enhancement/fusion curriculum training, evaluation, and diagnosis.

## Command index

### IranSeda

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_audiobooks --help
uv run python -m ml.speech_data.data_scraping.iranseda_download --help
uv run python -m ml.speech_data.data_scraping.iranseda_radio --help
```

### Dataset download, preparation, and management

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa --help
uv run python -m ml.speech_data.scripts.download_degradation_assets --help
uv run python -m ml.speech_data.scripts.download_filimo_persian_asr --help
uv run python -m ml.speech_data.scripts.download_fleurs_persian --help
uv run python -m ml.speech_data.scripts.download_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.download_youtube_persian_asr --help
uv run python -m ml.speech_data.scripts.compute_audio_hours --help
uv run python -m ml.speech_data.scripts.create_mixed_test_dataset --help
uv run python -m ml.speech_data.scripts.summarize_hf_audio_dataset --help
uv run python -m ml.speech_data.scripts.upload_hf_audio_dataset --help
bash ml/speech_data/scripts/upload_persian_audiobook_subset.sh --help
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 --help
uv run python -m ml.speech_data.scripts.prepare_degradation_assets --help
uv run python -m ml.speech_data.scripts.prepare_filimo_persian_asr --help
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian --help
uv run python -m ml.speech_data.scripts.prepare_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.prepare_youtube_persian_asr --help
uv run python -m ml.speech_data.scripts.convert_dataset_to_flac --help
uv run python -m ml.speech_data.scripts.verify_flac_conversion --help
```

### Degradation and long-audio tools

```bash
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip --help
uv run python -m ml.speech_data.generate_degraded_dataset --help
uv run python -m ml.speech_data.generate_degraded_pairs --help
uv run python -m ml.speech_data.generate_noise_added_dataset --help
uv run python -m ml.speech_data.inspect_manifest --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.transcribe_segments --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.select_refinement_subset --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.refine_transcriptions --help
uv run python -m ml.speech_data.validate_degraded_dataset --help
```

### ASR evaluation and training

```bash
uv run python -m ml.asr.eval_openrouter_stt --help
uv run python -m ml.asr.rescore_openrouter_stt --help
uv run python -m ml.asr.eval_elevenlabs_scribe --help
uv run python -m ml.asr.eval_ivira_avanegar --help
uv run python -m ml.asr.train_whisper_small --help
uv run python -m ml.pmct.train_whisper_small --help
uv run python -m ml.asr.eval_whisper_small --help
uv run python -m ml.asr.train_whisper_large_v3_turbo --help
uv run python -m ml.asr.eval_whisper_large_v3_turbo --help
uv run python -m ml.asr.train_whisper_medium --help
uv run python -m ml.asr.eval_whisper_medium --help
uv run python -m ml.asr.train_fastconformer --help
uv run python -m ml.asr.eval_fastconformer --help
uv run python -m ml.asr.eval_mixed_dataset --help
```

### Enhancement and fusion

```bash
uv run python -m ml.fusion.train_fusion --help
uv run python -m ml.fusion.eval_fusion --help
uv run python -m ml.enhancement.diagnose_enhancement --help
```

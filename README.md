# README.md

Project: OS-Optimized Speech Emotion Recognition using Wav2Vec2 and RAVDESS

## Overview

This repository contains a Windows-focused Jupyter Notebook pipeline for Speech Emotion Recognition on RAVDESS. The current notebook:

- downloads RAVDESS through `kagglehub`
- preprocesses audio with sequential and multiprocessing benchmarks
- caches normalized 16 kHz mono waveforms
- extracts embeddings with `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- averages the last 4 hidden layers and mean-pools them into one feature vector per sample
- standardizes embeddings and applies SMOTE only to the training split
- trains an MLP classifier with early stopping
- generates OS-analysis artifacts, QA results, and one end-of-notebook showcase sample

## Current notebook behavior

- The data split is stratified at the sample level by emotion using `train_test_split`.
- Actors can appear across train, validation, and test splits in the current implementation.
- Wav2Vec2 feature extraction runs in the main process, with CUDA mixed precision when available.
- Class weights are computed and stored in the checkpoint, but `USE_CLASS_WEIGHTED_LOSS = False` by default.
- The notebook includes low-level `os.open` / `os.read` / `os.close` read preview logic and fsync-backed atomic text writes.

## Main files

- `SER_Wav2Vec2_OS_Optimized_Windows_GPU.ipynb`
  Main notebook implementation.
- `os_ser_workers.py`
  Windows multiprocessing worker helpers for preprocessing.
- `requirements.txt`
  Python package list, excluding PyTorch.
- `spec.md`
  Project specification aligned to the notebook in this repository.
- `agent.md`
  Agent-role registry aligned to the notebook workflow in this repository.

## Setup

1. Install Python 3.10 or newer.
2. Install JupyterLab, Notebook, or use VS Code notebooks.
3. Install PyTorch separately using the official selector:
   `https://pytorch.org/get-started/locally/`
4. Install the remaining packages:
   `pip install -r requirements.txt`
5. Open `SER_Wav2Vec2_OS_Optimized_Windows_GPU.ipynb`.
6. Run the notebook from top to bottom.

If KaggleHub requests credentials:

- Download `kaggle.json` from Kaggle account settings.
- Place it at `C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json`.
- Do not commit `kaggle.json`.

## Notebook flow

Run sections in this order:

1. Setup
2. System resource snapshot
3. Dataset download and discovery
4. Worker module generation
5. Filename parser and low-level file utility validation
6. Sequential vs multiprocessing preprocessing benchmark
7. Full preprocessing with cache reuse
8. Stratified train/validation/test split
9. Wav2Vec2 feature extraction
10. Embedding standardization and SMOTE
11. MLP training
12. Evaluation
13. OS analysis
14. QA checks
15. Final output checklist
16. Suggested report summary
17. Showcase sample

## Generated outputs

Primary outputs are written under `SER_Wav2Vec2_OS_Project/`.

Important files to review after a run:

- `outputs/config.json`
- `outputs/reports/system_info.json`
- `outputs/reports/dataset_summary.json`
- `data/metadata/metadata_with_cache.csv`
- `data/metadata/preprocess_manifest.json`
- `data/metadata/train_split.csv`
- `data/metadata/val_split.csv`
- `data/metadata/test_split.csv`
- `outputs/reports/preprocessing_benchmark.csv`
- `outputs/reports/feature_extraction_benchmark.csv`
- `outputs/reports/class_balance_report.csv`
- `outputs/reports/training_history.csv`
- `outputs/reports/metrics.json`
- `outputs/reports/classification_report.txt`
- `outputs/reports/confusion_matrix.csv`
- `outputs/reports/os_analysis.md`
- `outputs/reports/qa_results.json`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/training_validation_loss.png`
- `outputs/figures/training_validation_accuracy.png`
- `outputs/checkpoints/best_emotion_mlp.pt`

## Model and training details

- Feature extractor: `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- Hidden-state aggregation: average of the last 4 hidden layers
- Pooling: mean pooling
- Default feature batch size: `2`
- Classifier: MLP `1024 -> 256 -> 128 -> 8`
- Training balancing: SMOTE on training embeddings only
- Loss: cross-entropy with label smoothing; class weights are available but disabled by default

## Demo suggestions

Show these items during presentation:

1. KaggleHub dataset download
2. Parsed metadata and emotion mapping
3. low-level OS read and atomic write evidence
4. sequential vs multiprocessing benchmark
5. cache hit and miss behavior
6. Wav2Vec2 feature extraction on GPU
7. SMOTE effect on the training embedding balance
8. final metrics and confusion matrix
9. OS analysis section
10. final showcase sample playback

## Split policy

The notebook currently uses sample-level stratified splitting by emotion. This preserves class balance across train, validation, and test, but it does not enforce speaker independence, so actor overlap across splits is expected.

## Troubleshooting

- CUDA out of memory:
  Reduce `BATCH_SIZE_FEATURES` in the notebook.

- Windows multiprocessing error:
  Restart the kernel and rerun from the top after the worker module cell has executed.

- Kaggle credential error:
  Verify `kaggle.json` is in `C:\Users\<YOUR_USERNAME>\.kaggle\`.

- `ModuleNotFoundError: No module named 'imblearn'`:
  Reinstall dependencies with `pip install -r requirements.txt`.

- Feature extraction is slow on CPU:
  Use a CUDA GPU or run on a smaller debug subset first.

- The training sample count is larger than the original training file count:
  This is expected because SMOTE creates synthetic training embeddings, not duplicate audio files.

# spec.md — OS-Optimized Speech Emotion Recognition with Wav2Vec2 on RAVDESS

## 1. Project Title

**OS-Optimized Speech Emotion Recognition (SER) using Wav2Vec2 Feature Extraction on RAVDESS**

## 2. Project Goal

Build a Windows-compatible Jupyter Notebook project that trains, tests, evaluates, and reviews a Speech Emotion Recognition classifier using:

- **Dataset:** RAVDESS Emotional Speech Audio, downloaded through **KaggleHub**.
- **Feature extractor:** `facebook/wav2vec2-base`.
- **Classifier:** lightweight neural classifier trained on Wav2Vec2 embeddings.
- **Operating Systems focus:** optimize CPU, memory, file I/O, GPU utilization, caching, multiprocessing, synchronization, and benchmarking.

This is **not** a clinical, dental, or anxiety-detection project. It is a general SER classifier for the eight RAVDESS speech emotion labels.

## 3. Target User and Environment

### Target platform
- Windows 10/11
- Jupyter Notebook or JupyterLab
- Python 3.10+
- NVIDIA GPU with CUDA-capable PyTorch installation preferred
- CPU-only fallback allowed, but final report should discuss the performance gap

### Recommended hardware
- NVIDIA GPU with at least 6 GB VRAM
- 16 GB RAM recommended
- At least 10 GB free disk space for dataset, cache, extracted features, and model outputs

## 4. Dataset Requirements

### Dataset source
Use KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
```

### Data format classification
- Raw `.wav` audio files are **unstructured data** because the sound waveform has no row-column schema.
- RAVDESS filenames are **semi-structured metadata** because fields such as modality, vocal channel, emotion, intensity, statement, repetition, and actor are encoded in a fixed filename pattern.
- Generated metadata tables such as `metadata.csv`, `splits.csv`, and metric tables are **structured data**.

### Expected RAVDESS speech labels
Use the emotion code from the filename:

| Code | Emotion |
|---|---|
| `01` | neutral |
| `02` | calm |
| `03` | happy |
| `04` | sad |
| `05` | angry |
| `06` | fearful |
| `07` | disgust |
| `08` | surprised |

### Filename format
Example:

```text
03-01-05-02-02-01-12.wav
```

Meaning:

```text
modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
```

The implementation must parse filenames and generate metadata columns:

- `path`
- `filename`
- `modality`
- `vocal_channel`
- `emotion_code`
- `emotion`
- `intensity`
- `statement`
- `repetition`
- `actor`
- `duration_sec`
- `cache_path`

## 5. Functional Requirements

### FR1 — Windows GPU setup
The notebook must:
1. Install or verify required packages.
2. Detect CUDA availability.
3. Print GPU name, CUDA version, CPU count, and RAM summary.
4. Fall back to CPU if GPU is unavailable.

### FR2 — Dataset download
The notebook must:
1. Download RAVDESS using KaggleHub.
2. Recursively find all `.wav` files.
3. Validate that files follow the RAVDESS filename convention.
4. Build a metadata table.

### FR3 — Audio preprocessing
The notebook must:
1. Convert audio to mono.
2. Resample to 16 kHz for Wav2Vec2.
3. Normalize waveform amplitude safely.
4. Save preprocessed waveforms to an on-disk cache.
5. Reuse cached files when available.

### FR4 — OS-optimized preprocessing
The notebook must implement and benchmark:
1. Sequential preprocessing.
2. Multiprocessing preprocessing using Windows-compatible process spawning.
3. Dynamic task scheduling using a process pool.
4. Synchronization for cache writes using file locks.
5. Atomic file writes using temporary files plus `os.replace`.

### FR5 — File management and system-call-level operations
The project must demonstrate file management through Python OS interfaces:

- `os.scandir` or `Path.rglob` for directory traversal.
- `os.stat` for file metadata.
- `os.open`, `os.read`, `os.write`, and `os.close` in at least one low-level file utility.
- `os.fsync` for durability when writing manifest/log files.
- `os.replace` for atomic cache updates.
- `os.unlink` or `Path.unlink` for cleanup of corrupted temp/cache files.
- `numpy.memmap` for memory-mapped feature storage.

### FR6 — Actor-independent split
The notebook must avoid speaker leakage by splitting by actor:

- Train: actors 1–18
- Validation: actors 19–21
- Test: actors 22–24

The test set must contain actors never seen during training.

### FR7 — Wav2Vec2 feature extraction
The notebook must:
1. Load `facebook/wav2vec2-base` through Hugging Face Transformers.
2. Use the Wav2Vec2 processor/feature extractor on raw audio arrays.
3. Extract hidden-state embeddings from the model.
4. Apply mean pooling over valid time frames.
5. Save embeddings to `numpy.memmap` files to reduce RAM pressure.
6. Use GPU, mixed precision, and batch processing when available.

### FR8 — Classifier training
The notebook must:
1. Train a lightweight MLP classifier on Wav2Vec2 embeddings.
2. Use validation loss for early stopping.
3. Save the best model checkpoint.
4. Log epoch-level metrics.

### FR9 — Evaluation
The notebook must report:

- Accuracy
- Macro F1
- Weighted F1
- Classification report
- Confusion matrix
- Per-class performance
- Runtime benchmark table
- Cache hit/miss rate
- Sequential vs multiprocessing speed comparison

### FR10 — Testing
The notebook must include tests for:

- RAVDESS filename parsing
- Emotion label mapping
- Cache file creation
- Actor split disjointness
- No missing labels
- Feature matrix shape
- No NaN or infinite feature values
- Model prediction output shape

### FR11 — Report-ready OS analysis
The notebook must generate a concise OS analysis section explaining:

- CPU scheduling: task queue, process pool, chunking, worker count.
- Process management: Windows spawn processes, worker module design.
- Synchronization: file locks and atomic writes.
- Memory management: memmap, batching, GPU memory constraints.
- I/O management: cache reuse, manifests, fsync, cache-hit ratio.
- File system concepts: directory traversal, metadata, low-level file reads/writes.
- Performance trade-offs: batch size vs VRAM, cache speed vs disk use, multiprocessing overhead vs speedup.

## 6. Non-Functional Requirements

### Reliability
- The notebook must be restartable.
- If a cached file already exists and is valid, preprocessing should not recompute it.
- If a cache file is corrupted, delete it and regenerate it.

### Reproducibility
- Set random seeds for Python, NumPy, and PyTorch.
- Save configuration to `outputs/config.json`.
- Save metadata and split tables.

### Maintainability
- Keep reusable worker functions in a separate generated Python module so Windows multiprocessing works from Jupyter.
- Keep notebook sections clearly labeled.
- Use simple function boundaries.

### Performance
- Avoid loading all raw audio into RAM at once.
- Use memory-mapped arrays for features.
- Use GPU only in the main process to avoid CUDA multiprocessing issues on Windows.
- Use CPU processes only for audio decoding/resampling/cache preparation.

## 7. Suggested Folder Structure

```text
SER_Wav2Vec2_OS_Project/
├─ notebooks/
│  └─ SER_Wav2Vec2_OS_Optimized_Windows_GPU.ipynb
├─ os_ser_workers.py
├─ data/
│  ├─ raw/                 # KaggleHub location may be external
│  ├─ cache_audio_16k/
│  ├─ metadata/
│  └─ features/
├─ outputs/
│  ├─ checkpoints/
│  ├─ reports/
│  ├─ figures/
│  └─ logs/
├─ spec.md
├─ agent.md
└─ README_walkthrough.txt
```

## 8. Model Design

### Feature extractor
Use:

```text
facebook/wav2vec2-base
```

Feature extraction strategy:

1. Load preprocessed waveform at 16 kHz.
2. Pass batched audio to Wav2Vec2.
3. Extract `last_hidden_state`.
4. Mean-pool across valid frames.
5. Save one 768-dimensional vector per audio file.

### Classifier
Use a small MLP:

```text
Input: 768-dimensional Wav2Vec2 embedding
Dense 768 → 256
ReLU
Dropout
Dense 256 → 8 emotions
```

## 9. OS Concept Mapping to Grading

| OS concept | Implementation evidence |
|---|---|
| CPU scheduling | Process pool distributes audio preprocessing jobs dynamically. |
| Process management | Windows-compatible spawn workers in `os_ser_workers.py`. |
| Synchronization | File locks protect cache writes; atomic replacement prevents partial files. |
| Memory management | `numpy.memmap`, batch inference, mixed precision, controlled batch size. |
| I/O management | Cache reuse, metadata manifests, fsync-protected writes, benchmark logs. |
| File management | Directory traversal, stat metadata, low-level read/write utility, atomic writes. |
| Performance trade-offs | Runtime benchmark table and written analysis. |

## 10. Acceptance Criteria

The project is complete when:

1. The notebook runs from top to bottom on Windows.
2. RAVDESS is downloaded through KaggleHub.
3. Metadata is created correctly.
4. Audio preprocessing cache is created.
5. Sequential vs multiprocessing benchmark is generated.
6. Wav2Vec2 features are extracted and saved.
7. A classifier trains successfully.
8. Test metrics and confusion matrix are produced.
9. Tests pass.
10. The generated report explains how OS concepts improve or affect the ML pipeline.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| CUDA not available | Run on CPU fallback and report slower runtime. |
| Windows multiprocessing issues in notebooks | Generate a separate `os_ser_workers.py` module. |
| GPU out-of-memory | Reduce `BATCH_SIZE_FEATURES`. |
| Cache corruption from interrupted runs | Use file locks, temp files, and atomic replacement. |
| Speaker leakage | Use actor-independent split. |
| Slow feature extraction | Cache embeddings with memmap. |

## 12. Recommended Final Presentation Flow

1. Problem and objective.
2. Dataset and data format.
3. Pipeline overview.
4. Wav2Vec2 feature extraction.
5. OS optimization design.
6. Benchmark results.
7. Model evaluation results.
8. Trade-offs and limitations.
9. Demo: run notebook sections and show generated outputs.
10. Q&A preparation: file management, multiprocessing, synchronization, and memory management.

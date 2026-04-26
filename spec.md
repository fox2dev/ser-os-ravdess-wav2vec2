# spec.md — OS-Optimized Speech Emotion Recognition with Wav2Vec2 on RAVDESS

## 1. Project Title

**OS-Optimized Speech Emotion Recognition (SER) using Wav2Vec2 Feature Extraction on RAVDESS**

## 2. Project Goal

Build a Windows-compatible Jupyter Notebook that:

- downloads the RAVDESS emotional speech dataset through `kagglehub`
- preprocesses audio with Windows-safe multiprocessing and cache reuse
- extracts embeddings with `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- trains a lightweight emotion classifier
- evaluates the classifier and produces report-ready Operating Systems evidence

This project is a general Speech Emotion Recognition pipeline for the eight RAVDESS speech emotion labels. It is not a clinical or medical system.

## 3. Target Environment

### Platform

- Windows 10/11
- Jupyter Notebook, JupyterLab, or VS Code notebooks
- Python 3.10+
- NVIDIA GPU with CUDA-capable PyTorch preferred
- CPU fallback supported

### Recommended hardware

- NVIDIA GPU with at least 6 GB VRAM
- 16 GB RAM recommended
- 10 GB or more free disk space

## 4. Dataset Requirements

### Dataset source

Use KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
```

### Data format classification

- Raw `.wav` files are unstructured data.
- RAVDESS filenames are semi-structured metadata.
- Generated CSV, JSON, and metric tables are structured data.

### Expected labels

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

### Filename parsing

Example:

```text
03-01-05-02-02-01-12.wav
```

Meaning:

```text
modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
```

The implementation must parse filenames into these columns:

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

### FR1 — Windows environment setup

The notebook must:

1. verify package availability
2. detect CUDA availability
3. print GPU, CPU, RAM, and OS information
4. save configuration and system information for the report

### FR2 — Dataset discovery and metadata

The notebook must:

1. download RAVDESS using KaggleHub
2. recursively locate `.wav` files
3. validate filenames against the RAVDESS convention
4. create `metadata_with_cache.csv`
5. save a dataset summary JSON file

### FR3 — Audio preprocessing

The notebook must:

1. load audio as mono
2. resample audio to 16 kHz
3. peak-normalize safely
4. cache preprocessed waveforms as `.npy`
5. reuse valid cache files on rerun

### FR4 — OS-optimized preprocessing

The notebook must implement and benchmark:

1. sequential preprocessing
2. multiprocessing preprocessing using Windows-safe worker spawning
3. process-pool task scheduling
4. file-lock synchronization for cache writes
5. atomic cache writes with temp files and `os.replace`

### FR5 — File management and low-level OS operations

The project must demonstrate:

- directory traversal with `Path.rglob`
- file metadata access with `os.stat`
- low-level read preview with `os.open`, `os.read`, and `os.close`
- durable atomic text writes with `os.open`, `os.write`, `os.fsync`, `os.close`, and `os.replace`
- corrupted cache cleanup with `Path.unlink`
- `numpy.memmap` for feature storage

### FR6 — Stratified data split

The notebook must:

1. create train, validation, and test splits with ratios `0.75 / 0.125 / 0.125`
2. stratify at the sample level by emotion
3. ensure each split keeps full emotion coverage
4. save `train_split.csv`, `val_split.csv`, and `test_split.csv`

The current implementation does not enforce actor-disjoint or speaker-independent splitting.

### FR7 — Wav2Vec2 feature extraction

The notebook must:

1. load `jonatasgrosman/wav2vec2-large-xlsr-53-english`
2. process cached waveforms in batches
3. extract hidden-state embeddings
4. average the last 4 hidden layers
5. mean-pool embeddings over time
6. save features and labels with `numpy.memmap`
7. use GPU and CUDA mixed precision when available

### FR8 — Classifier training

The notebook must:

1. standardize embeddings using training statistics
2. apply SMOTE only to the training embeddings
3. compute class weights for checkpoint metadata
4. train an MLP classifier
5. use validation macro F1 with validation loss tie-breaking for checkpoint selection
6. use validation loss scheduling and early stopping
7. save the best checkpoint and training history

The current notebook keeps `USE_CLASS_WEIGHTED_LOSS = False` by default.

### FR9 — Evaluation and reporting

The notebook must report:

- accuracy
- macro F1
- weighted F1
- classification report
- confusion matrix
- class balance report
- preprocessing and feature-extraction benchmarks
- cache hit and miss counts
- training loss and accuracy plots
- a final showcase sample section

### FR10 — Testing

The notebook must include QA checks for:

- filename parsing
- emotion label mapping
- cache file existence
- split coverage across all labels
- feature matrix shapes
- no NaN or infinite features
- SMOTE balancing result
- prediction output shape
- normalized prediction probabilities

### FR11 — Report-ready OS analysis

The notebook must generate an OS analysis section covering:

- process scheduling and worker-count trade-offs
- Windows multiprocessing constraints
- synchronization and atomic writes
- memmap usage and RAM pressure reduction
- cache reuse and I/O behavior
- low-level file operation evidence
- batch size vs GPU memory trade-offs

## 6. Non-Functional Requirements

### Reliability

- The notebook must be restartable.
- Existing valid cache files must be reused.
- Corrupted cache files must be removed and regenerated.

### Reproducibility

- Set seeds for Python, NumPy, and PyTorch.
- Save `outputs/config.json`.
- Save metadata, split tables, metrics, and QA results.

### Maintainability

- Keep multiprocessing worker logic in `os_ser_workers.py`.
- Keep notebook sections clearly labeled.
- Keep outputs under `SER_Wav2Vec2_OS_Project/`.

### Performance

- Do not load all raw audio into RAM at once.
- Use memmap for extracted features.
- Keep CUDA work in the main process.
- Restrict multiprocessing to CPU-bound preprocessing.

## 7. Project Structure

```text
ser-os-ravdess-wav2vec2/
├─ SER_Wav2Vec2_OS_Optimized_Windows_GPU.ipynb
├─ os_ser_workers.py
├─ requirements.txt
├─ README.md
├─ spec.md
├─ agent.md
└─ SER_Wav2Vec2_OS_Project/
   ├─ data/
   │  ├─ cache_audio_16k/
   │  ├─ metadata/
   │  └─ features/
   └─ outputs/
      ├─ checkpoints/
      ├─ reports/
      ├─ figures/
      └─ logs/
```

## 8. Model Design

### Feature extractor

Use `jonatasgrosman/wav2vec2-large-xlsr-53-english`.

Strategy:

1. load cached 16 kHz waveform
2. batch audio for Wav2Vec2 inference
3. read hidden states
4. average the last 4 hidden layers
5. mean-pool across time
6. save one 1024-dimensional vector per sample

### Classifier

Use a lightweight MLP:

```text
Input: 1024
Dense 1024 → 256
LayerNorm
GELU
Dropout 0.25
Dense 256 → 128
GELU
Dropout 0.15
Dense 128 → 8
```

## 9. OS Concept Mapping

| OS concept | Implementation evidence |
|---|---|
| CPU scheduling | Process pool distributes preprocessing tasks across workers. |
| Process management | Windows-safe worker logic is isolated in `os_ser_workers.py`. |
| Synchronization | `FileLock` protects cache writes and `os.replace` prevents partial updates. |
| Memory management | `numpy.memmap`, batching, and CUDA mixed precision reduce pressure on RAM and VRAM. |
| I/O management | Cached waveforms, manifests, fsync-backed writes, and benchmark logs are generated. |
| File management | Directory traversal, `os.stat`, low-level preview reads, and atomic writes are demonstrated. |
| Performance trade-offs | The notebook records benchmark tables and an OS analysis narrative. |

## 10. Acceptance Criteria

The project is complete when:

1. the notebook runs top-to-bottom on Windows
2. RAVDESS downloads through KaggleHub
3. metadata, cache manifest, and dataset summary are generated
4. sequential and multiprocessing preprocessing are benchmarked
5. Wav2Vec2 features are extracted and saved
6. the classifier trains successfully
7. evaluation metrics and confusion matrix are produced
8. QA checks pass
9. report files are written under `SER_Wav2Vec2_OS_Project/outputs/`
10. the final notebook includes the showcase sample section

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| CUDA unavailable | Run on CPU and report the slower runtime. |
| Windows multiprocessing issues | Keep reusable workers in `os_ser_workers.py` and rerun from the top. |
| GPU out of memory | Reduce `BATCH_SIZE_FEATURES`. |
| Cache corruption after interruption | Use locks, temp files, and atomic replacement. |
| Training class imbalance | Use SMOTE on training embeddings; optional class weights remain available in configuration metadata. |
| Sample-level split allows actor overlap | Report that speaker independence is not enforced in the current notebook. |

## 12. Recommended Presentation Flow

1. problem and objective
2. dataset and filename-derived labels
3. preprocessing pipeline and cache design
4. OS optimizations and benchmarks
5. Wav2Vec2 feature extraction
6. classifier training on standardized and SMOTE-balanced embeddings
7. evaluation results
8. limitations and trade-offs
9. showcase sample playback
10. Q&A on multiprocessing, synchronization, memmap, and file operations

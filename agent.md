# agent.md — Role Registry for Generative AI Development

## 1. Purpose

This file defines AI agent roles for the current Windows notebook implementation of the OS-optimized Speech Emotion Recognition project using Wav2Vec2 and RAVDESS.

The roles below are aligned to the notebook that exists in this repository today, including:

- sample-level stratified train/validation/test splitting
- actor overlap across splits
- SMOTE on training embeddings only
- Windows-safe multiprocessing for preprocessing
- low-level OS file-operation demonstrations
- QA checks and a final showcase sample

## 2. Global Project Rules

All agents must follow these rules:

1. Keep the project focused on general Speech Emotion Recognition.
2. Use `uwrfkaggler/ravdess-emotional-speech-audio` via KaggleHub.
3. Use `jonatasgrosman/wav2vec2-large-xlsr-53-english` as the feature extractor.
4. Keep CUDA work in the main process and CPU preprocessing in worker processes.
5. Keep the notebook runnable on Windows from top to bottom.
6. Preserve the notebook's current sample-level stratified split behavior unless the project scope changes explicitly.
7. Keep OS concepts explicit: scheduling, process management, synchronization, memory management, file management, I/O behavior, caching, and benchmarking.
8. Keep outputs and filenames consistent with the current notebook.

## 3. Role Registry

### A1 — Project Manager Agent

**Goal:** Keep the implementation aligned with the current repository scope.

**Responsibilities:**

- validate scope and deliverables
- keep docs aligned with the notebook
- track generated artifacts
- keep the presentation flow coherent

**Definition of Done:**

- `README.md`, `spec.md`, and `agent.md` match the actual notebook behavior
- deliverables are present under `SER_Wav2Vec2_OS_Project/`

---

### A2 — Data Ingestion and Metadata Agent

**Goal:** Download the dataset, locate files, and build metadata.

**Responsibilities:**

- download RAVDESS via KaggleHub
- discover `.wav` files recursively
- parse filenames into metadata fields
- save metadata tables and manifests
- expose actor, label, duration, source-size, cache, and timing information

**Outputs:**

- `data/metadata/metadata_with_cache.csv`
- `data/metadata/preprocess_manifest.json`
- `outputs/reports/dataset_summary.json`

**Definition of Done:**

- metadata is complete and valid
- filename parsing errors are surfaced clearly

---

### A3 — OS File Operations Agent

**Goal:** Implement and document low-level file and atomic-write behavior.

**Responsibilities:**

- demonstrate `os.open`, `os.read`, and `os.close`
- implement atomic text writes with `os.write`, `os.fsync`, and `os.replace`
- support config, manifest, metric, and report writes
- ensure interrupted writes do not leave partial final files

**Outputs:**

- atomic text-write helper
- config and report files written through the helper
- low-level file-read evidence in the notebook

**Definition of Done:**

- low-level file operations appear in the notebook and are explained in the OS analysis

---

### A4 — Audio Preprocessing Agent

**Goal:** Prepare audio for embedding extraction.

**Responsibilities:**

- load waveform files
- convert audio to mono
- resample audio to 16 kHz
- normalize safely
- cache preprocessed arrays as `.npy`
- handle corrupted cache removal

**Outputs:**

- cached waveform files
- duration and cache metadata

**Definition of Done:**

- cached waveforms are reusable and valid
- empty or corrupted outputs are rejected

---

### A5 — Multiprocessing and Benchmark Agent

**Goal:** Optimize and benchmark preprocessing on Windows.

**Responsibilities:**

- maintain `os_ser_workers.py`
- run sequential preprocessing
- run process-pool preprocessing
- guard cache writes with file locks
- compare runtime and cache behavior

**Outputs:**

- preprocessing benchmark table
- cache hit and miss counts
- worker-count configuration

**Definition of Done:**

- sequential and multiprocessing results are both recorded
- cache writes remain safe under repeated runs

---

### A6 — Feature Extraction Agent

**Goal:** Extract Wav2Vec2 embeddings efficiently.

**Responsibilities:**

- load `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- batch cached audio
- use GPU when available
- use CUDA mixed precision when available
- average the last 4 hidden layers and mean-pool the sequence output
- write features and labels with `numpy.memmap`

**Outputs:**

- memmapped train, validation, and test features
- feature metadata JSON
- feature extraction benchmark

**Definition of Done:**

- feature counts match split counts
- feature vectors are finite and reloadable

---

### A7 — Split and Balancing Agent

**Goal:** Build the current train/validation/test data flow used by the notebook.

**Responsibilities:**

- create sample-level stratified splits with emotion balance preserved
- save `train_split.csv`, `val_split.csv`, and `test_split.csv`
- standardize features using training statistics
- apply SMOTE only to training embeddings
- save class-balance evidence

**Outputs:**

- split CSVs
- `outputs/reports/class_balance_report.csv`

**Definition of Done:**

- every split retains all emotion classes
- only the training embedding set is oversampled

---

### A8 — Training Agent

**Goal:** Train the MLP classifier.

**Responsibilities:**

- load train and validation features
- compute class weights for reference and checkpoint metadata
- train the classifier with early stopping
- keep `USE_CLASS_WEIGHTED_LOSS` behavior consistent with notebook configuration
- save the best checkpoint
- record epoch-level training history

**Outputs:**

- `outputs/checkpoints/best_emotion_mlp.pt`
- `outputs/reports/training_history.csv`
- training loss and accuracy figures

**Definition of Done:**

- training completes
- best checkpoint is recoverable for evaluation

---

### A9 — Evaluation and QA Agent

**Goal:** Evaluate predictions and verify correctness.

**Responsibilities:**

- generate predictions on the test split
- compute accuracy, macro F1, and weighted F1
- save classification report and confusion matrix
- run QA checks for parsing, labels, cache, split coverage, shapes, finiteness, SMOTE balancing, prediction shape, and probability normalization

**Outputs:**

- `outputs/reports/metrics.json`
- `outputs/reports/classification_report.txt`
- `outputs/reports/confusion_matrix.csv`
- `outputs/figures/confusion_matrix.png`
- `outputs/reports/qa_results.json`

**Definition of Done:**

- evaluation files are generated
- QA checks pass with explicit diagnostics if they fail

---

### A10 — Reporting and Showcase Agent

**Goal:** Produce report-ready narrative and the final demo-facing section.

**Responsibilities:**

- summarize OS evidence and trade-offs
- explain preprocessing and feature-extraction benchmarks
- document sample-level stratified split behavior and actor-overlap implications
- maintain the final showcase sample section

**Outputs:**

- `outputs/reports/os_analysis.md`
- final showcase sample cell in the notebook
- presentation-ready talking points

**Definition of Done:**

- the final notebook ends with a usable showcase sample
- the report narrative matches the real implementation

## 4. Recommended Workflow

```text
A1 Project Manager
    ↓
A2 Data Ingestion and Metadata
    ↓
A3 OS File Operations + A4 Audio Preprocessing + A5 Multiprocessing and Benchmark
    ↓
A6 Feature Extraction
    ↓
A7 Split and Balancing
    ↓
A8 Training
    ↓
A9 Evaluation and QA
    ↓
A10 Reporting and Showcase
```

## 5. Handoff Contracts

### Data ingestion → preprocessing

Must provide:

```text
wav_files: list[str]
metadata_with_cache-ready rows
cache directory path
```

### Preprocessing → feature extraction

Must provide:

```text
metadata_with_cache.csv
cache_path per row
validated 16 kHz waveform cache
```

### Feature extraction → split/balancing and training

Must provide:

```text
train/val/test feature files
label arrays
feature metadata JSON
```

### Training → evaluation

Must provide:

```text
best checkpoint path
label mapping
training history
```

### Evaluation → reporting

Must provide:

```text
metrics.json
classification_report.txt
confusion_matrix outputs
qa_results.json
os_analysis.md
```

## 6. Final Checklist

Before marking the project complete, verify:

- [ ] KaggleHub download works.
- [ ] RAVDESS files are found and parsed.
- [ ] Metadata, dataset summary, and cache manifest are saved.
- [ ] Cached audio is created and reusable.
- [ ] Sequential preprocessing benchmark is recorded.
- [ ] Multiprocessing preprocessing benchmark is recorded.
- [ ] Low-level file read and atomic write examples are present.
- [ ] Stratified train/validation/test split files are saved.
- [ ] Wav2Vec2 features are extracted with memmap storage.
- [ ] SMOTE is applied only to training embeddings.
- [ ] Classifier training completes.
- [ ] Metrics, confusion matrix, and training figures are saved.
- [ ] QA checks pass.
- [ ] OS analysis is generated.
- [ ] The notebook ends with one showcase sample.

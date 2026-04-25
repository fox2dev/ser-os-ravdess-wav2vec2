# agent.md — Role Registry for Generative AI Development

## 1. Purpose

This file defines the AI agent roles needed to build, train, test, evaluate, and review the OS-optimized Speech Emotion Recognition project using Wav2Vec2 and RAVDESS.

The agents should collaborate to produce a working Windows Jupyter Notebook implementation plus report-ready evidence for an Operating Systems final project.

## 2. Global Project Rules

All agents must follow these rules:

1. Keep the project focused on **general Speech Emotion Recognition**, not clinical anxiety or dentistry.
2. Use **RAVDESS Emotional Speech Audio** through KaggleHub.
3. Use **Wav2Vec2 as the feature extractor**.
4. Use **actor-independent splitting** to avoid speaker leakage.
5. Keep GPU work in the main process; use multiprocessing only for CPU audio preprocessing.
6. Make the notebook runnable on Windows.
7. Include OS concepts explicitly: process management, scheduling, synchronization, memory management, file management, I/O optimization, caching, and benchmarking.
8. Every major implementation choice must support the final report and presentation.

## 3. Role Registry

### A1 — Project Manager Agent

**Goal:** Keep the whole project aligned with the assignment and rubric.

**Responsibilities:**
- Validate scope.
- Track deliverables.
- Check that the notebook, `spec.md`, `agent.md`, and README are complete.
- Ensure report-ready explanations are generated.

**Inputs:**
- `spec.md`
- Assignment rubric
- Notebook outputs

**Outputs:**
- Completion checklist
- Presentation flow
- Final project summary

**Definition of Done:**
- All acceptance criteria from `spec.md` are satisfied.
- The final outputs support the OS project rubric.

---

### A2 — Data Ingestion and File System Agent

**Goal:** Implement dataset download, discovery, metadata creation, and file management.

**Responsibilities:**
- Download dataset using KaggleHub.
- Recursively find `.wav` files.
- Parse RAVDESS filenames.
- Build metadata tables.
- Implement low-level file utilities using `os.open`, `os.read`, `os.write`, `os.fsync`, and `os.replace`.
- Save `metadata.csv`, split CSVs, and manifest files.

**Inputs:**
- Kaggle dataset ID: `uwrfkaggler/ravdess-emotional-speech-audio`
- RAVDESS filename convention

**Outputs:**
- `metadata.csv`
- `splits_train.csv`
- `splits_val.csv`
- `splits_test.csv`
- `run_manifest.json`

**Definition of Done:**
- All valid `.wav` files are found.
- Metadata columns are complete.
- Invalid files are reported clearly.
- Low-level file operations are demonstrated in code.

---

### A3 — OS Optimization Agent

**Goal:** Implement and explain OS-level optimizations.

**Responsibilities:**
- Design Windows-compatible multiprocessing.
- Create the worker module `os_ser_workers.py`.
- Implement process pool preprocessing.
- Add cache synchronization with file locks.
- Benchmark sequential vs multiprocessing preprocessing.
- Explain scheduling, memory, synchronization, file-system, and I/O trade-offs.

**Inputs:**
- Audio file list
- Cache directory
- Worker function requirements

**Outputs:**
- Preprocessing benchmark table
- Cache hit/miss table
- OS analysis text
- Worker count recommendation

**Definition of Done:**
- Sequential and parallel runs are measured.
- Cache writes are safe.
- Runtime and speedup are reported.
- The notebook explains why each OS technique is used.

---

### A4 — Audio Preprocessing Agent

**Goal:** Prepare audio for Wav2Vec2.

**Responsibilities:**
- Load `.wav` files.
- Convert stereo/multichannel audio to mono.
- Resample audio to 16 kHz.
- Normalize waveform amplitude.
- Save cached `.npy` waveforms.
- Detect corrupted cache files and regenerate them.

**Inputs:**
- Raw `.wav` file path
- Target sample rate

**Outputs:**
- Cached 16 kHz waveform files
- Duration metadata
- Cache hit/miss status

**Definition of Done:**
- All cached audio files can be loaded.
- Each waveform is mono float32 at 16 kHz.
- No empty or invalid waveform is accepted.

---

### A5 — Wav2Vec2 Feature Extraction Agent

**Goal:** Extract reusable speech embeddings with Wav2Vec2.

**Responsibilities:**
- Load `facebook/wav2vec2-base`.
- Batch cached waveforms.
- Use GPU when available.
- Use `torch.no_grad()` for inference.
- Use mixed precision on CUDA where safe.
- Mean-pool hidden states into one vector per audio file.
- Store features in `numpy.memmap`.

**Inputs:**
- Cached waveform files
- Metadata split tables
- Wav2Vec2 model name

**Outputs:**
- `train_features.mmap`
- `val_features.mmap`
- `test_features.mmap`
- Feature metadata JSON files
- Feature extraction benchmark

**Definition of Done:**
- Feature matrices match the number of samples in each split.
- Feature dimension matches Wav2Vec2 hidden size.
- No NaN or infinite features are present.
- Extraction can resume from existing features.

---

### A6 — Model Training Agent

**Goal:** Train the emotion classifier.

**Responsibilities:**
- Load memory-mapped embeddings.
- Standardize features using training split statistics only.
- Train an MLP classifier.
- Use validation loss for early stopping.
- Save the best checkpoint.
- Log training metrics.

**Inputs:**
- Train/validation features
- Train/validation labels

**Outputs:**
- `best_emotion_mlp.pt`
- Training history CSV
- Loss and accuracy curves

**Definition of Done:**
- Training completes without errors.
- Best validation checkpoint is saved.
- Training history is reproducible.

---

### A7 — Evaluation Agent

**Goal:** Evaluate the trained model on the held-out actor-independent test split.

**Responsibilities:**
- Load the best checkpoint.
- Predict test labels.
- Compute accuracy, macro F1, weighted F1.
- Produce a classification report.
- Plot and save confusion matrix.
- Identify strongest and weakest emotion classes.

**Inputs:**
- Test features
- Test labels
- Best checkpoint

**Outputs:**
- `classification_report.txt`
- `metrics.json`
- `confusion_matrix.png`
- Evaluation summary

**Definition of Done:**
- Test metrics are generated.
- Confusion matrix is saved.
- Per-class results are explained.

---

### A8 — Testing and QA Agent

**Goal:** Verify correctness and prevent silent errors.

**Responsibilities:**
- Write tests inside the notebook.
- Test filename parsing.
- Test label mapping.
- Test actor split disjointness.
- Test cache existence.
- Test feature shapes.
- Test no NaN/Inf.
- Test prediction shape.

**Inputs:**
- Metadata
- Features
- Model predictions

**Outputs:**
- Test result section in notebook
- Pass/fail checklist

**Definition of Done:**
- All tests pass.
- Any failure gives a specific diagnostic message.

---

### A9 — Report and Presentation Agent

**Goal:** Convert implementation details into report-ready explanations.

**Responsibilities:**
- Summarize objective, dataset, data format, and method.
- Explain OS involvement clearly.
- Create benchmark tables.
- Explain trade-offs:
  - cache vs disk space
  - multiprocessing vs overhead
  - batch size vs GPU memory
  - memmap vs RAM
  - fsync safety vs write speed
- Prepare Q&A answers.

**Inputs:**
- Notebook outputs
- Metrics
- Benchmark tables
- OS analysis section

**Outputs:**
- Report-ready markdown
- Presentation outline
- Q&A preparation notes

**Definition of Done:**
- A non-technical audience can understand the flow.
- OS concepts are explicitly tied to code evidence.

---

### A10 — Code Review Agent

**Goal:** Review quality, maintainability, and reproducibility.

**Responsibilities:**
- Check that paths work on Windows.
- Check random seeds.
- Check that Kaggle credentials are not hard-coded.
- Check that cache files are ignored from version control.
- Check that model and data paths are configurable.
- Check that no training/test leakage exists.
- Check that notebook cells can run in order.

**Inputs:**
- Notebook
- Worker module
- Output folder structure

**Outputs:**
- Review checklist
- Required fixes
- Final approval notes

**Definition of Done:**
- No critical reproducibility or data leakage issue remains.
- Notebook is ready for demonstration.

## 4. Recommended Agent Workflow

```text
A1 Project Manager
    ↓
A2 Data Ingestion and File System Agent
    ↓
A3 OS Optimization Agent + A4 Audio Preprocessing Agent
    ↓
A5 Wav2Vec2 Feature Extraction Agent
    ↓
A6 Model Training Agent
    ↓
A7 Evaluation Agent
    ↓
A8 Testing and QA Agent
    ↓
A9 Report and Presentation Agent
    ↓
A10 Code Review Agent
```

## 5. Handoff Contracts

### Data ingestion → OS/audio preprocessing
Must provide:

```text
wav_files: list[str]
metadata.csv
cache directory path
```

### Audio preprocessing → feature extraction
Must provide:

```text
metadata_with_cache.csv
cache_path for each row
validated 16 kHz waveform cache
```

### Feature extraction → model training
Must provide:

```text
train_features.mmap
val_features.mmap
test_features.mmap
label arrays
feature metadata JSON
```

### Model training → evaluation
Must provide:

```text
best checkpoint path
label mapping
scaler or feature normalization settings
training history
```

### Evaluation → report
Must provide:

```text
metrics.json
classification_report.txt
confusion_matrix.png
benchmark tables
OS analysis text
```

## 6. Final Checklist for AI Builder

Before marking the project complete, verify:

- [ ] KaggleHub download works.
- [ ] RAVDESS files are found.
- [ ] Filename parser works.
- [ ] Metadata is saved.
- [ ] Audio preprocessing cache is created.
- [ ] Sequential preprocessing benchmark is recorded.
- [ ] Multiprocessing preprocessing benchmark is recorded.
- [ ] File locking and atomic writes are used.
- [ ] Actor-independent split is used.
- [ ] Wav2Vec2 features are extracted.
- [ ] Features are saved with memmap.
- [ ] Classifier trains.
- [ ] Test metrics are reported.
- [ ] Confusion matrix is saved.
- [ ] Unit tests pass.
- [ ] OS analysis section is generated.
- [ ] README walkthrough is complete.

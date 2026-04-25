# README_walkthrough.txt

Project: OS-Optimized Speech Emotion Recognition using Wav2Vec2 and RAVDESS

Current notebook behavior:
- Stratified sample-level split by emotion for train, validation, and test.
- SMOTE is applied only to the training embeddings after Wav2Vec2 feature extraction.
- The confusion matrix uses a custom blue/teal color map for readability.
- SMOTE increases the number of training samples used by the classifier, but it does not create duplicate `.wav` files.

## Files

- spec.md
  Requirements for the project.
- agent.md
  Role registry for generative AI agents.
- SER_Wav2Vec2_OS_Optimized_Windows_GPU.ipynb
  Main Jupyter Notebook implementation.
- README_walkthrough.txt
  This walkthrough file.

## Windows setup

1. Install Python 3.10 or newer.
2. Install JupyterLab or run through VS Code notebooks.
3. Install NVIDIA driver if using GPU.
4. Install PyTorch with CUDA using the official PyTorch selector.
5. Install Python packages:
   `pip install -r requirements.txt`
6. Open the notebook.
7. Run the setup cells.
8. If KaggleHub requests credentials:
   - Download kaggle.json from your Kaggle account settings.
   - Place it at:
     C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json
   - Do not commit kaggle.json to GitHub.

## How to run

1. Open:
   SER_Wav2Vec2_OS_Optimized_Windows_GPU.ipynb

2. Run sections in order:
   - Setup
   - Dataset download
   - Worker module generation
   - Audio preprocessing benchmark
   - Full preprocessing
   - Stratified split
   - Wav2Vec2 feature extraction
   - Embedding standardization + SMOTE
   - Classifier training on balanced embeddings
   - Evaluation
   - OS analysis
   - Tests

3. Main outputs will be generated under:
   SER_Wav2Vec2_OS_Project/

4. New reports to check after running:
   - `outputs/reports/class_balance_report.csv`
   - `outputs/reports/confusion_matrix.csv`
   - `outputs/figures/confusion_matrix.png`

## What to show in the demo

1. Dataset download through KaggleHub.
2. Metadata table with parsed emotion labels.
3. Sequential vs multiprocessing benchmark.
4. Cache files and manifest files.
5. Wav2Vec2 GPU feature extraction.
6. Stratified split counts showing that `neutral` remains the minority class before SMOTE.
7. `class_balance_report.csv` showing that only the training embeddings were oversampled.
8. Training logs.
9. Evaluation metrics and the updated confusion matrix.
10. OS analysis section.

## Troubleshooting

- CUDA out of memory:
  Reduce BATCH_SIZE_FEATURES in the notebook.

- Windows multiprocessing error:
  Restart the kernel and run the notebook from the top. Make sure the worker module cell has already created os_ser_workers.py.

- Kaggle credential error:
  Add kaggle.json to C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json.

- Dataset download is slow:
  Re-run the cell. KaggleHub caches downloaded datasets.

- Feature extraction is slow on CPU:
  Use a CUDA GPU or reduce the notebook to a smaller debug subset first.

- `ModuleNotFoundError: No module named 'imblearn'`:
  Install dependencies again with `pip install -r requirements.txt`.

- The number of training samples looks larger than the number of files:
  This is expected after SMOTE. The notebook creates synthetic embedding vectors for minority classes, not duplicate audio files.

## Expected result

The notebook should train a classifier for the eight RAVDESS emotions:
neutral, calm, happy, sad, angry, fearful, disgust, surprised.

The split CSV files should still reflect the real dataset files only. The classifier training set can be larger because SMOTE operates on the extracted embedding vectors after scaling.

The report should emphasize that the OS contribution is not the model itself, but the resource-management pipeline around the model:
multiprocessing, scheduling, synchronization, memory-mapped files, file management, caching, system-call-level operations, and benchmarked trade-offs.

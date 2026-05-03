# 🏥 MedClassify — Medical Report Classifier v2.0

A machine learning web app that classifies medical transcription reports into **23 specialties** using TF-IDF + LinearSVC, trained on the MTSamples dataset.

---

## 🌟 Features

- 🔍 **Single report prediction** with confidence score and colour-coded confidence guide
- 📂 **Bulk CSV classification** — upload a CSV and classify all records at once
- ✨ **Symptom expander** — short inputs like `"cough and cold"` or `"chest pain"` are automatically expanded to clinical context for better accuracy
- ⚠️ **Confidence warnings** — low/moderate/high thresholds with actionable suggestions
- 📋 **23 built-in example templates** — one per supported specialty, auto-fills all fields
- 📊 **Top 5 specialty candidates** with probability bars
- 🔑 **Keyword hint chips** — shows typical keywords for the predicted specialty
- 🌓 **Light / dark mode** toggle
- 📉 **Confusion matrix viewer** — per-class recall heatmap from last training run

---

## 🤖 Tech Stack

| Component   | Detail                                              |
|-------------|-----------------------------------------------------|
| Frontend    | Streamlit                                           |
| Model       | LinearSVC (CalibratedClassifierCV)                  |
| Vectorizer  | TF-IDF — unigrams + bigrams, 50k features           |
| Weighting   | Keywords ×5 · Description ×2 · Transcription ×1    |
| Dataset     | MTSamples — 4,999 records                           |
| Classes     | 23 specialties (MIN_SAMPLES = 40)                   |
| Libraries   | scikit-learn, pandas, numpy, joblib, streamlit      |

---

## 📊 Model Performance

- **Accuracy:** ~78–79% on test set (20% held-out split, stratified)
- **Algorithm:** LinearSVC with best-C cross-validation search over `[1, 2, 3, 5, 8, 10, 15]`
- **Input:** keywords (×5) + description (×2) + transcription (×1) concatenated
- **Split:** 80% train / 20% test, `random_state=42`, stratified by class

### 23 Supported Specialties

| | | |
|---|---|---|
| Cardiovascular / Pulmonary | Consult - History and Phy. | Discharge Summary |
| ENT - Otolaryngology | Emergency Room Reports | Gastroenterology |
| General Medicine | Hematology - Oncology | Nephrology |
| Neurology | Neurosurgery | Obstetrics / Gynecology |
| Office Notes | Ophthalmology | Orthopedic |
| Pain Management | Pediatrics - Neonatal | Podiatry |
| Psychiatry / Psychology | Radiology | SOAP / Chart / Progress Notes |
| Surgery | Urology | |

> Specialties with fewer than 40 samples in MTSamples are excluded (e.g. Allergy/Immunology, Bariatrics). To include more classes, lower `MIN_SAMPLES` in `train_model.py` — but expect reduced accuracy on underrepresented classes.

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Download MTSamples from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) and place `mtsamples.csv` in the project root.

### 3. Train the model (run once)
```bash
python train_model.py
```
This runs a cross-validation hyperparameter search, trains the final pipeline, saves all model artefacts to `model/`, and generates the confusion matrix PNG.

### 4. Launch the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
medical_report_classifier/
├── app.py                        ← Streamlit web application
├── train_model.py                ← Preprocessing + model training
├── api.py                        ← API layer
├── requirements.txt              ← Python dependencies
├── mtsamples.csv                 ← Dataset (place here)
├── medclassify.html              ← Standalone HTML interface
├── .streamlit/
│   └── config.toml               ← Streamlit theme config
└── model/                        ← Auto-created after training
    ├── pipeline.joblib            ← TF-IDF + LinearSVC pipeline
    ├── label_encoder.joblib       ← LabelEncoder for 23 classes
    ├── specialty_info.json        ← Keyword hints per specialty
    ├── confusion_matrix.png       ← Row-normalised recall heatmap
    └── confusion_matrix.csv       ← Raw confusion matrix data
```

---

## 💡 Tips for Best Results

**Keywords carry the most weight (×5 at inference).** For high-confidence predictions:

1. **Include the specialty name in keywords** — e.g. `cardiovascular pulmonary, chest pain, ECG, troponin`
2. **Use all three fields** — transcription + description + keywords together outperform any single field
3. **Short inputs trigger the symptom expander** — inputs under 15 words are automatically expanded to clinical vocabulary
4. **Low confidence?** Add medications, diagnosis terms, or procedure names specific to the specialty

### Confidence thresholds

| Score | Meaning |
|-------|---------|
| ≥ 70% | High confidence — reliable prediction |
| 40–69% | Moderate — add more clinical detail |
| < 40% | Low — input may be too short or ambiguous |

---

## 📋 Bulk CSV Format

Upload a CSV with any combination of these columns:

```
description, transcription, keywords
```

At least one column is required. Missing columns default to empty strings. Output adds `predicted_specialty` and `confidence` columns and is downloadable as CSV.

---

## 🔧 Retraining / Customisation

To change the number of supported classes, edit `MIN_SAMPLES` in `train_model.py`:

```python
MIN_SAMPLES = 40   # default — 23 classes
MIN_SAMPLES = 20   # more classes, lower accuracy
MIN_SAMPLES = 100  # fewer classes, higher accuracy per class
```

Then retrain:
```bash
python train_model.py
streamlit run app.py
```

---

## 📦 Dataset

- **Source:** [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) — real de-identified medical transcription samples
- **Total records:** 4,999
- **Text fields used:** `description`, `transcription`, `keywords`
- **License:** Public domain / Kaggle dataset
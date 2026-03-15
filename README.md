# medical_report_classifier
🏥 A Streamlit web app that classifies medical transcriptions into 40 specialties using TF-IDF + Logistic Regression ML model trained on MTSamples dataset.
# 🏥 Medical Report Classifier

A machine learning web app that classifies medical transcription reports 
into 40 specialties using TF-IDF + Logistic Regression.

## 🌟 Features
- 🔍 Single report prediction with confidence score
- 📂 Bulk CSV upload and classification
- ✨ Symptom expander for short inputs (e.g. "cough and cold")
- ⚠️ Low confidence warnings with suggestions
- 📋 8 built-in example medical report templates
- 📊 Top 5 specialty candidates with probability bars

## 🤖 Tech Stack
| Component | Detail |
|---|---|
| Frontend | Streamlit |
| Model | Logistic Regression |
| Vectorizer | TF-IDF (unigrams + bigrams, 50k features) |
| Dataset | MTSamples — 4,999 records, 40 specialties |
| Libraries | scikit-learn, pandas, numpy, joblib |

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Download MTSamples from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
and put `mtsamples.csv` in the project folder.

### 3. Train the model (run once)
```bash
python train_model.py
```

### 4. Launch the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

## 📁 Project Structure
```
medical_classifier/
├── app.py               ← Streamlit web application
├── train_model.py       ← Preprocessing + model training
├── requirements.txt     ← Python dependencies
├── mtsamples.csv        ← Dataset (place here)
└── model/               ← Auto-created after training
    ├── pipeline.joblib
    ├── label_encoder.joblib
    └── specialty_info.json
```

## 📊 Model Performance
- **Accuracy:** ~92% on test set
- **Input:** description + transcription + keywords combined
- **Classes:** 40 medical specialties
- **Split:** 80% train / 20% test

## 📋 Dataset
- **Source:** MTSamples (4,999 medical transcription samples)
- **Specialties:** 40 (Surgery, Cardiology, Neurology, Orthopedic, etc.)
- **Text fields:** description, transcription, keywords

"""
api.py  —  FastAPI bridge for MedClassify HTML frontend
────────────────────────────────────────────────────────
Place this file in the SAME folder as app.py and your model/ directory.

Install deps (once):
    pip install fastapi uvicorn

Run:
    uvicorn api:app --reload --port 8000

Then open medclassify.html in your browser — it will call this server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
import joblib
import json
import re

# ── Load the same model used by app.py ────────────────────────────────────────
pipeline = joblib.load("model/pipeline.joblib")
le       = joblib.load("model/label_encoder.joblib")

with open("model/specialty_info.json") as f:
    specialty_info = json.load(f)

# ── Reuse helpers from app.py (copy-pasted to keep api.py self-contained) ────

SYMPTOM_EXPANSIONS = {
    "cough": "persistent cough dry cough productive cough respiratory tract upper airway",
    "cold": "common cold rhinitis nasal congestion runny nose sneezing upper respiratory infection",
    "cough and cold": "patient presents with cough cold runny nose nasal congestion sore throat sneezing mild fever upper respiratory tract infection rhinitis antihistamines decongestants",
    "fever": "elevated temperature febrile pyrexia chills rigors sweating infection inflammation",
    "headache": "headache migraine cephalgia pain head pressure throbbing nausea photophobia neurology",
    "chest pain": "chest pain angina pectoris myocardial infarction cardiac ischemia ECG troponin shortness of breath cardiovascular",
    "back pain": "lower back pain lumbar pain spinal disc herniation musculoskeletal orthopedic",
    "stomach pain": "abdominal pain epigastric pain gastritis peptic ulcer nausea vomiting gastrointestinal",
    "knee pain": "knee pain joint pain arthritis orthopedic ligament meniscus swelling",
    "diabetes": "diabetes mellitus type 2 hyperglycemia insulin blood glucose HbA1c endocrinology general medicine",
    "hypertension": "hypertension high blood pressure cardiovascular antihypertensive medication",
    "anxiety": "anxiety disorder generalized anxiety panic attacks psychiatric mental health",
    "depression": "major depressive disorder mood disorder psychiatry antidepressants counseling",
    "skin rash": "dermatitis eczema skin rash erythema lesions dermatology pruritus",
    "urinary": "urinary tract infection dysuria frequency urgency urology kidney bladder",
    "pregnancy": "obstetrics pregnancy prenatal antenatal gestational gynecology",
    "fracture": "bone fracture orthopedic trauma X-ray cast immobilization",
    "cancer": "malignancy tumor oncology chemotherapy radiation biopsy",
    "asthma": "asthma bronchospasm wheezing shortness of breath inhaler bronchodilator pulmonary cardiovascular",
    "allergy": "allergic reaction hypersensitivity antihistamine rhinitis immunology general medicine",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_symptoms(text: str):
    if not text or len(text.split()) > 15:
        return text, False
    text_lower = text.lower().strip()
    if text_lower in SYMPTOM_EXPANSIONS:
        return SYMPTOM_EXPANSIONS[text_lower], True
    for key, expansion in SYMPTOM_EXPANSIONS.items():
        if key in text_lower or text_lower in key:
            return expansion + " " + text, True
    if len(text.split()) <= 3:
        return text + " " + text + " " + text + " patient presents with symptoms diagnosis treatment", True
    return text, False


def predict_single(description="", transcription="", keywords=""):
    trans_expanded, trans_was = expand_symptoms(transcription)
    desc_expanded,  desc_was  = expand_symptoms(description)

    combined = (
        clean_text(keywords) + " " + clean_text(keywords) + " " +
        clean_text(keywords) + " " + clean_text(keywords) + " " +
        clean_text(keywords) + " " +
        clean_text(desc_expanded) + " " + clean_text(desc_expanded) + " " +
        clean_text(trans_expanded)
    ).strip()

    if not combined:
        return None, None, None, False

    proba      = pipeline.predict_proba([combined])[0]
    top5_idx   = np.argsort(proba)[::-1][:5]
    top5_labels = le.inverse_transform(top5_idx)
    top5_scores = proba[top5_idx]

    return (
        top5_labels[0],
        float(top5_scores[0]),
        list(zip(top5_labels.tolist(), top5_scores.tolist())),
        trans_was or desc_was
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MedClassify API", version="1.0")

# Allow the HTML file to call this API from any origin (file:// or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    transcription: str = ""
    description:   str = ""
    keywords:      str = ""


class PredictResponse(BaseModel):
    specialty:    str
    confidence:   float                        # 0.0 – 1.0
    top5:         List[Tuple[str, float]]      # [[label, score], ...]
    was_expanded: bool


@app.get("/")
def health():
    return {"status": "ok", "model": "LinearSVC", "classes": len(le.classes_)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: ReportRequest):
    specialty, confidence, top5, was_expanded = predict_single(
        description   = req.description,
        transcription = req.transcription,
        keywords      = req.keywords,
    )

    if specialty is None:
        return PredictResponse(
            specialty    = "Unknown",
            confidence   = 0.0,
            top5         = [],
            was_expanded = False,
        )

    return PredictResponse(
        specialty    = specialty,
        confidence   = round(confidence, 4),
        top5         = [(label, round(float(score), 4)) for label, score in top5],
        was_expanded = was_expanded,
    )


# ── Optional: bulk CSV endpoint ───────────────────────────────────────────────
import pandas as pd
from fastapi import UploadFile, File
import io


@app.post("/predict_bulk")
async def predict_bulk(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required = {"description", "transcription", "keywords"}
    if not required.intersection(df.columns):
        return {"error": "CSV must contain at least one of: description, transcription, keywords"}

    desc  = df.get("description",   pd.Series([""] * len(df))).fillna("")
    trans = df.get("transcription", pd.Series([""] * len(df))).fillna("")
    kw    = df.get("keywords",      pd.Series([""] * len(df))).fillna("")

    combined = (
        kw.apply(clean_text) + " " + kw.apply(clean_text) + " " +
        kw.apply(clean_text) + " " + kw.apply(clean_text) + " " +
        kw.apply(clean_text) + " " +
        desc.apply(clean_text) + " " + desc.apply(clean_text) + " " +
        trans.apply(clean_text)
    )

    preds  = pipeline.predict(combined)
    probas = pipeline.predict_proba(combined).max(axis=1)
    labels = le.inverse_transform(preds)

    df["predicted_specialty"] = labels
    df["confidence"]          = (probas * 100).round(2)

    return df.to_dict(orient="records")
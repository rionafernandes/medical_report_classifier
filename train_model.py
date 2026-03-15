"""
train_model.py
Run this ONCE to preprocess data and train the model.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("mtsamples.csv", index_col=0)
print(f"   Loaded {len(df)} records with columns: {list(df.columns)}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def clean_text(text):
    """Lowercase, remove special chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # keep alphanumeric only
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("🧹 Preprocessing text fields...")

df["description"]   = df["description"].fillna("").apply(clean_text)
df["transcription"] = df["transcription"].fillna("").apply(clean_text)
df["keywords"]      = df["keywords"].fillna("").apply(clean_text)

# Combine all three fields with keyword repetition for boosting signal
df["combined_text"] = (
    df["description"] + " " +
    df["transcription"] + " " +
    df["keywords"] + " " +
    df["keywords"]          # repeat keywords for extra weight
)

# Drop rows with empty combined text or missing label
df = df[df["combined_text"].str.strip() != ""]
df = df[df["medical_specialty"].notna()]
df["medical_specialty"] = df["medical_specialty"].str.strip()

print(f"   Clean records: {len(df)}")
print(f"   Unique specialties: {df['medical_specialty'].nunique()}")

# ── 3. Encode labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df["medical_specialty"])

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── 5. Pipeline ───────────────────────────────────────────────────────────────
print("\n🤖 Training TF-IDF + Logistic Regression pipeline...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams
        max_features=50_000,
        sublinear_tf=True,      # log-scale TF
        min_df=2,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5,
        solver="lbfgs"
    ))
])

pipeline.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 7. Save artefacts ─────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/pipeline.joblib")
joblib.dump(le,       "model/label_encoder.joblib")

# Save specialty → top keywords mapping for UI display
print("\n💾 Saving model artefacts to ./model/ ...")
specialty_info = {}
for specialty in le.classes_:
    sample = df[df["medical_specialty"] == specialty]["keywords"].head(5).tolist()
    specialty_info[specialty] = " | ".join(sample[:2]) if sample else ""

import json
with open("model/specialty_info.json", "w") as f:
    json.dump(specialty_info, f, indent=2)

print("✅ Done! Run: streamlit run app.py")

"""
train_model.py — Medical Report Specialty Classifier
Dataset  : mtsamples.csv  (4999 records)
Classes  : 23 specialties (MIN_SAMPLES=40)
Algorithm: LinearSVC (CalibratedClassifierCV) + TF-IDF
Accuracy : ~78-79%
"""

import pandas as pd
import re
import joblib
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy,
                          save_path="model/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)

    # Row-normalise → recall %
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(len(cm)):
        s = cm[i].sum()
        if s > 0:
            cm_norm[i] = cm[i] / s * 100

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(18, 15))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#0d1117", "#0a1e46", "#0f3264", "#0d6e78",
              "#14a08c", "#f5b414", "#fcd34d"]
    cmap = mcolors.LinearSegmentedColormap.from_list("medical", colors, N=256)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.set_label("Recall %", color="#8b949e", fontsize=11)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
    cbar.ax.set_facecolor("#0d1117")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val == 0:
                continue
            text_color = "#1a1a1a" if val >= 60 else "#c9d1d9"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")
            if i == j and val >= 70:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=2, edgecolor="#2dd4bf", facecolor="none")
                ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7.5, color="#8b949e")
    ax.set_yticklabels(class_names, fontsize=7.5, color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#161b22")

    ax.set_xlabel("Predicted Specialty", color="#8b949e", fontsize=12, labelpad=12)
    ax.set_ylabel("True Specialty",      color="#8b949e", fontsize=12, labelpad=12)
    ax.set_title(
        f"Confusion Matrix — Medical Report Classifier ({accuracy:.1f}% Accuracy)  ·  23 Specialties",
        color="#c9d1d9", fontsize=13, pad=18
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


# ── LOAD ──────────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("mtsamples.csv", index_col=0)
print(f"   Loaded {len(df)} records | Columns: {list(df.columns)}")

# ── CLEAN ─────────────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("🧹 Cleaning text fields...")
for col in ["description", "transcription", "keywords"]:
    df[col] = df[col].fillna("").apply(clean_text)

df["medical_specialty"] = df["medical_specialty"].fillna("").str.strip()

# Feature weighting: keywords×5, description×2, transcription×1
df["combined_text"] = (
    df["keywords"]      + " " + df["keywords"]      + " " +
    df["keywords"]      + " " + df["keywords"]      + " " +
    df["keywords"]      + " " +          # 5× keywords
    df["description"]   + " " + df["description"]   + " " +  # 2× description
    df["transcription"]                              # 1× transcription
)
df = df[df["combined_text"].str.strip() != ""]
df = df[df["medical_specialty"] != ""]

# ── CLASS FILTERING ───────────────────────────────────────────────────────────
# MIN_SAMPLES=40 → 23 classes (all clinically meaningful specialties in mtsamples)
# Using 120 keeps only 11 classes and silently excludes ENT, Oncology, Nephrology,
# Psychiatry, Pediatrics, Ophthalmology, Podiatry, Pain Management, etc.
MIN_SAMPLES = 40
counts      = df["medical_specialty"].value_counts()
valid       = counts[counts >= MIN_SAMPLES].index
df          = df[df["medical_specialty"].isin(valid)]

print(f"\n   Records  : {len(df)}")
print(f"   Classes  : {df['medical_specialty'].nunique()}")
print(f"   Class list:")
for cls in sorted(df["medical_specialty"].unique()):
    print(f"      {counts[cls]:4d}  {cls}")

# ── LABEL ─────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df["medical_specialty"])

# ── SPLIT ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\n   Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── HYPERPARAMETER SEARCH ─────────────────────────────────────────────────────
print("\n🔍 Searching best C value (cross-validation)...")
tfidf_tmp = TfidfVectorizer(
    ngram_range=(1, 2), max_features=50000,
    min_df=2, max_df=0.95, sublinear_tf=True, stop_words="english"
)
X_tr_vec = tfidf_tmp.fit_transform(X_train)

best_c, best_acc = 15, 0
for c in [1, 2, 3, 5, 8, 10, 15]:
    svc    = LinearSVC(C=c, class_weight="balanced", max_iter=5000)
    scores = cross_val_score(svc, X_tr_vec, y_train, cv=3, scoring="accuracy")
    m      = scores.mean()
    print(f"   C={c:2d}  →  CV accuracy: {m*100:.2f}%")
    if m > best_acc:
        best_acc, best_c = m, c

print(f"\n✅ Best C = {best_c}  (CV accuracy: {best_acc*100:.2f}%)")

# ── TRAIN FINAL PIPELINE ──────────────────────────────────────────────────────
print("\n🤖 Training final pipeline...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000,
        min_df=2, max_df=0.95, sublinear_tf=True, stop_words="english"
    )),
    ("clf", CalibratedClassifierCV(
        LinearSVC(C=best_c, class_weight="balanced", max_iter=5000), cv=3
    ))
])
pipeline.fit(X_train, y_train)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

plot_confusion_matrix(y_test, y_pred, le.classes_, acc*100,
                      save_path="model/confusion_matrix.png")

joblib.dump(pipeline, "model/pipeline.joblib")
joblib.dump(le,       "model/label_encoder.joblib")

# Top keyword snippets per specialty → used by app.py for keyword hint chips
specialty_info = {}
for spec in le.classes_:
    kws = df[df["medical_specialty"] == spec]["keywords"].head(5).tolist()
    specialty_info[spec] = " | ".join(kws) if kws else ""

with open("model/specialty_info.json", "w") as f:
    json.dump(specialty_info, f, indent=2)

print(f"\n💾 Saved → ./model/")
print(f"✅ Done!  {df['medical_specialty'].nunique()} specialties · {acc*100:.1f}% accuracy")
print("▶  Run:  streamlit run app.py")
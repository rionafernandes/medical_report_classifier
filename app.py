"""
app.py  –  Medical Report Specialty Classifier
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Report Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0d1117; }
    .result-card {
        background: linear-gradient(135deg, #1a2332 0%, #162032 100%);
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warn-card {
        background: #2d1f0e;
        border: 1px solid #f0883e;
        border-left: 4px solid #f0883e;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: #f0883e;
        font-size: 0.88rem;
    }
    .expanded-card {
        background: #0d2a1a;
        border: 1px solid #2ea043;
        border-left: 4px solid #2ea043;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: #7ee787;
        font-size: 0.82rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .confidence-bar {
        height: 8px;
        background: #21262d;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.3rem 0;
    }
    .confidence-fill { height: 100%; border-radius: 4px; }
    .header-title { font-size: 2.2rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.2rem; }
    .header-sub { color: #8b949e; font-size: 1rem; margin-bottom: 1.5rem; }
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: white !important; border: none !important;
        border-radius: 6px !important; font-weight: 600 !important;
        padding: 0.5rem 2rem !important; transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(31,111,235,0.4) !important;
    }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Symptom Expander Dictionary ───────────────────────────────────────────────
SYMPTOM_EXPANSIONS = {
    "cough": "persistent cough dry cough productive cough respiratory tract upper airway",
    "cold": "common cold rhinitis nasal congestion runny nose sneezing upper respiratory infection",
    "cough and cold": "patient presents with cough cold runny nose nasal congestion sore throat sneezing mild fever upper respiratory tract infection rhinitis antihistamines decongestants",
    "fever": "elevated temperature febrile pyrexia chills rigors sweating infection inflammation",
    "headache": "headache migraine cephalgia pain head pressure throbbing nausea photophobia",
    "chest pain": "chest pain angina pectoris myocardial infarction cardiac ischemia ECG troponin shortness of breath",
    "back pain": "lower back pain lumbar pain spinal disc herniation musculoskeletal orthopedic",
    "stomach pain": "abdominal pain epigastric pain gastritis peptic ulcer nausea vomiting gastrointestinal",
    "knee pain": "knee pain joint pain arthritis orthopedic ligament meniscus swelling",
    "diabetes": "diabetes mellitus type 2 hyperglycemia insulin blood glucose HbA1c endocrinology",
    "hypertension": "hypertension high blood pressure cardiovascular antihypertensive medication",
    "anxiety": "anxiety disorder generalized anxiety panic attacks psychiatric mental health",
    "depression": "major depressive disorder mood disorder psychiatry antidepressants counseling",
    "skin rash": "dermatitis eczema skin rash erythema lesions dermatology pruritus",
    "urinary": "urinary tract infection dysuria frequency urgency urology kidney bladder",
    "pregnancy": "obstetrics pregnancy prenatal antenatal gestational gynecology",
    "fracture": "bone fracture orthopedic trauma X-ray cast immobilization",
    "cancer": "malignancy tumor oncology chemotherapy radiation biopsy",
    "asthma": "asthma bronchospasm wheezing shortness of breath inhaler bronchodilator pulmonary",
    "allergy": "allergic reaction hypersensitivity antihistamine rhinitis immunology",
}

# ── Example Templates ─────────────────────────────────────────────────────────
EXAMPLE_TEMPLATES = {
    "Cardiovascular": {
        "description": "A 58-year-old male presents with chest pain and shortness of breath.",
        "transcription": "Patient is a 58-year-old male with a history of hypertension presenting with acute onset chest pain radiating to the left arm, associated with diaphoresis and shortness of breath. ECG shows ST elevation in leads II, III, aVF. Troponin levels are elevated at 2.4 ng/mL. Patient was started on aspirin 325mg, heparin drip, and nitroglycerin. Cardiology was consulted for emergent catheterization. Diagnosis: Acute inferior myocardial infarction.",
        "keywords": "chest pain, myocardial infarction, ECG, troponin, cardiovascular, angina, heparin"
    },
    "Neurology": {
        "description": "Patient presents with sudden onset severe headache and confusion.",
        "transcription": "A 45-year-old female presents with sudden onset thunderclap headache, described as the worst headache of her life, associated with neck stiffness, photophobia, and mild confusion. CT head reveals subarachnoid hemorrhage. Lumbar puncture shows xanthochromia. Neurosurgery consulted for aneurysm clipping. Started on nimodipine for vasospasm prevention.",
        "keywords": "headache, subarachnoid hemorrhage, neurology, CT scan, lumbar puncture, aneurysm"
    },
    "Cough & Cold": {
        "description": "Patient with persistent cough, cold, nasal congestion and mild fever.",
        "transcription": "Patient presents with 5-day history of productive cough, nasal congestion, runny nose, sore throat, and low-grade fever of 99.8F. Lungs clear to auscultation. No wheezing. Throat mildly erythematous. No stridor. Diagnosis: Upper respiratory tract infection. Prescribed antihistamines, decongestants, and advised rest and hydration.",
        "keywords": "cough, cold, upper respiratory infection, nasal congestion, fever, antihistamine, ENT"
    },
    "Orthopedic": {
        "description": "A 32-year-old athlete with acute knee pain after sports injury.",
        "transcription": "Patient is a 32-year-old male athlete who twisted his right knee during a football game. Presents with immediate pain, swelling, and inability to bear weight. Positive anterior drawer test and Lachman test indicating ACL tear. MRI confirms complete ACL rupture with medial meniscus tear. Orthopedic surgery recommended for ACL reconstruction.",
        "keywords": "knee pain, ACL tear, orthopedic, MRI, ligament, sports injury, meniscus"
    },
    "Gastroenterology": {
        "description": "Patient with abdominal pain, nausea, and vomiting.",
        "transcription": "A 40-year-old female presents with 3-day history of epigastric pain, nausea, vomiting, and bloating. Pain worsens after meals. History of H. pylori infection 2 years ago. Endoscopy reveals gastric ulcer. Biopsy taken. Started on triple therapy: amoxicillin, clarithromycin, and proton pump inhibitor. Diagnosis: Peptic ulcer disease.",
        "keywords": "abdominal pain, peptic ulcer, gastroscopy, H. pylori, gastroenterology, endoscopy"
    },
    "Oncology": {
        "description": "Patient with unexplained weight loss and fatigue.",
        "transcription": "A 62-year-old male with 3-month history of progressive fatigue, unintentional weight loss of 15 pounds, night sweats, and enlarged lymph nodes in the neck and axilla. CBC shows lymphocytosis. CT scan reveals mediastinal lymphadenopathy. Biopsy confirms diffuse large B-cell lymphoma. Oncology consulted for CHOP chemotherapy regimen.",
        "keywords": "lymphoma, oncology, chemotherapy, weight loss, lymphadenopathy, biopsy, cancer"
    },
    "Pediatrics": {
        "description": "8-year-old child with fever, ear pain, and irritability.",
        "transcription": "An 8-year-old male presents with 2-day history of right ear pain, fever of 101.5F, irritability, and decreased hearing. Otoscopic examination reveals bulging, erythematous tympanic membrane with purulent effusion. Diagnosis: Acute otitis media. Prescribed amoxicillin for 10 days. Parents counseled on follow-up.",
        "keywords": "otitis media, ear pain, pediatrics, fever, tympanic membrane, amoxicillin, ENT"
    },
    "Obstetrics": {
        "description": "28-week pregnant female with elevated blood pressure.",
        "transcription": "A 30-year-old female at 28 weeks gestation presents with blood pressure of 158/102 mmHg, headache, visual disturbances, and 2+ proteinuria on urinalysis. Fetal heart rate is normal. Diagnosis: Preeclampsia. Started on magnesium sulfate for seizure prophylaxis and labetalol for blood pressure control.",
        "keywords": "preeclampsia, obstetrics, pregnancy, blood pressure, proteinuria, magnesium sulfate"
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipeline = joblib.load("model/pipeline.joblib")
    le       = joblib.load("model/label_encoder.joblib")
    with open("model/specialty_info.json") as f:
        info = json.load(f)
    return pipeline, le, info


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


def predict_single(pipeline, le, description="", transcription="", keywords=""):
    trans_expanded, trans_was_expanded = expand_symptoms(transcription)
    desc_expanded,  desc_was_expanded  = expand_symptoms(description)
    combined = (clean_text(desc_expanded) + " " +
                clean_text(trans_expanded) + " " +
                clean_text(keywords) + " " +
                clean_text(keywords))
    combined = combined.strip()
    if not combined:
        return None, None, None, False
    was_expanded = trans_was_expanded or desc_was_expanded
    proba        = pipeline.predict_proba([combined])[0]
    top5_idx     = np.argsort(proba)[::-1][:5]
    top5_labels  = le.inverse_transform(top5_idx)
    top5_scores  = proba[top5_idx]
    return top5_labels[0], top5_scores[0], list(zip(top5_labels, top5_scores)), was_expanded


def predict_batch(pipeline, le, df_in: pd.DataFrame) -> pd.DataFrame:
    desc  = df_in.get("description",   pd.Series([""] * len(df_in))).fillna("")
    trans = df_in.get("transcription", pd.Series([""] * len(df_in))).fillna("")
    kw    = df_in.get("keywords",      pd.Series([""] * len(df_in))).fillna("")
    combined = (desc.apply(clean_text) + " " + trans.apply(clean_text) + " " +
                kw.apply(clean_text) + " " + kw.apply(clean_text))
    preds  = pipeline.predict(combined)
    probas = pipeline.predict_proba(combined).max(axis=1)
    labels = le.inverse_transform(preds)
    out = df_in.copy()
    out["predicted_specialty"] = labels
    out["confidence"]          = (probas * 100).round(2)
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedClassifier")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Single Prediction", "📂 Bulk Classification"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("""
    - **Algorithm:** Logistic Regression  
    - **Features:** TF-IDF (1-2 grams)  
    - **Input:** description + transcription + keywords  
    - **Classes:** 40 medical specialties  
    """)
    st.markdown("---")
    st.markdown("**Confidence Guide**")
    st.markdown("""
    - 🟢 ≥ 70% — High confidence  
    - 🟡 40–69% — Moderate  
    - 🔴 < 40% — Low (add more detail)  
    """)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    pipeline, le, specialty_info = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found. Please run `python train_model.py` first.\n\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":
    st.markdown('<div class="header-title">🔍 Single Report Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Paste a medical report or load an example template to classify the specialty.</div>', unsafe_allow_html=True)

    # ── Session state for template ──
    if "tpl_desc" not in st.session_state:
        st.session_state.tpl_desc = ""
        st.session_state.tpl_trans = ""
        st.session_state.tpl_kw = ""

    # ── Example Templates ──
    with st.expander("📋 Load an Example Template", expanded=False):
        st.caption("Click any template to auto-fill the report fields below:")
        cols = st.columns(4)
        for i, name in enumerate(EXAMPLE_TEMPLATES.keys()):
            with cols[i % 4]:
                if st.button(f"🏥 {name}", key=f"tpl_{i}", use_container_width=True):
                    st.session_state.tpl_desc  = EXAMPLE_TEMPLATES[name]["description"]
                    st.session_state.tpl_trans = EXAMPLE_TEMPLATES[name]["transcription"]
                    st.session_state.tpl_kw    = EXAMPLE_TEMPLATES[name]["keywords"]
                    st.success(f"Loaded: {name}")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 📝 Enter Report Details")
        tab1, tab2, tab3 = st.tabs(["Transcription", "Description", "Keywords"])

        with tab1:
            transcription_input = st.text_area(
                "Transcription", value=st.session_state.tpl_trans, height=220,
                placeholder="Paste full transcription, or just type: 'cough and cold', 'chest pain'...",
                label_visibility="collapsed"
            )
        with tab2:
            description_input = st.text_area(
                "Description", value=st.session_state.tpl_desc, height=100,
                placeholder="e.g. A 45-year-old male presents with chest pain...",
                label_visibility="collapsed"
            )
        with tab3:
            keywords_input = st.text_area(
                "Keywords", value=st.session_state.tpl_kw, height=100,
                placeholder="e.g. cardiology, chest pain, ECG, troponin...",
                label_visibility="collapsed"
            )

        predict_btn = st.button("🚀 Classify Report", use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Prediction Result")

        if predict_btn and model_loaded:
            if not any([transcription_input.strip(), description_input.strip(), keywords_input.strip()]):
                st.warning("Please enter at least one field.")
            else:
                total_words = len((transcription_input + description_input + keywords_input).split())
                with st.spinner("Analyzing..."):
                    predicted, confidence, top5, was_expanded = predict_single(
                        pipeline, le,
                        description=description_input,
                        transcription=transcription_input,
                        keywords=keywords_input
                    )

                if predicted:
                    conf_pct = confidence * 100
                    color = "#2ea043" if conf_pct >= 70 else "#f0c000" if conf_pct >= 40 else "#f85149"
                    emoji = "🟢" if conf_pct >= 70 else "🟡" if conf_pct >= 40 else "🔴"

                    if was_expanded:
                        st.markdown("""
                        <div class="expanded-card">
                            ✨ <strong>Symptom Expander Active</strong><br>
                            Short input detected — automatically expanded to clinical context for better accuracy.
                        </div>""", unsafe_allow_html=True)

                    if conf_pct < 40:
                        st.markdown(f"""
                        <div class="warn-card">
                            ⚠️ <strong>Low Confidence ({conf_pct:.1f}%)</strong><br>
                            Input too short or vague. Add symptoms, medications, diagnosis details or use a template above.
                        </div>""", unsafe_allow_html=True)
                    elif conf_pct < 70:
                        st.markdown(f"""
                        <div class="warn-card" style="background:#2a2200;border-color:#f0c000;color:#f0c000;">
                            💡 <strong>Moderate Confidence ({conf_pct:.1f}%)</strong><br>
                            Adding more clinical details will improve prediction accuracy.
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">Predicted Specialty</div>
                        <div style="font-size:1.6rem;font-weight:700;color:#e6edf3;margin:0.4rem 0;">{predicted}</div>
                        <div style="font-size:0.9rem;color:{color};margin-bottom:0.8rem;">
                            {emoji} Confidence: <strong>{conf_pct:.1f}%</strong>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{min(conf_pct,100)}%;background:{color};"></div>
                        </div>
                        <div style="font-size:0.75rem;color:#8b949e;margin-top:0.6rem;">{total_words} words analyzed</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("#### 📊 Top 5 Candidates")
                    for i, (label, score) in enumerate(top5):
                        pct = score * 100
                        bar_color = color if i == 0 else "#388bfd"
                        st.markdown(f"""
                        <div style="margin:0.5rem 0;">
                            <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#c9d1d9;">
                                <span>{"🏆 " if i==0 else f"{i+1}. "}{label}</span>
                                <span style="color:{'#e6edf3' if i==0 else '#8b949e'}">{pct:.1f}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width:{min(pct,100)}%;background:{bar_color};"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#8b949e;">
                <div style="font-size:3rem;margin-bottom:1rem;">🩺</div>
                <div>Load a <strong>template</strong> above or enter report details,<br>then click <strong>Classify Report</strong></div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Bulk Classification
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Bulk Classification":
    st.markdown('<div class="header-title">📂 Bulk CSV Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Upload a CSV with medical records to classify all at once.</div>', unsafe_allow_html=True)

    st.info("**Expected CSV columns** (at least one required): `description` · `transcription` · `keywords`")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file and model_loaded:
        df_up = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df_up)} records | Columns: {list(df_up.columns)}")

        with st.expander("👁️ Preview data (first 5 rows)"):
            st.dataframe(df_up.head(), use_container_width=True)

        if st.button("🚀 Run Bulk Classification", use_container_width=True):
            if not any(c in df_up.columns for c in ["description", "transcription", "keywords"]):
                st.error("CSV must have at least one of: description, transcription, keywords")
            else:
                with st.spinner(f"Classifying {len(df_up)} records..."):
                    df_result = predict_batch(pipeline, le, df_up)

                st.success(f"✅ Done! Classified {len(df_result)} records.")

                high   = (df_result["confidence"] >= 70).sum()
                medium = ((df_result["confidence"] >= 40) & (df_result["confidence"] < 70)).sum()
                low    = (df_result["confidence"] < 40).sum()

                st.markdown("### 📊 Summary")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Records", len(df_result))
                c2.metric("Specialties Found", df_result["predicted_specialty"].nunique())
                c3.metric("Avg Confidence", f"{df_result['confidence'].mean():.1f}%")
                c4.metric("🟢 High Conf", high)
                c5.metric("🔴 Low Conf", low)

                if low > 0:
                    st.markdown(f"""
                    <div class="warn-card">
                        ⚠️ <strong>{low} records</strong> have low confidence (&lt;40%).
                        These may have missing or very short text. Consider enriching them for better accuracy.
                    </div>""", unsafe_allow_html=True)

                st.markdown("### 🏷️ Specialty Distribution")
                dist = df_result["predicted_specialty"].value_counts().reset_index()
                dist.columns = ["Specialty", "Count"]
                st.bar_chart(dist.set_index("Specialty")["Count"])

                st.markdown("### 📋 Results Table")
                display_cols = ["predicted_specialty", "confidence"] + \
                               [c for c in df_result.columns if c not in ["predicted_specialty", "confidence"]]
                st.dataframe(df_result[display_cols], use_container_width=True, height=400)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="classified_medical_reports.csv",
                    mime="text/csv",
                    use_container_width=True
                )

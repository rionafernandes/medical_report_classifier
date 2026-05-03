"""
app.py  —  Medical Report Specialty Classifier
Run   : streamlit run app.py
Model : LinearSVC + TF-IDF  |  23 classes  |  ~78-79% accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import json
import os
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedClassify",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
D = st.session_state.dark_mode

if D:
    BG         = "#0d1117";  SURFACE  = "#161b22";  SURFACE2 = "#1a2332"
    BORDER     = "#30363d";  TEXT     = "#e6edf3";  TEXT2    = "#8b949e"
    TEXT3      = "#c9d1d9";  ACCENT   = "#58a6ff";  BTN_BG   = "linear-gradient(135deg,#1f6feb,#388bfd)"
    INPUT_BG   = "#161b22";  TAB_BG   = "#0d1117";  TAB_ACT  = "#21262d"
    SIDEBAR_BG = "#0d1117";  CHIP_BG  = "#21262d";  CHIP_COL = "#8b949e"
    DIVIDER    = "#21262d";  BAR_BG   = "#21262d";  EMPTY_COL= "#8b949e"
    WARN_BG    = "#2d1f0e";  WARN_BOR = "#f0883e";  WARN_COL = "#f0883e"
    MOD_BG     = "#2a2200";  MOD_BOR  = "#f0c000";  MOD_COL  = "#f0c000"
    INFO_BG    = "#0d2a1a";  INFO_BOR = "#2ea043";  INFO_COL = "#7ee787"
    KW_BG      = "#161b22";  KW_BOR   = "#30363d";  KW_TITLE = "#58a6ff"
    TAG_BG     = "#1f3a5f";  TAG_COL  = "#79c0ff"
    GREEN      = "#2ea043";  YELLOW   = "#f0c000";  RED      = "#f85149"
    TOP_BAR    = "#388bfd";  RANK_COL = "#8b949e"
    SCORE_HIGH = "#e6edf3";  SCORE_LOW= "#8b949e"
else:
    BG         = "#f7f8fa";  SURFACE  = "#ffffff";  SURFACE2 = "#f0f2f5"
    BORDER     = "#e8eaed";  TEXT     = "#111318";  TEXT2    = "#6b7280"
    TEXT3      = "#374151";  ACCENT   = "#2563eb";  BTN_BG   = "#111318"
    INPUT_BG   = "#ffffff";  TAB_BG   = "#f0f2f5";  TAB_ACT  = "#ffffff"
    SIDEBAR_BG = "#ffffff";  CHIP_BG  = "#f0f2f5";  CHIP_COL = "#374151"
    DIVIDER    = "#e8eaed";  BAR_BG   = "#e8eaed";  EMPTY_COL= "#9ca3af"
    WARN_BG    = "#fffbeb";  WARN_BOR = "#fde68a";  WARN_COL = "#92400e"
    MOD_BG     = "#fef9c3";  MOD_BOR  = "#fde047";  MOD_COL  = "#713f12"
    INFO_BG    = "#eff6ff";  INFO_BOR = "#bfdbfe";  INFO_COL = "#1e40af"
    KW_BG      = "#f8fafc";  KW_BOR   = "#e8eaed";  KW_TITLE = "#2563eb"
    TAG_BG     = "#dbeafe";  TAG_COL  = "#1e40af"
    GREEN      = "#16a34a";  YELLOW   = "#ca8a04";  RED      = "#dc2626"
    TOP_BAR    = "#2563eb";  RANK_COL = "#9ca3af"
    SCORE_HIGH = "#111318";  SCORE_LOW= "#6b7280"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{{font-family:'IBM Plex Sans',sans-serif;background-color:{BG};color:{TEXT};}}
.main{{background-color:{BG};}}
.block-container{{padding:2rem 2.5rem 3rem;max-width:1280px;}}
section[data-testid="stSidebar"]{{background:{SIDEBAR_BG};border-right:1px solid {BORDER};}}
section[data-testid="stSidebar"] .block-container{{padding:2rem 1.5rem;}}
.header-title{{font-size:2rem;font-weight:700;color:{TEXT};margin-bottom:0.2rem;letter-spacing:-0.02em;}}
.header-sub{{color:{TEXT2};font-size:0.95rem;margin-bottom:1.5rem;}}
.section-label{{font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT2};margin-bottom:0.6rem;}}
.stTextArea textarea{{background-color:{INPUT_BG}!important;color:{TEXT}!important;border:1px solid {BORDER}!important;
    font-family:'IBM Plex Mono',monospace!important;font-size:0.84rem!important;border-radius:8px!important;}}
.stTextArea textarea:focus{{border-color:{ACCENT}!important;box-shadow:0 0 0 3px rgba(88,166,255,0.1)!important;}}
.stTabs [data-baseweb="tab-list"]{{gap:0;background:{TAB_BG};border-radius:8px;padding:3px;border:none;}}
.stTabs [data-baseweb="tab"]{{border-radius:6px;font-size:0.82rem;font-weight:500;color:{TEXT2};
    padding:0.35rem 1rem;border:none;background:transparent;}}
.stTabs [aria-selected="true"]{{background:{TAB_ACT}!important;color:{TEXT}!important;box-shadow:0 1px 3px rgba(0,0,0,0.15);}}
.stButton>button{{background:{BTN_BG}!important;color:#ffffff!important;border:none!important;
    border-radius:8px!important;font-weight:600!important;font-size:0.88rem!important;
    padding:0.55rem 1.75rem!important;transition:opacity 0.15s,transform 0.1s!important;}}
.stButton>button:hover{{opacity:0.88!important;transform:translateY(-1px)!important;}}
.result-card{{background:{SURFACE2};border:1px solid {BORDER};border-left:4px solid {ACCENT};
    border-radius:10px;padding:1.4rem;margin:0.5rem 0;}}
.warn-card{{background:{WARN_BG};border:1px solid {WARN_BOR};border-left:4px solid {WARN_BOR};
    border-radius:8px;padding:0.9rem 1.2rem;margin:0.5rem 0;color:{WARN_COL};font-size:0.87rem;}}
.mod-card{{background:{MOD_BG};border:1px solid {MOD_BOR};border-left:4px solid {MOD_BOR};
    border-radius:8px;padding:0.9rem 1.2rem;margin:0.5rem 0;color:{MOD_COL};font-size:0.87rem;}}
.info-card{{background:{INFO_BG};border:1px solid {INFO_BOR};border-left:4px solid {INFO_BOR};
    border-radius:8px;padding:0.9rem 1.2rem;margin:0.5rem 0;color:{INFO_COL};font-size:0.87rem;
    font-family:'IBM Plex Mono',monospace;}}
.keyword-card{{background:{KW_BG};border:1px solid {KW_BOR};border-radius:8px;
    padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.8rem;font-family:'IBM Plex Mono',monospace;}}
.conf-bar-bg{{height:7px;background:{BAR_BG};border-radius:4px;overflow:hidden;margin:0.3rem 0 0.8rem;}}
.conf-bar-fill{{height:100%;border-radius:4px;}}
.candidate-row{{display:flex;justify-content:space-between;align-items:center;
    padding:0.45rem 0;border-bottom:1px solid {DIVIDER};font-size:0.84rem;}}
.candidate-row:last-child{{border-bottom:none;}}
.brand{{font-size:1.1rem;font-weight:700;color:{TEXT};letter-spacing:-0.02em;}}
.brand-tag{{font-size:0.72rem;color:{TEXT2};}}
.spec-chip{{display:inline-block;background:{CHIP_BG};color:{CHIP_COL};border-radius:5px;
    padding:0.15rem 0.5rem;font-size:0.71rem;margin:0.15rem;font-weight:500;}}
hr{{border-color:{BORDER}!important;margin:1.2rem 0!important;}}
div[data-testid="stMetricValue"]{{font-size:1.7rem!important;font-weight:700!important;color:{ACCENT}!important;}}
div[data-testid="stMetricLabel"]{{font-size:0.72rem!important;color:{TEXT2}!important;
    font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.05em!important;}}
details summary{{font-size:0.85rem!important;color:{TEXT3}!important;}}
div[data-testid="stRadio"] label{{font-size:0.88rem!important;color:{TEXT3}!important;}}
[data-testid="stExpander"] .stButton>button{{background:{SURFACE}!important;color:{TEXT3}!important;
    border:1px solid {BORDER}!important;font-size:0.8rem!important;font-weight:500!important;
    padding:0.4rem 0.7rem!important;}}
[data-testid="stExpander"] .stButton>button:hover{{border-color:{ACCENT}!important;
    color:{ACCENT}!important;opacity:1!important;transform:none!important;}}
</style>
""", unsafe_allow_html=True)


# ── EXACT 23 CLASS NAMES (from mtsamples.csv, MIN_SAMPLES=40) ────────────────
ALL_CLASSES = [
    "Cardiovascular / Pulmonary",
    "Consult - History and Phy.",
    "Discharge Summary",
    "ENT - Otolaryngology",
    "Emergency Room Reports",
    "Gastroenterology",
    "General Medicine",
    "Hematology - Oncology",
    "Nephrology",
    "Neurology",
    "Neurosurgery",
    "Obstetrics / Gynecology",
    "Office Notes",
    "Ophthalmology",
    "Orthopedic",
    "Pain Management",
    "Pediatrics - Neonatal",
    "Podiatry",
    "Psychiatry / Psychology",
    "Radiology",
    "SOAP / Chart / Progress Notes",
    "Surgery",
    "Urology",
]

# ── SYMPTOM EXPANSIONS ────────────────────────────────────────────────────────
# Keys = common short inputs. Values = expanded clinical text matching real
# keyword patterns found in mtsamples.csv per specialty.
SYMPTOM_EXPANSIONS = {
    # Cardiovascular / Pulmonary
    "chest pain": "cardiovascular pulmonary chest pain angina pectoris coronary artery disease myocardial infarction ECG troponin stenosis ventricular catheterization angiography heparin cardiac",
    "heart attack": "cardiovascular pulmonary myocardial infarction coronary artery disease chest pain troponin ECG ventricular catheterization cardiac stenosis heparin",
    "shortness of breath": "cardiovascular pulmonary respiratory dyspnea shortness breath pulmonary embolism heart failure ventricular coronary artery disease",
    "hypertension": "cardiovascular pulmonary hypertension blood pressure coronary artery disease heart disease general medicine",

    # ENT - Otolaryngology
    "cough": "ENT otolaryngology cough nasal congestion upper respiratory infection sinusitis rhinitis tonsillectomy adenoidectomy myringotomy otitis media",
    "cold": "ENT otolaryngology common cold rhinitis nasal congestion runny nose sneezing upper respiratory infection sinusitis antihistamines decongestants",
    "cough and cold": "ENT otolaryngology cough cold nasal congestion runny nose sore throat sneezing upper respiratory infection rhinitis antihistamines decongestants sinusitis otitis",
    "ear pain": "ENT otolaryngology otitis media ear pain tube myringotomy tympanic membrane hearing loss adenoidectomy",
    "sore throat": "ENT otolaryngology sore throat tonsillectomy adenoidectomy pharyngitis nasal upper respiratory sinusitis",
    "sinus": "ENT otolaryngology sinusitis nasal cavity chronic sinus congestion otolaryngology rhinitis",
    "hearing loss": "ENT otolaryngology hearing loss otitis media tube tympanic myringotomy audiometry",

    # Gastroenterology
    "stomach pain": "gastroenterology abdominal pain epigastric colonoscopy colon endoscopy esophagus laparoscopic cholecystectomy gallbladder peptic ulcer bowel hernia",
    "abdominal pain": "gastroenterology abdominal pain colon colonoscopy laparoscopic cholecystectomy esophageal hernia bowel endoscopy gallbladder",
    "nausea vomiting": "gastroenterology nausea vomiting abdominal pain colon endoscopy esophagus bowel peptic ulcer laparoscopic",
    "colonoscopy": "gastroenterology colonoscopy colon polyp endoscopy bowel laparoscopic abdomen",

    # Neurology
    "headache": "neurology headache migraine brain cerebral temporal artery carotid nerve cervical disc spine conduction",
    "stroke": "neurology cerebral stroke brain carotid artery temporal nerve spine disc conduction loss",
    "seizure": "neurology seizure brain cerebral epilepsy EEG temporal nerve spine",
    "dizziness": "neurology dizziness vertigo brain cerebral vestibular carotid artery temporal",
    "numbness": "neurology numbness nerve cervical disc spine conduction brain cerebral carpal",

    # Neurosurgery
    "brain surgery": "neurosurgery brain craniotomy tumor shunt cerebral neurosurgery fusion discectomy anterior cervical",
    "spine surgery": "neurosurgery cervical discectomy anterior fusion lumbar spine herniated disc decompression nerve neurosurgery",
    "back surgery": "neurosurgery lumbar cervical discectomy anterior fusion herniated disc decompression nerve neurosurgery",

    # Orthopedic
    "knee pain": "orthopedic knee pain ligament meniscus fracture joint arthroscopy ACL carpal tunnel tendon spine disc cervical tourniquet",
    "back pain": "orthopedic lumbar back pain cervical disc spine fracture joint tendon tourniquet orthopedic",
    "fracture": "orthopedic fracture bone joint tendon cervical spine carpal tourniquet cast immobilization",
    "joint pain": "orthopedic joint pain knee fracture carpal tunnel tendon disc spine cervical lumbar arthritis",
    "shoulder pain": "orthopedic shoulder joint tendon fracture cervical spine arthroscopy rotator cuff",

    # Urology
    "urinary": "urology urinary bladder inguinal hernia prostate cystoscopy catheter urinary tract infection kidney stone scrotal",
    "prostate": "urology prostate PSA bladder cystoscopy catheter urinary scrotal inguinal hernia",
    "kidney stone": "urology kidney stone renal ureteral nephrology cystoscopy catheter bladder stent",
    "hematuria": "urology hematuria bladder prostate PSA cystoscopy TURBT catheter urinary scrotal kidney",

    # Nephrology
    "kidney disease": "nephrology renal kidney disease dialysis ureteral fistula catheter stent bladder laparoscopic vein abdomen pelvis",
    "renal failure": "nephrology renal failure kidney disease dialysis creatinine electrolytes vein catheter stent",

    # Obstetrics / Gynecology
    "pregnancy": "obstetrics gynecology pregnancy prenatal vaginal uterus fetal pelvic hysterectomy uterine intrauterine antepartum gestational",
    "vaginal": "obstetrics gynecology vaginal hysterectomy uterus pelvic uterine pregnancy fetal intrauterine delivery",
    "gynecology": "obstetrics gynecology vaginal uterus pelvic hysterectomy uterine pregnancy fetal ovarian",

    # Hematology - Oncology
    "cancer": "hematology oncology cancer carcinoma biopsy breast mass lymph node excision cell tumor chemotherapy radiation",
    "tumor": "hematology oncology tumor mass carcinoma biopsy lymph node excision breast cell cancer chemotherapy",
    "lymphoma": "hematology oncology lymphoma lymph node cancer biopsy carcinoma cell chemotherapy radiation",
    "biopsy": "hematology oncology biopsy carcinoma mass lymph node breast cancer cell excision tumor",

    # Psychiatry / Psychology
    "depression": "psychiatry psychology depression disorder psychiatric mental axis bipolar assessment abuse behavior",
    "anxiety": "psychiatry psychology anxiety disorder psychiatric mental health panic assessment behavior depression",
    "mental health": "psychiatry psychology psychiatric mental health disorder depression anxiety bipolar axis assessment behavior",
    "adhd": "psychiatry psychology ADHD attention deficit disorder psychiatric mental pediatrics assessment behavior",

    # Ophthalmology
    "eye pain": "ophthalmology eye lens intraocular cataract chamber anterior phacoemulsification capsular extraction",
    "vision": "ophthalmology vision lens intraocular cataract chamber anterior phacoemulsification capsular vitrectomy retina",
    "cataract": "ophthalmology cataract lens intraocular phacoemulsification chamber anterior capsular extraction",

    # Pain Management
    "pain injection": "pain management injection epidural steroid needle fluoroscopy nerve block lumbar joint",
    "epidural": "pain management epidural injection steroid nerve block lumbar fluoroscopy",
    "nerve block": "pain management nerve block injection epidural steroid fluoroscopy lumbar joint",

    # Podiatry
    "foot pain": "podiatry foot plantar metatarsal ankle joint phalanx proximal tendon osteotomy bunionectomy",
    "ankle": "podiatry ankle foot plantar metatarsal joint phalanx tendon osteotomy",

    # Pediatrics - Neonatal
    "child": "pediatrics neonatal child infant well check otitis media pulmonary respiratory chest tube heart septal congestion",
    "newborn": "pediatrics neonatal newborn infant respiratory distress NICU surfactant CPAP heart septal tachypnea",
    "baby": "pediatrics neonatal infant newborn well check respiratory otitis media heart septal congestion",

    # Radiology
    "xray": "radiology chest abdomen spine ultrasound contrast scan brain carotid artery perfusion myocardial",
    "x-ray": "radiology chest abdomen spine ultrasound contrast scan brain carotid artery perfusion",
    "mri": "radiology MRI brain spine carotid artery contrast scan cerebral perfusion abdomen",
    "ct scan": "radiology CT scan brain chest abdomen contrast carotid artery spine ultrasound perfusion",
    "ultrasound": "radiology ultrasound abdomen pelvis kidney scrotal breast carotid artery contrast scan",

    # Emergency Room
    "emergency": "emergency room reports acute fracture bleeding headache chest respiratory dental intracranial angiogram swelling",

    # General Medicine
    "diabetes": "general medicine diabetes blood glucose hypertension respiratory infection exam pressure physical",
    "fever": "general medicine fever infection respiratory blood pressure exam physical hypertension",
    "allergy": "allergy immunology allergic rhinitis antihistamine nasal respiratory general medicine ENT otolaryngology",
}

# ── EXAMPLE TEMPLATES (matching exact class names from mtsamples.csv) ─────────
EXAMPLE_TEMPLATES = {
    "Cardiovascular / Pulmonary": {
        "description": "A 58-year-old male presents with chest pain and shortness of breath.",
        "transcription": "Patient is a 58-year-old male with a history of hypertension presenting with acute onset chest pain radiating to the left arm, associated with diaphoresis and shortness of breath. ECG shows ST elevation in leads II, III, aVF. Troponin levels are elevated at 2.4 ng/mL. Patient was started on aspirin 325mg, heparin drip, and nitroglycerin. Cardiology consulted for emergent catheterization. Diagnosis: Acute inferior myocardial infarction.",
        "keywords": "cardiovascular pulmonary, chest pain, myocardial infarction, ECG, troponin, coronary artery, stenosis, ventricular, catheterization, heparin, cardiac"
    },
    "ENT - Otolaryngology": {
        "description": "Patient with productive cough, nasal congestion and sore throat.",
        "transcription": "Patient presents with 5-day history of productive cough, nasal congestion, runny nose, sore throat, and low-grade fever. Lungs clear to auscultation. Throat mildly erythematous. No stridor. Diagnosis: Upper respiratory tract infection with sinusitis. Prescribed antihistamines, decongestants, and nasal spray. Advised rest and hydration.",
        "keywords": "ENT otolaryngology, nasal congestion, otitis media, upper respiratory infection, cough, sore throat, sinusitis, antihistamine, decongestant, tonsillectomy, adenoidectomy, myringotomy"
    },
    "Neurology": {
        "description": "Patient presents with sudden onset severe headache and confusion.",
        "transcription": "A 45-year-old female presents with sudden onset thunderclap headache, described as the worst headache of her life, associated with neck stiffness, photophobia, and mild confusion. CT head reveals subarachnoid hemorrhage. Lumbar puncture shows xanthochromia. Neurosurgery consulted for aneurysm clipping. Started on nimodipine for vasospasm prevention.",
        "keywords": "neurology, headache, brain, cerebral, subarachnoid hemorrhage, CT scan, lumbar puncture, carotid artery, temporal, nerve, conduction, aneurysm"
    },
    "Neurosurgery": {
        "description": "Patient with cervical disc herniation requiring surgical decompression.",
        "transcription": "A 48-year-old male with 6-month history of cervical radiculopathy, progressive arm weakness, and numbness. MRI confirms C5-C6 disc herniation with cord compression. Patient underwent anterior cervical discectomy and fusion (ACDF). Intraoperative neuromonitoring used throughout. No complications. Patient discharged on postoperative day 2.",
        "keywords": "neurosurgery, cervical, discectomy, anterior, fusion, lumbar, herniated disc, decompression, nerve, craniotomy, nucleus pulposus, spine"
    },
    "Orthopedic": {
        "description": "A 32-year-old athlete with acute knee pain after sports injury.",
        "transcription": "Patient is a 32-year-old male athlete who twisted his right knee during a football game. Presents with immediate pain, swelling, and inability to bear weight. Positive anterior drawer test and Lachman test indicating ACL tear. MRI confirms complete ACL rupture with medial meniscus tear. Orthopedic surgery recommended for ACL reconstruction.",
        "keywords": "orthopedic, knee, fracture, carpal tunnel, cervical, tendon, joint, ligament, ACL, meniscus, spine, tourniquet, disc, sports injury"
    },
    "Gastroenterology": {
        "description": "Patient with abdominal pain, nausea, and vomiting.",
        "transcription": "A 40-year-old female presents with 3-day history of epigastric pain, nausea, vomiting, and bloating. Pain worsens after meals. History of H. pylori infection 2 years ago. Endoscopy reveals gastric ulcer. Biopsy taken. Started on triple therapy: amoxicillin, clarithromycin, and proton pump inhibitor. Diagnosis: Peptic ulcer disease.",
        "keywords": "gastroenterology, colon, laparoscopic, colonoscopy, abdomen, cholecystectomy, abdominal pain, esophagus, hernia, endoscopy, bowel, gallbladder"
    },
    "Hematology - Oncology": {
        "description": "Patient with unexplained weight loss, fatigue, and enlarged lymph nodes.",
        "transcription": "A 62-year-old male with 3-month history of progressive fatigue, unintentional weight loss, night sweats, and enlarged lymph nodes in the neck and axilla. CBC shows lymphocytosis. CT scan reveals mediastinal lymphadenopathy. Biopsy confirms diffuse large B-cell lymphoma. Oncology consulted for CHOP chemotherapy regimen.",
        "keywords": "hematology oncology, carcinoma, biopsy, breast, mass, cancer, cell, lymph node, excision, tumor, chemotherapy, radiation, leukemia, lymphoma"
    },
    "Nephrology": {
        "description": "Patient with chronic kidney disease and declining renal function.",
        "transcription": "A 64-year-old male with known CKD stage 3 presents with worsening creatinine levels of 3.8 mg/dL, reduced urine output, and bilateral leg edema. Renal ultrasound shows bilateral cortical thinning. Nephrology consulted for potential dialysis planning. AV fistula creation discussed. Electrolytes managed and dietary modifications advised.",
        "keywords": "nephrology, renal, abdomen, pelvis, vein, kidney, stone, ureteral, fistula, catheter, disease, stent, bladder, laparoscopic"
    },
    "Obstetrics / Gynecology": {
        "description": "28-week pregnant female with elevated blood pressure.",
        "transcription": "A 30-year-old female at 28 weeks gestation presents with blood pressure of 158/102 mmHg, headache, visual disturbances, and 2+ proteinuria on urinalysis. Fetal heart rate is normal. Diagnosis: Preeclampsia. Started on magnesium sulfate for seizure prophylaxis and labetalol for blood pressure control.",
        "keywords": "obstetrics gynecology, vaginal, uterus, pregnancy, pelvic, hysterectomy, fetal, uterine, intrauterine, antepartum, proteinuria, magnesium sulfate, gravida"
    },
    "Ophthalmology": {
        "description": "Patient with reduced vision following cataract surgery.",
        "transcription": "A 72-year-old female presents with reduced vision in the right eye following cataract extraction 3 months ago. Slit-lamp examination reveals posterior capsule opacification. Nd:YAG laser capsulotomy performed under topical anaesthesia. Intraocular pressure normal at 14 mmHg. Vision improved to 20/30 post-procedure.",
        "keywords": "ophthalmology, lens, intraocular, cataract, anterior chamber, phacoemulsification, capsular, extraction, vitrectomy, speculum, retina, eye"
    },
    "Urology": {
        "description": "Male patient with hematuria and lower urinary tract symptoms.",
        "transcription": "A 65-year-old male presents with 3-week history of gross hematuria, urinary frequency, urgency, and weak stream. PSA elevated at 8.2 ng/mL. Digital rectal exam reveals enlarged prostate. CT urogram shows bladder mass. Cystoscopy performed — biopsy confirms transitional cell carcinoma. Referred for TURBT.",
        "keywords": "urology, bladder, inguinal hernia, prostate, cystoscopy, catheter, urinary tract, PSA, TURBT, scrotal, repair, hematuria"
    },
    "Radiology": {
        "description": "CT scan interpretation for chest mass evaluation.",
        "transcription": "CT chest with contrast performed for evaluation of right upper lobe mass noted on prior chest X-ray. Findings reveal a 2.3 cm spiculated mass in the right upper lobe with mediastinal lymphadenopathy. No pleural effusion. Impression: Findings highly suspicious for primary lung malignancy. Recommend PET scan and biopsy.",
        "keywords": "radiology, artery, brain, carotid, spine, ultrasound, perfusion, abdomen, contrast, scan, chest, myocardial, CT, MRI, X-ray"
    },
    "Surgery": {
        "description": "Patient requires emergency appendectomy for acute appendicitis.",
        "transcription": "A 25-year-old male presents with 24-hour history of right lower quadrant pain, fever of 101.2F, nausea, and rebound tenderness. WBC elevated at 14,000. CT abdomen confirms acute appendicitis with no perforation. Patient taken to OR for laparoscopic appendectomy. Procedure completed without complications. Discharged POD2.",
        "keywords": "surgery, laparoscopic, appendectomy, coronary artery, catheter, anterior, cervical, carpal tunnel, biopsy, hernia, incision, appendicitis"
    },
    "Pain Management": {
        "description": "Patient requiring epidural steroid injection for lumbar radiculopathy.",
        "transcription": "A 52-year-old female with 4-month history of left leg radiculopathy and lower back pain radiating to L4-L5 distribution. Conservative therapy failed. Fluoroscopy-guided lumbar epidural steroid injection performed at L4-L5 interspace with 80mg triamcinolone. Patient tolerated procedure well. Follow-up in 2 weeks.",
        "keywords": "pain management, injection, epidural, steroid, needle, fluoroscopy, nerve block, lumbar, joint, trigger point, block"
    },
    "Psychiatry / Psychology": {
        "description": "Patient with major depressive disorder and anxiety.",
        "transcription": "A 34-year-old female presents with 6-month history of persistent low mood, anhedonia, insomnia, poor appetite, and anxiety. PHQ-9 score of 19 (severe). No psychosis. Beck Depression Inventory completed. Axis I: Major Depressive Disorder, severe. Axis II: deferred. Started on sertraline 50mg and referred for CBT. Safety plan established.",
        "keywords": "psychiatry psychology, disorder, psychiatric, depression, axis, mental health, Beck inventory, bipolar, abuse, psychological assessment, consultation, behavior"
    },
    "Pediatrics - Neonatal": {
        "description": "Well-child check for a 1-month-old infant.",
        "transcription": "A 1-month-old male presents for routine well-child check. Birth weight 3.4 kg, current weight 4.2 kg. Feeding well — breastfed every 2-3 hours. No respiratory distress. Heart sounds normal, no murmur detected. Developmental milestones appropriate. Otitis media screening negative. Immunizations given per schedule. Parents counseled on safe sleep.",
        "keywords": "pediatrics neonatal, child, well check, otitis media, pulmonary, respiratory, chest tube, heart, septal, infant, congestion, newborn"
    },
    "Podiatry": {
        "description": "Patient with bunion deformity and forefoot pain.",
        "transcription": "A 58-year-old female presents with painful bunion deformity of the right first metatarsophalangeal joint with associated hallux valgus. Conservative management failed. Patient underwent Austin-Youngswick bunionectomy with screw fixation. Osteotomy of the first metatarsal performed. Proximal phalanx realigned. Weight-bearing in postop shoe at 2 weeks.",
        "keywords": "podiatry, foot, plantar, metatarsal, ankle, joint, phalanx, proximal, tendon, osteotomy, bunionectomy, interphalangeal, distal"
    },
    "Discharge Summary": {
        "description": "Discharge summary for patient admitted with acute respiratory failure.",
        "transcription": "Patient is a 74-year-old male admitted with acute-on-chronic respiratory failure secondary to COPD exacerbation. Hospital course included nebulized bronchodilators, systemic corticosteroids, and non-invasive ventilation. Responded well to therapy. Discharge summary dictated. Chronic heart failure and renal disease managed. Referred for speech therapy and pulmonary rehab. Discharged home with oxygen.",
        "keywords": "discharge summary, respiratory, renal, failure, chronic, heart, pulmonary, therapy, anemia, coronary, hysterectomy, disease, speech"
    },
    "Emergency Room Reports": {
        "description": "Patient presenting to emergency room with acute headache and vomiting.",
        "transcription": "A 38-year-old male presents to the emergency department with sudden severe headache, nausea, vomiting, and photophobia. BP 178/102. CT head shows no hemorrhage. Lumbar puncture performed — CSF clear. Treated with IV analgesia and antiemetics. Neurology consulted. Diagnosis: Severe migraine. Discharged with outpatient neurology follow-up.",
        "keywords": "emergency room reports, acute, fracture, bleeding, headache, chest, respiratory, dental, intracranial, angiogram, swelling, nephrostomy"
    },
    "General Medicine": {
        "description": "Annual physical examination with multiple chronic conditions.",
        "transcription": "A 67-year-old female presents for annual physical examination. History of hypertension, type 2 diabetes, and hypothyroidism. Blood pressure 138/84. Blood glucose 142 mg/dL. HbA1c 7.4%. Physical exam unremarkable. Medications reviewed. Referred for mammogram and colonoscopy screening. Flu and pneumococcal vaccines administered. Follow-up in 3 months.",
        "keywords": "general medicine, respiratory, blood, disease, exam, pain, infection, physical, hypertension, diabetes, pressure, heart, cough"
    },
    "Consult - History and Phy.": {
        "description": "Consultation for new patient with complex medical history.",
        "transcription": "Patient referred for consultation. A 55-year-old male with history of chronic back pain, hypertension, diabetes mellitus type 2, and obesity. BMI 38. Presenting concern is worsening lumbar pain and difficulty ambulating. Detailed history and physical examination completed. Labs reviewed. Dietary counseling provided. Plan includes further imaging and specialist referral.",
        "keywords": "consult, weight, loss, pain, disease, blood, lumbar, cancer, physical, consultation, bypass, dietary, gastric, diabetes, infection"
    },
    "SOAP / Chart / Progress Notes": {
        "description": "Progress note for patient with diabetes and pulmonary disease.",
        "transcription": "S: Patient reports stable blood sugars. Mild shortness of breath at rest. No chest pain. O: BP 132/78, HR 84, SpO2 94% on 2L. Chest clear. A: Type 2 diabetes mellitus, stable. COPD with mild exacerbation. Atrial fibrillation — rate controlled. P: Continue current medications. Increase diuretic dose. Follow-up in 1 week. Dietary counselling reinforced.",
        "keywords": "SOAP chart progress notes, blood, diabetes, pulmonary, disease, weight, atrial, chest, dietary, chronic, respiratory"
    },
    "Office Notes": {
        "description": "Office visit note for routine follow-up.",
        "transcription": "Patient presents for routine follow-up visit. No acute complaints. Physical examination performed — respiratory, cardiovascular, and abdominal systems reviewed. Throat clear, no erythema. Nose clear. All systems stable. Labs reviewed and within normal limits. Medications continued. Patient advised to return in 3 months or sooner if symptoms develop.",
        "keywords": "office notes, physical exam, respiratory, nose, throat, male, medical transcription, review systems, follow-up"
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
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
    # Exact match
    if text_lower in SYMPTOM_EXPANSIONS:
        return SYMPTOM_EXPANSIONS[text_lower], True
    # Partial match
    for key, expansion in SYMPTOM_EXPANSIONS.items():
        if key in text_lower or text_lower in key:
            return expansion + " " + text, True
    # Very short free text — boost it
    if len(text.split()) <= 3:
        return (text + " ") * 3 + "patient presents with symptoms diagnosis treatment", True
    return text, False


def predict_single(pipeline, le, description="", transcription="", keywords=""):
    trans_exp, trans_flag = expand_symptoms(transcription)
    desc_exp,  desc_flag  = expand_symptoms(description)
    kw_c    = clean_text(keywords)
    desc_c  = clean_text(desc_exp)
    trans_c = clean_text(trans_exp)
    # Mirror train_model.py weighting: keywords×5, description×2, transcription×1
    combined = " ".join([kw_c]*5 + [desc_c]*2 + [trans_c]).strip()
    if not combined:
        return None, None, None, False
    proba    = pipeline.predict_proba([combined])[0]
    top5_idx = np.argsort(proba)[::-1][:5]
    return (le.inverse_transform(top5_idx)[0],
            proba[top5_idx][0],
            list(zip(le.inverse_transform(top5_idx), proba[top5_idx])),
            trans_flag or desc_flag)


def predict_batch(pipeline, le, df_in: pd.DataFrame) -> pd.DataFrame:
    desc  = df_in.get("description",   pd.Series([""] * len(df_in))).fillna("")
    trans = df_in.get("transcription", pd.Series([""] * len(df_in))).fillna("")
    kw    = df_in.get("keywords",      pd.Series([""] * len(df_in))).fillna("")
    kw_c  = kw.apply(clean_text)
    dc    = desc.apply(clean_text)
    tc    = trans.apply(clean_text)
    combined = kw_c+" "+kw_c+" "+kw_c+" "+kw_c+" "+kw_c+" "+dc+" "+dc+" "+tc
    preds  = pipeline.predict(combined)
    probas = pipeline.predict_proba(combined).max(axis=1)
    out = df_in.copy()
    out["predicted_specialty"] = le.inverse_transform(preds)
    out["confidence"]          = (probas * 100).round(2)
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    col_brand, col_toggle = st.columns([3, 1])
    with col_brand:
        st.markdown('<div class="brand">🏥 MedClassify</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-tag">Medical Report Classifier · v2.0</div>', unsafe_allow_html=True)
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("☀️" if D else "🌙", key="theme_btn", help="Toggle light/dark mode"):
            st.session_state.dark_mode = not D
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🔍 Single Prediction", "📂 Bulk Classification", "📊 Confusion Matrix"],
        label_visibility="collapsed"
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
    n_classes = "23"
    if os.path.exists("model/label_encoder.joblib"):
        try:
            _le = joblib.load("model/label_encoder.joblib")
            n_classes = str(len(_le.classes_))
        except Exception:
            pass

    st.markdown(f"""
    <div style="font-size:0.82rem;color:{TEXT3};line-height:2.1;">
        Algorithm &nbsp;·&nbsp; <span style="color:{TEXT};font-weight:600;">LinearSVC (Calibrated)</span><br>
        Features &nbsp;&nbsp;·&nbsp; <span style="color:{TEXT};font-weight:600;">TF-IDF 1-2gram 50k</span><br>
        Weighting &nbsp;·&nbsp; <span style="color:{TEXT};font-weight:600;">KW×5 · Desc×2 · Trans×1</span><br>
        Classes &nbsp;&nbsp;&nbsp;·&nbsp; <span style="color:{TEXT};font-weight:600;">{n_classes} specialties</span><br>
        Dataset &nbsp;&nbsp;&nbsp;·&nbsp; <span style="color:{TEXT};font-weight:600;">mtsamples.csv (4999)</span><br>
        Accuracy &nbsp;&nbsp;·&nbsp; <span style="color:{TEXT};font-weight:600;">~78-79%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">23 Supported Specialties</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="spec-chip">{c}</span>' for c in ALL_CLASSES)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Confidence Guide</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem;line-height:2.3;color:{TEXT3};">
        <span style="color:{GREEN};font-weight:700;">●</span>&nbsp; ≥ 70% &nbsp;High confidence<br>
        <span style="color:{YELLOW};font-weight:700;">●</span>&nbsp; 40–69% &nbsp;Moderate<br>
        <span style="color:{RED};font-weight:700;">●</span>&nbsp; &lt; 40% &nbsp;Low — add more detail
    </div>
    """, unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
try:
    pipeline, le, specialty_info = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found. Run `python train_model.py` first.\n\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":
    st.markdown('<div class="header-title">🔍 Single Report Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Paste a medical report or load an example template to classify the specialty.</div>', unsafe_allow_html=True)

    if "tpl_desc" not in st.session_state:
        st.session_state.tpl_desc = st.session_state.tpl_trans = st.session_state.tpl_kw = ""

    with st.expander("📋 Load an Example Template", expanded=False):
        st.caption("Click a template to auto-fill the fields below:")
        cols = st.columns(4)
        for i, name in enumerate(EXAMPLE_TEMPLATES.keys()):
            with cols[i % 4]:
                if st.button(f"🏥 {name}", key=f"tpl_{i}", width="stretch"):
                    st.session_state.tpl_desc  = EXAMPLE_TEMPLATES[name]["description"]
                    st.session_state.tpl_trans = EXAMPLE_TEMPLATES[name]["transcription"]
                    st.session_state.tpl_kw    = EXAMPLE_TEMPLATES[name]["keywords"]
                    st.rerun()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 📝 Enter Report Details")
        tab1, tab2, tab3 = st.tabs(["Transcription", "Description", "Keywords"])
        with tab1:
            transcription_input = st.text_area(
                "Transcription", value=st.session_state.tpl_trans, height=220,
                placeholder="Paste full transcription, or type a symptom: 'cough and cold', 'chest pain', 'knee pain'…",
                label_visibility="collapsed"
            )
        with tab2:
            description_input = st.text_area(
                "Description", value=st.session_state.tpl_desc, height=100,
                placeholder="e.g. A 45-year-old male presents with chest pain…",
                label_visibility="collapsed"
            )
        with tab3:
            keywords_input = st.text_area(
                "Keywords", value=st.session_state.tpl_kw, height=100,
                placeholder="e.g. cardiovascular pulmonary, chest pain, ECG, troponin, coronary artery…",
                label_visibility="collapsed"
            )
        st.caption("💡 Keywords carry the most weight — applied 5× at inference. Use specialty name in keywords for best results.")
        predict_btn = st.button("🚀 Classify Report", width="stretch")

    with col2:
        st.markdown("#### 🎯 Prediction Result")

        if predict_btn and model_loaded:
            if not any([transcription_input.strip(), description_input.strip(), keywords_input.strip()]):
                st.warning("Please enter at least one field.")
            else:
                total_words = len((transcription_input + description_input + keywords_input).split())
                with st.spinner("Analyzing…"):
                    predicted, confidence, top5, was_expanded = predict_single(
                        pipeline, le,
                        description=description_input,
                        transcription=transcription_input,
                        keywords=keywords_input
                    )

                if predicted:
                    conf_pct = confidence * 100
                    if conf_pct >= 70:
                        color, emoji = GREEN, "🟢"
                    elif conf_pct >= 40:
                        color, emoji = YELLOW, "🟡"
                    else:
                        color, emoji = RED, "🔴"

                    if was_expanded:
                        st.markdown(
                            '<div class="info-card">✨ <strong>Symptom Expander Active</strong><br>'
                            'Short input detected — expanded to clinical context for better accuracy.</div>',
                            unsafe_allow_html=True
                        )
                    if conf_pct < 40:
                        st.markdown(
                            f'<div class="warn-card">⚠️ <strong>Low Confidence ({conf_pct:.1f}%)</strong><br>'
                            'Add symptoms, medications, or diagnosis detail — or include the specialty name in Keywords.</div>',
                            unsafe_allow_html=True
                        )
                    elif conf_pct < 70:
                        st.markdown(
                            f'<div class="mod-card">💡 <strong>Moderate Confidence ({conf_pct:.1f}%)</strong><br>'
                            'More clinical detail will improve accuracy.</div>',
                            unsafe_allow_html=True
                        )

                    # Main result card
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:0.72rem;color:{TEXT2};text-transform:uppercase;letter-spacing:1px;">Predicted Specialty</div>
                        <div style="font-size:1.5rem;font-weight:700;color:{TEXT};margin:0.35rem 0 0.2rem;letter-spacing:-0.01em;">{predicted}</div>
                        <div style="font-size:0.88rem;color:{color};margin-bottom:0.5rem;">
                            {emoji} Confidence: <strong>{conf_pct:.1f}%</strong>
                        </div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{min(conf_pct,100):.1f}%;background:{color};"></div>
                        </div>
                        <div style="font-size:0.73rem;color:{TEXT2};">{total_words} words · LinearSVC · TF-IDF 50k · 23 classes</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Keyword hints from specialty_info.json
                    raw_hint = specialty_info.get(predicted, "")
                    if raw_hint:
                        stop = {"the","a","an","and","or","of","in","to","for","with","on","at","is",
                                "are","was","were","be","been","by","from","as","it","its","this","that"}
                        seen, tags = set(), []
                        for seg in raw_hint.split("|"):
                            for w in re.sub(r"[^a-z0-9 ]", " ", seg.lower()).split():
                                if len(w) > 3 and w not in stop and w not in seen:
                                    seen.add(w); tags.append(w)
                                if len(tags) == 14:
                                    break
                            if len(tags) == 14:
                                break
                        if tags:
                            tag_html = " ".join(
                                f'<span style="display:inline-block;background:{TAG_BG};color:{TAG_COL};'
                                f'border-radius:4px;padding:1px 7px;margin:2px 2px;font-size:0.77rem;">{t}</span>'
                                for t in tags
                            )
                            st.markdown(f"""
                            <div class="keyword-card">
                                <span style="color:{KW_TITLE};font-weight:600;font-size:0.8rem;">
                                    🔑 Typical keywords for {predicted}
                                </span><br>
                                <div style="margin-top:0.4rem;">{tag_html}</div>
                            </div>""", unsafe_allow_html=True)

                    # Top 5 candidates
                    st.markdown(f"<div style='margin-top:0.8rem;margin-bottom:0.3rem;font-size:0.9rem;font-weight:600;color:{TEXT};'>📊 Top 5 Candidates</div>", unsafe_allow_html=True)
                    for i, (label, score) in enumerate(top5):
                        pct       = score * 100
                        bar_color = color if i == 0 else TOP_BAR
                        rank_str  = "🏆" if i == 0 else str(i + 1)
                        lbl_color = SCORE_HIGH if i == 0 else TEXT3
                        pct_color = SCORE_HIGH if i == 0 else SCORE_LOW
                        st.markdown(f"""
                        <div class="candidate-row">
                            <span style="color:{RANK_COL};font-size:0.75rem;width:22px;">{rank_str}</span>
                            <span style="color:{lbl_color};flex:1;margin:0 0.5rem;font-weight:{'600' if i==0 else '400'};">{label}</span>
                            <span style="color:{pct_color};font-family:'IBM Plex Mono',monospace;font-size:0.82rem;">{pct:.1f}%</span>
                        </div>
                        <div class="conf-bar-bg" style="margin:0 0 0.25rem;">
                            <div class="conf-bar-fill" style="width:{min(pct,100):.1f}%;background:{bar_color};"></div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem 1rem;color:{EMPTY_COL};">
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
        st.markdown(f"**Loaded:** {len(df_up)} records | Columns: `{list(df_up.columns)}`")

        with st.expander("👁️ Preview data (first 5 rows)"):
            st.dataframe(df_up.head(), width="stretch")

        if st.button("🚀 Run Bulk Classification", width="stretch"):
            if not any(c in df_up.columns for c in ["description", "transcription", "keywords"]):
                st.error("CSV must have at least one of: description, transcription, keywords")
            else:
                with st.spinner(f"Classifying {len(df_up)} records…"):
                    df_result = predict_batch(pipeline, le, df_up)

                st.success(f"✅ Done! Classified {len(df_result)} records.")
                high   = (df_result["confidence"] >= 70).sum()
                medium = ((df_result["confidence"] >= 40) & (df_result["confidence"] < 70)).sum()
                low    = (df_result["confidence"] < 40).sum()

                st.markdown("### 📊 Summary")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Total Records",  len(df_result))
                c2.metric("Specialties",    df_result["predicted_specialty"].nunique())
                c3.metric("Avg Confidence", f"{df_result['confidence'].mean():.1f}%")
                c4.metric("🟢 High",        high)
                c5.metric("🟡 Moderate",    medium)
                c6.metric("🔴 Low",         low)

                if low > 0:
                    st.markdown(f'<div class="warn-card">⚠️ <strong>{low} records</strong> have low confidence (&lt;40%). Consider enriching short or vague text fields.</div>', unsafe_allow_html=True)

                st.markdown("### 🏷️ Specialty Distribution")
                dist = df_result["predicted_specialty"].value_counts().reset_index()
                dist.columns = ["Specialty", "Count"]
                st.bar_chart(dist.set_index("Specialty")["Count"])

                st.markdown("### 📋 Results Table")
                display_cols = ["predicted_specialty", "confidence"] + \
                               [c for c in df_result.columns if c not in ["predicted_specialty","confidence"]]
                st.dataframe(df_result[display_cols], width="stretch", height=400)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="classified_medical_reports.csv",
                    mime="text/csv",
                    width="stretch"
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Confusion Matrix":
    st.markdown('<div class="header-title">📊 Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Per-class recall heatmap generated after the last training run.</div>', unsafe_allow_html=True)

    cm_path = "model/confusion_matrix.png"
    if os.path.exists(cm_path):
        img = Image.open(cm_path)
        st.image(img, width="stretch",
                 caption="Row-normalised recall % across 23 specialties. Diagonal cells ≥ 70% are highlighted with teal border.")

        st.markdown("#### 🔍 How to read this matrix")
        st.markdown("""
        - Each **row** = a true specialty; each **column** = what the model predicted.
        - Each cell shows **recall %** — how often that true class was correctly identified.
        - **Diagonal cells** are correct predictions — higher is better.
        - Cells with a **teal border** indicate ≥ 70% recall for that specialty.
        - **Off-diagonal cells** reveal which specialties are most commonly confused with each other.
        - Specialties with fewer samples (e.g. Podiatry=47, Office Notes=51) will naturally have lower recall.
        """)

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            with open(cm_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PNG",
                    data=f,
                    file_name="confusion_matrix.png",
                    mime="image/png",
                    width="stretch"
                )
    else:
        st.warning("No confusion matrix found at `model/confusion_matrix.png`.")
        st.markdown("Run `python train_model.py` to train the model and generate the matrix.")
        st.markdown(f"""
        <div style="text-align:center;padding:4rem 1rem;color:{EMPTY_COL};">
            <div style="font-size:3rem;margin-bottom:1rem;">📉</div>
            <div>Train the model first to view the confusion matrix.</div>
        </div>""", unsafe_allow_html=True)
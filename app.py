import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# Hugging Face Spaces runs this file from /app/app.py. Locally, the file may also
# be run from the project root. These paths make imports and model loading robust.
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add both project root and src to Python path. This is needed for:
# 1) importing src.nlp_explainer
# 2) loading the joblib model, which references functions from src/modeling.py
for path in [PROJECT_ROOT, SRC_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from src.nlp_explainer import generate_explanation, get_nlp_mode
except Exception:
    try:
        from nlp_explainer import generate_explanation, get_nlp_mode
    except Exception:
        def get_nlp_mode() -> str:
            return "Rule-based fallback explanation"

        def generate_explanation(profile: dict, prediction_chf: float, metrics: dict | None = None) -> str:
            skills = str(profile.get("skills", "")).replace(";", ", ")
            return (
                f"Die Prognose von ungefähr CHF {prediction_chf:,.0f} basiert auf Rolle, Erfahrung, Standort, "
                f"Ausbildung und Skills ({skills}). Besonders relevante Faktoren sind Berufserfahrung, technische "
                "Skills wie Python/SQL/Cloud sowie der Schweizer Standort. Die Schätzung ist keine Garantie, sondern "
                "eine datenbasierte Orientierung; Unsicherheiten entstehen durch Datenqualität, regionale Unterschiede "
                "und unterrepräsentierte Rollen."
            )

st.set_page_config(page_title="Swiss Salary Intelligence Assistant", page_icon="💼", layout="centered")

st.title("Swiss Salary Intelligence Assistant")
st.write("ML-Gehaltsprognose kombiniert mit OpenAI-gestützter NLP-Erklärung für Tech-Profile in der Schweiz.")
st.caption(f"NLP mode: {get_nlp_mode()}. If no OpenAI key is configured, the app automatically uses a deterministic fallback.")

model_path = PROJECT_ROOT / "models" / "salary_model.joblib"
metrics_path = PROJECT_ROOT / "models" / "metrics.json"

if not SRC_DIR.exists():
    st.error("Der Ordner `src/` fehlt. Bitte den vollständigen Projektordner auf Hugging Face hochladen, nicht nur `app.py`.")
    st.stop()

if not model_path.exists():
    st.error("Kein Modell gefunden. Bitte den Ordner `models/` inklusive `salary_model.joblib` hochladen.")
    st.stop()

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data
def load_metrics(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

try:
    model = load_model(model_path)
except Exception as exc:
    st.error("Das Modell konnte nicht geladen werden. Prüfe, ob `src/modeling.py` und `models/salary_model.joblib` im Space vorhanden sind.")
    st.exception(exc)
    st.stop()

metrics = load_metrics(metrics_path)

with st.form("profile"):
    job_title = st.selectbox("Job title", ["Data Scientist", "Data Engineer", "Software Developer", "Business Analyst", "ML Engineer", "IT Consultant", "Other"])
    years_experience = st.slider("Jahre Berufserfahrung", 0, 30, 3)
    education = st.selectbox("Ausbildung", ["Apprenticeship", "Bachelor", "Master", "PhD", "Other"])
    location = st.selectbox("Standort", ["Zurich", "Bern", "Basel", "Geneva", "Remote CH", "Other CH"])
    skills = st.multiselect(
        "Skills",
        ["python", "sql", "cloud", "machine learning", "docker", "power bi", "java", "project management", "spark", "nlp"],
        default=["python", "sql"],
    )
    submitted = st.form_submit_button("Gehalt schätzen und erklären")

if submitted:
    profile = {
        "job_title": job_title,
        "years_experience": years_experience,
        "education": education,
        "location": location,
        "skills": ";".join(skills),
    }
    prediction = float(model.predict(pd.DataFrame([profile]))[0])
    st.metric("Geschätztes Jahresgehalt", f"CHF {prediction:,.0f}".replace(",", "'"))
    lower, upper = prediction * 0.88, prediction * 1.12
    st.write(f"Orientierende Spanne: **CHF {lower:,.0f} – CHF {upper:,.0f}**".replace(",", "'"))
    st.subheader("NLP-Erklärung")
    with st.spinner("Erklärung wird generiert..."):
        st.write(generate_explanation(profile, prediction, metrics))
    with st.expander("Modellmetriken"):
        st.json(metrics)

import os
from textwrap import dedent

SYSTEM_MESSAGE = """You are a careful Swiss salary advisor for an AI Applications student project.
Explain machine-learning salary predictions in German.
Never claim that a salary prediction is guaranteed.
Mention uncertainty and possible data limitations.
Keep the answer concise, practical and helpful."""


def build_prompt(profile: dict, prediction_chf: float, metrics: dict | None = None) -> str:
    """Build the structured prompt that is sent to the LLM.

    The prompt connects the ML Numeric Data block with the NLP block: it contains
    the model prediction, the user profile and model-quality information.
    """
    skills = profile.get("skills", "")
    metrics_text = metrics if metrics else "nicht verfügbar"
    return dedent(f"""
    Erkläre die folgende ML-Gehaltsprognose für ein Tech-Profil in der Schweiz.

    Profil:
    - Job title: {profile.get('job_title')}
    - Erfahrung: {profile.get('years_experience')} Jahre
    - Ausbildung: {profile.get('education')}
    - Standort: {profile.get('location')}
    - Skills: {skills}

    ML-Prognose:
    - Erwartetes Jahresgehalt: CHF {prediction_chf:,.0f}

    Modell-Metriken, falls vorhanden:
    {metrics_text}

    Schreibe die Antwort auf Deutsch und strukturiere sie in drei kurze Abschnitte:
    1. Warum diese Prognose plausibel ist
    2. Welche Skills oder Faktoren den Lohn beeinflussen
    3. Was die Person verbessern könnte und welche Unsicherheiten bestehen

    Regeln:
    - Keine erfundenen Fakten oder Garantien.
    - Keine zu langen Absätze.
    - Erkläre die Prognose verständlich für eine nicht-technische Person.
    - Weise darauf hin, dass das Ergebnis eine datenbasierte Schätzung ist.
    """).strip()


def rule_based_explanation(profile: dict, prediction_chf: float) -> str:
    """Deterministic fallback explanation if no OpenAI key is available.

    This keeps the Hugging Face deployment usable even without API access.
    """
    skills = str(profile.get("skills", "")).lower()
    positives = []
    if "cloud" in skills:
        positives.append("Cloud-Kenntnisse")
    if "machine learning" in skills or "ml" in skills:
        positives.append("Machine-Learning-Skills")
    if "sql" in skills:
        positives.append("SQL")
    if "python" in skills:
        positives.append("Python")
    if "docker" in skills:
        positives.append("Docker")
    if profile.get("location") in ["Zurich", "Geneva", "Basel"]:
        positives.append(f"Standort {profile.get('location')}")

    main = ", ".join(positives[:4]) if positives else "Berufserfahrung, Rolle, Ausbildung und Standort"
    return (
        f"**Warum diese Prognose plausibel ist:** Die geschätzte Lohnspanne liegt bei ungefähr "
        f"CHF {prediction_chf:,.0f} pro Jahr. Die Schätzung basiert auf Rolle, Erfahrung, Standort, "
        f"Ausbildung und Skills. Wichtige positive Faktoren sind {main}.\n\n"
        "**Einflussfaktoren:** Technische Skills wie Python, SQL, Cloud, Docker oder Machine Learning "
        "können die Prognose positiv beeinflussen, besonders in daten- und cloudnahen Rollen. "
        "Mehr Berufserfahrung erhöht die erwartete Lohnspanne typischerweise ebenfalls.\n\n"
        "**Empfehlung und Unsicherheit:** Für eine bessere Positionierung wären vertiefte Cloud-, "
        "Data-Engineering- oder ML-Kompetenzen sinnvoll. Die Aussage ist keine Garantie, sondern eine "
        "datenbasierte Orientierung. Unsicherheiten entstehen durch Selbstangaben, regionale Unterschiede "
        "und unterrepräsentierte Rollen im Trainingsdatensatz."
    )


def generate_openai_explanation(profile: dict, prediction_chf: float, metrics: dict | None = None) -> str:
    """Generate an explanation with OpenAI if OPENAI_API_KEY is configured."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = build_prompt(profile, prediction_chf, metrics)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()


def generate_explanation(profile: dict, prediction_chf: float, metrics: dict | None = None) -> str:
    """Main NLP function used by the Streamlit app.

    Priority:
    1. Use OpenAI GPT if OPENAI_API_KEY exists.
    2. Fall back to deterministic rule-based NLG if the key is missing or API call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return rule_based_explanation(profile, prediction_chf)

    try:
        return generate_openai_explanation(profile, prediction_chf, metrics)
    except Exception:
        # Do not expose API errors to end users in the main explanation.
        return rule_based_explanation(profile, prediction_chf)


def get_nlp_mode() -> str:
    """Return a simple mode label for the app UI."""
    return "OpenAI GPT explanation" if os.getenv("OPENAI_API_KEY") else "Rule-based fallback explanation"

import os
from textwrap import dedent

SYSTEM_MESSAGE = """You are a careful Swiss salary advisor. Explain ML salary predictions in German.
Never claim the prediction is guaranteed. Mention data limitations. Use concise, helpful wording."""


def build_prompt(profile: dict, prediction_chf: float, metrics: dict | None = None) -> str:
    skills = profile.get("skills", "")
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
    {metrics or "nicht verfügbar"}

    Anforderungen an die Antwort:
    1. Kurze Erklärung der wichtigsten Einflussfaktoren.
    2. Konkrete Skill-Empfehlungen.
    3. Hinweis auf Unsicherheit und mögliche Biases.
    4. Keine erfundenen Fakten.
    Antwort auf Deutsch.
    """).strip()


def rule_based_explanation(profile: dict, prediction_chf: float) -> str:
    skills = str(profile.get("skills", "")).lower()
    positives = []
    if "cloud" in skills: positives.append("Cloud-Kenntnisse")
    if "machine learning" in skills or "ml" in skills: positives.append("Machine-Learning-Skills")
    if "sql" in skills: positives.append("SQL")
    if "python" in skills: positives.append("Python")
    if profile.get("location") in ["Zurich", "Geneva", "Basel"]:
        positives.append(f"Standort {profile.get('location')}")
    main = ", ".join(positives[:4]) if positives else "Berufserfahrung, Rolle und Standort"
    return (
        f"Die geschätzte Lohnspanne liegt bei ungefähr CHF {prediction_chf:,.0f} pro Jahr. "
        f"Wichtige positive Faktoren sind {main}. "
        "Für eine höhere Prognose wären vertiefte Cloud-, Data-Engineering- oder ML-Kompetenzen besonders relevant. "
        "Die Aussage ist keine Garantie, sondern eine datenbasierte Schätzung; Unsicherheiten entstehen durch Selbstangaben, regionale Unterschiede und unterrepräsentierte Rollen."
    )


def generate_explanation(profile: dict, prediction_chf: float, metrics: dict | None = None) -> str:
    prompt = build_prompt(profile, prediction_chf, metrics)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return rule_based_explanation(profile, prediction_chf)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return rule_based_explanation(profile, prediction_chf) + f"\n\nFallback genutzt, weil LLM-Aufruf fehlschlug: {exc}"

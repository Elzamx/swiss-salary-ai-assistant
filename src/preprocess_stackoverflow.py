import argparse
import pandas as pd
from pathlib import Path

ROLE_MAP = {
    "Developer, full-stack": "Software Developer",
    "Developer, back-end": "Software Developer",
    "Developer, front-end": "Software Developer",
    "Data scientist or machine learning specialist": "Data Scientist",
    "Data or business analyst": "Business Analyst",
    "Engineer, data": "Data Engineer",
    "Machine learning specialist": "ML Engineer",
}


def first_matching_role(dev_type: str) -> str:
    if pd.isna(dev_type):
        return "Other"
    parts = [p.strip() for p in str(dev_type).split(";")]
    for p in parts:
        if p in ROLE_MAP:
            return ROLE_MAP[p]
    return parts[0] if parts else "Other"


def normalize_education(ed: str) -> str:
    if pd.isna(ed):
        return "Unknown"
    s = str(ed).lower()
    if "master" in s:
        return "Master"
    if "bachelor" in s:
        return "Bachelor"
    if "doctoral" in s or "phd" in s:
        return "PhD"
    if "secondary" in s or "associate" in s or "college" in s:
        return "Apprenticeship"
    return "Other"


def parse_years(x):
    if pd.isna(x):
        return None
    s = str(x).replace("Less than 1 year", "0").replace("More than 50 years", "50")
    try:
        return float(s)
    except ValueError:
        return None


def combine_skills(row):
    cols = ["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith", "ToolsTechHaveWorkedWith", "AISearchDevHaveWorkedWith"]
    skills = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            skills.extend([p.strip().lower() for p in str(row[c]).split(";") if p.strip()])
    return ";".join(sorted(set(skills)))


def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path, low_memory=False)
    salary_col = "ConvertedCompYearly"
    needed = [salary_col, "DevType", "YearsCodePro", "EdLevel", "Country"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Stack Overflow columns: {missing}")

    out = pd.DataFrame({
        "job_title": df["DevType"].apply(first_matching_role),
        "years_experience": df["YearsCodePro"].apply(parse_years),
        "education": df["EdLevel"].apply(normalize_education),
        "location": df["Country"].fillna("Unknown"),
        "salary_chf": pd.to_numeric(df[salary_col], errors="coerce"),
    })
    out["skills"] = df.apply(combine_skills, axis=1)

    out = out.dropna(subset=["salary_chf", "years_experience"])
    out = out[(out["salary_chf"] >= 30000) & (out["salary_chf"] <= 350000)]
    out = out[out["skills"].str.len() > 0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(out)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/salary_profiles.csv")
    args = parser.parse_args()
    preprocess(args.input, args.output)

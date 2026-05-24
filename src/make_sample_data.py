import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

ROLES = ["Data Scientist", "Data Engineer", "Software Developer", "Business Analyst", "ML Engineer", "IT Consultant"]
EDU = ["Bachelor", "Master", "PhD", "Apprenticeship"]
LOC = ["Zurich", "Bern", "Basel", "Geneva", "Remote CH", "Other CH"]
SKILLS = ["python", "sql", "cloud", "machine learning", "docker", "power bi", "java", "project management", "spark", "nlp"]


def main(n: int = 800):
    rows = []
    for _ in range(n):
        role = np.random.choice(ROLES)
        exp = int(np.random.gamma(2.2, 3.0))
        exp = max(0, min(exp, 25))
        edu = np.random.choice(EDU, p=[0.45, 0.35, 0.08, 0.12])
        loc = np.random.choice(LOC, p=[0.34, 0.12, 0.12, 0.12, 0.14, 0.16])
        skill_count = np.random.randint(2, 7)
        skill_list = list(np.random.choice(SKILLS, size=skill_count, replace=False))
        base = 72000
        base += exp * 3200
        base += {"Data Scientist": 12000, "Data Engineer": 14000, "Software Developer": 9000, "Business Analyst": 5000, "ML Engineer": 18000, "IT Consultant": 7000}[role]
        base += {"Bachelor": 7000, "Master": 12000, "PhD": 16000, "Apprenticeship": 3000}[edu]
        base += {"Zurich": 13000, "Bern": 5000, "Basel": 7000, "Geneva": 9000, "Remote CH": 6000, "Other CH": 0}[loc]
        for s in skill_list:
            base += {"python": 3500, "sql": 2500, "cloud": 7000, "machine learning": 6500, "docker": 3500, "power bi": 2200, "java": 3000, "project management": 4000, "spark": 5500, "nlp": 5000}[s]
        salary = max(50000, base + np.random.normal(0, 11000))
        rows.append({"job_title": role, "years_experience": exp, "education": edu, "location": loc, "skills": ";".join(skill_list), "salary_chf": round(salary, 0)})
    out = Path("data/raw/sample_salary_profiles.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

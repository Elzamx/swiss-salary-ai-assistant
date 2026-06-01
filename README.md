---
title: Swiss Salary AI Assistant
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# Swiss Job Salary Intelligence Assistant

AI Applications project for the ZHAW module *KI-Anwendungen*.

## Project Overview

This project combines:
- **ML Numeric Data** for salary prediction
- **NLP / LLM** for natural language explanation

The application predicts annual salaries for technology professionals based on job role, experience, education, location and technical skills. The prediction is then explained in natural language using an OpenAI-based explanation layer with a deterministic fallback.

## AI Block Integration

1. The user enters a job profile in the Streamlit app.
2. The ML model predicts an expected annual salary.
3. The prediction, profile and model metrics are inserted into a structured prompt.
4. OpenAI GPT generates a German explanation and career recommendations.
5. If no API key is configured, the app uses a rule-based fallback explanation.

## Deployment

Hugging Face Spaces:

https://huggingface.co/spaces/miftaelz/SwissSalary

## OpenAI Setup on Hugging Face

The app works best with an OpenAI API key.

In the Hugging Face Space, go to:

```text
Settings → Variables and Secrets
```

Add this secret:

```text
OPENAI_API_KEY = your_api_key_here
```

Optional:

```text
OPENAI_MODEL = gpt-4o-mini
```

If `OPENAI_API_KEY` is missing, the app still works with the deterministic NLP fallback.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local OpenAI usage, set the environment variable before starting the app:

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

## Repository Structure

```text
app.py                  Streamlit application
src/                    ML and NLP logic
models/                 trained models
data/                   datasets
docs/                   project documentation
requirements.txt        Python dependencies
```

## Author

Elza Miftari  
ZHAW – AI Applications Project

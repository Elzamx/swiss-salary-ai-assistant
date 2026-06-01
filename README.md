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

```

## Author

Elza Miftari  
ZHAW – AI Applications Project

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
- ML Numeric Data
- NLP

The application predicts annual salaries for technology professionals based on:
- job role
- experience
- education
- location
- technical skills

The prediction is then explained in natural language using an NLP-based explanation layer.

---

## Deployment

https://huggingface.co/spaces/miftaelz/SwissSalary

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Structure

```text
app.py                  Streamlit application
src/                    ML and NLP logic
models/                 trained models
data/                   datasets
docs/                   project documentation
```

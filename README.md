# Hospital Readmission Prediction API

This repository contains the backend API for the Hospital Readmission Prediction project. It provides machine learning model inference for 30-day readmission risk, intended for use with the clinical decision-support web application.

**Frontend repository:** [readmissions_frontend](https://github.com/lkallen/readmissions_frontend)
**Live Frontend Application:** [https://readmissionprediction.vercel.app/](https://readmissionprediction.vercel.app/)

## Features

- Accepts structured patient data via POST requests
- Returns a probability of 30-day hospital readmission and a binary risk flag
- Built with FastAPI, using a pre-trained scikit-learn model

## Data Source

- The model was trained on a **high-quality, realistic synthetic dataset** that reflects real-world hospital patterns while preserving patient privacy.
- [Hospital Patient Readmission Dataset (Kaggle)](https://www.kaggle.com/datasets/mohamedasak/hospital-patient-readmission-dataset/data)

## Tech Stack

- FastAPI (API framework)
- Pydantic (data validation)
- pandas (preprocessing)
- scikit-learn (model)
- joblib (model serialization)

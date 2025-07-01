# 💼 Job Prediction & Recommendation Portal

An end-to-end job portal web application built for **both job seekers and companies**, with an integrated **Machine Learning model** to score job applications based on relevance. This project combines **Flask**, **MongoDB**, and **scikit-learn** into a fully functional, intelligent recruitment system.

---

## 🚀 Features

### 🔹 For Companies
- Post new job listings
- View applications with a generated **match score**
- **Accept or reject** applicants based on their score

### 🔹 For Job Seekers
- Apply for jobs using:
  - Resume
  - CV
  - Short cover letter
- View:
  - **Match score** for each application
  - **Acceptance or rejection** status
  - **Skill suggestions** for improvement (if rejected)

---

## 🧠 Machine Learning Model

A custom-built model predicts how well a candidate fits a job based on:

- Job description
- Candidate’s resume & cover letter
- Skills match
- Education background
- Years of experience
- Industry alignment
- Previous application history

### 🔍 Tech & Tools Used

| Component        | Usage                                                |
|------------------|------------------------------------------------------|
| `TfidfVectorizer` | Extracts meaningful words from resumes & job posts  |
| `cosine_similarity` | Measures relevance of cover letters                |
| `RandomForestClassifier` | Core classification model for success prediction |
| `StandardScaler` | Normalizes numerical features                        |
| `LabelEncoder`   | Encodes categorical variables like education level   |
| `cross_val_score`| Validates model accuracy and performance             |

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JS
- **Database**: MongoDB
- **ML/NLP**: scikit-learn, pandas, numpy, nltk
- **Deployment Ready**: `.env` support, logging, and model persistence via `joblib`

---

## 📂 Folder Structure


<div align="center">

# 🚀 Machine Learning Analytics Platform
### *An End-to-End Web Platform for Dataset Preprocessing, Multi-Algorithm Training & Model Evaluation*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-6.0-092E20.svg?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DRF](https://img.shields.io/badge/Django_REST_Framework-3.16-red.svg)](https://www.django-rest-framework.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.5-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-key-features) • [Architecture](#-system-architecture) • [Supported Algorithms](#-supported-algorithms) • [Tech Stack](#-technology-stack) • [Getting Started](#-getting-started) • [API Documentation](#-api-endpoints)

</div>

---

## 📌 Overview

**Machine Learning Analytics Platform** is a full-stack, modular web application built to streamline the entire Machine Learning workflow—from exploratory data analysis and feature engineering to multi-algorithm model training, cross-validation, and interactive evaluation. 

Designed for both data scientists and students, the platform couples robust backend execution (Scikit-learn pipelines & Django REST Framework) with interactive data visualization (Plotly & Matplotlib) and built-in educational guides detailing algorithm intuitions and mathematical principles.

---

## ✨ Key Features

### 1. 📊 Dataset Ingestion & Preprocessing
- **Multi-Source Ingestion**: Upload tabular datasets (CSV, Excel `.xlsx`) or import directly via external URL.
- **Automated EDA**: Immediate summary statistics, missing values audit, data types detection, and column distributions.
- **Configurable Preprocessing Pipelines**:
  - Missing value imputation (`mean`, `median`, `most_frequent`, `constant`, row/column drop).
  - Categorical feature encoding (`OneHotEncoder`, `OrdinalEncoder`).
  - Feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
  - Polynomial feature interaction & Variance Threshold filtering.
- **Manual Feature Engineering**: Derive, transform, or filter feature subsets prior to training.

### 2. 🤖 Multi-Algorithm Model Training
- **Extensive Model Catalog**: Support for 15+ classification, regression, and clustering algorithms across multiple algorithmic families.
- **Dynamic Hyperparameter Tuning**: Parameter schemas validated at runtime via JSON Schema.
- **Cross-Validation Framework**: Automated Stratified K-Fold (classification) and K-Fold (regression) with metrics aggregation (mean and standard deviation).
- **Model Persistence**: Serialized pipeline artifacts saved for instant inference and reproducibility.

### 3. 📈 Interactive Analytics & Evaluation
- **Rich Visual Diagnostics**:
  - Actual vs. Predicted scatter plots with regression lines and residual distributions.
  - Interactive Confusion Matrices with multi-class heatmaps.
  - Model Learning Curves & fold performance variance.
  - Interactive correlation matrices and distribution histograms powered by **Plotly**.
- **Comprehensive Metrics**: Accuracy, Precision, Recall, Macro F1-Score, RMSE, MAE, R² Score.

### 4. 🔮 Real-Time Inference & Predictions
- **Single & Batch Prediction**: Submit new data points through the intuitive UI or REST API to receive immediate model predictions and class probabilities.
- **Model Comparison**: Benchmark multiple experiment runs side-by-side on the same dataset split.

### 5. 📚 Built-in Educational Guides
- Detailed mathematical formulations, loss functions, intuition walkthroughs, prerequisites, strengths, and limitations for each supported algorithm.

---

## 🧠 Supported Algorithms

| Family | Algorithm | Supported Task |
| :--- | :--- | :--- |
| **Linear Models** | Simple Linear Regression | Regression |
| | Multiple Linear Regression | Regression |
| | Logistic Regression (L1/L2/ElasticNet) | Classification |
| **Tree-Based** | Decision Tree Classifier | Classification |
| | Decision Tree Regressor | Regression |
| | Random Forest Classifier | Classification |
| | Random Forest Regressor | Regression |
| **Kernel / Distance** | Support Vector Machine (SVC) | Classification |
| | Support Vector Regression (SVR) | Regression |
| | k-Nearest Neighbors (k-NN Classifier) | Classification |
| | k-Nearest Neighbors (k-NN Regressor) | Regression |
| **Probabilistic** | Gaussian Naive Bayes | Classification |
| | Multinomial Naive Bayes | Classification |
| | Bernoulli Naive Bayes | Classification |
| **Clustering** | K-Means Clustering | Clustering |
| **Neural Networks** | Multi-Layer Perceptron (MLPClassifier) | Classification |
| | Multi-Layer Perceptron (MLPRegressor) | Regression |

---

## 🏗️ System Architecture

```
ml_platform/
├── apps/
│   ├── datasets/         # Dataset models, upload handlers, EDA & preprocessors
│   ├── ml_models/        # Pipeline builders, experiment runners, metrics & persistence
│   │   ├── services/     # Core domain logic (pipeline_builder, experiment_runner, etc.)
│   │   └── management/   # Seed commands & database initializers
│   ├── visualization/    # Interactive Plotly & Matplotlib chart generation
│   ├── api/              # RESTful API endpoints, viewsets & serializers (v1)
│   └── users/            # Authentication, accounts & workspace profiles
├── config/               # Django root settings, WSGI/ASGI & root routing
├── templates/            # Django responsive HTML5 templates & modular components
├── static/               # CSS styles, JavaScript interactive charts, asset bundles
├── media/                # Uploaded datasets and serialized model artifacts (.joblib)
├── manage.py
└── requirements.txt      # Project dependencies
```

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Django 6.0, Django REST Framework (DRF)
- **Machine Learning & Data Science**: Scikit-learn, Pandas, NumPy, SciPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Storage / Database**: SQLite (Development) / PostgreSQL compatible (Production)
- **Frontend**: HTML5, Modern CSS3, JavaScript (ES6+), Bootstrap

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/louahabi79/ml_platform.git
cd ml_platform
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Migrations & Seed Algorithms
```bash
python manage.py migrate
python manage.py seed_algorithms
```

### 5. Create a Superuser (Optional)
```bash
python manage.py createsuperuser
```

### 6. Start the Development Server
```bash
python manage.py runserver
```

Open your browser and navigate to `http://127.0.0.1:8000/`.

---

## 🔌 API Endpoints (v1)

The platform provides a fully documented REST API for integration:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET / POST` | `/api/v1/datasets/` | List or upload datasets |
| `GET` | `/api/v1/datasets/{id}/preview/` | Preview dataset rows and schema |
| `GET / POST` | `/api/v1/pipelines/` | List or configure preprocessing pipelines |
| `GET / POST` | `/api/v1/experiments/` | List or launch ML model training experiments |
| `GET` | `/api/v1/experiments/{id}/report/` | Retrieve training report, metrics & charts |
| `POST` | `/api/v1/experiments/{id}/predict/`| Execute predictions with a trained model |
| `GET` | `/api/v1/algorithms/` | List available ML algorithms and hyperparameter schemas |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Developed with ❤️ by <b>Oussama Allouch</b>
</div>

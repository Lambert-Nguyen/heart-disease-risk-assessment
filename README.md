# Heart Disease Risk Assessment System

**CMPE 257 - Machine Learning (Fall 2025)**
San Jose State University

**Team**: Lam Nguyen, James Pham, Le Duy Vu, Vi Thi Tuong Nguyen

---

## Project Overview

A **machine learning system** for predicting heart disease severity using clinical data. Features a full-stack implementation with React frontend, Flask backend, and advanced ML pipeline using **Hierarchical Classification** for improved accuracy.

### Key Achievements

- **Binary Classification**: 85.1% F1-score (**13% above 75% target**)
- **Hierarchical Severity**: 71.4% F1-score (Two-stage: SVM + Random Forest)
- **Full-Stack Demo**: Working end-to-end application
- **Advanced Techniques**: Hierarchical classification, ensemble methods, BorderlineSMOTE

---

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup (Flask API)

```bash
# 1. Navigate to project directory
cd heart-disease-risk-assessment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backend server
python src/api/app.py
```

Backend running at **http://localhost:8000**

### Frontend Setup (React App)

```bash
# 1. Navigate to frontend directory
cd frontendRedesign

# 2. Install dependencies
npm install

# 3. Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# 4. Run development server
npm run dev
```

✅ Frontend running at **http://localhost:3000** (or 5173)

### Test the Application

1. Open http://localhost:3000 in your browser
2. Click **"Start Your Assessment"**
3. Accept terms & conditions
4. Fill the assessment form with test data (see [frontendRedesign/DEMO.md](frontendRedesign/DEMO.md) for test cases)
5. View results with risk level, probability chart, and action items

---

## Performance Results

### Binary Classification (Disease Detection)

| Model | Test F1 | Accuracy | Status |
|-------|---------|----------|--------|
| **SVM (RBF)** | **0.8530** | 0.8533 | **BEST** |
| XGBoost | 0.8471 | 0.8478 | - |
| Logistic Regression | 0.8312 | 0.8315 | - |
| Random Forest | 0.8259 | 0.8261 | - |
| Gradient Boosting | 0.8091 | 0.8098 | - |

**Achievement**: **85.3% F1** vs 75% target → **+13.7% above goal** 

### Multi-class Classification (Hierarchical Approach)

| Approach | Test F1 | Accuracy | Methodology | Status |
|----------|---------|----------|-------------|--------|
| **Hierarchical (SVM + RF)** | **0.7141** | 0.7174 | Binary → Severity | **BEST** |
| Multi-class (Random Forest) | 0.6991 | 0.7011 | Direct 3-class | - |
| Multi-class (Gradient Boosting) | 0.6610 | 0.6576 | Direct 3-class | - |

**Current Model**: Hierarchical Classification
- **Stage 1**: SVM (RBF kernel) Binary Classifier (Disease vs No Disease) - F1: 0.8530
- **Stage 2**: Random Forest Multi-class (Tuned: n_estimators=200, max_depth=10) - For disease cases only
- **Classes**: 0 (No Disease), 1 (Mild Disease), 2 (Severe Disease)
- **Per-Class Performance**: Class 0 (F1=0.83, Recall=0.87), Class 1 (F1=0.68, Recall=0.64), Class 2 (F1=0.48, Recall=0.48)

---

## System Architecture

```
┌─────────────────┐      HTTP/JSON       ┌──────────────────┐
│  React Frontend │  ───────────────────> │   Flask Backend  │
│  (TypeScript)   │  <─────────────────── │    (Python)      │
│  Port 3000      │                       │    Port 8000     │
└─────────────────┘                       └──────────────────┘
        │                                          │
        v                                          v
    TailwindCSS                              ML Pipeline
    React Hook Form                    ┌──────────────────┐
    Recharts                           │  Preprocessing   │
    Axios                              │  - KNN Imputer   │
                                       │  - Label Encoder │
                                       │  - BorderlineSMOTE│
                                       └──────────────────┘
                                                 │
                                                 v
                                       ┌──────────────────┐
                                       │  Hierarchical    │
                                       │  Classifier      │
                                       │  SVM + RF        │
                                       │  F1 = 0.7141     │
                                       └──────────────────┘
```

---

## Project Structure

```
heart-disease-risk-assessment-1/
│
├── 📓 notebooks/
│   ├── 01_exploratory_data_analysis.ipynb  EDA & visualization
│   ├── 02_data_preprocessing.ipynb         Data preprocessing
│   └── 03_model_training.ipynb             Model development (Hierarchical)
│
├── 🔧 src/api/
│   ├── app.py                          Flask API (3 endpoints)
│   └── README.md                       Backend documentation
│
├── frontendRedesign/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.tsx                Landing page
│   │   │   └── Assessment.tsx          Main assessment form
│   │   ├── components/                 Reusable UI components
│   │   ├── services/                   API integration
│   │   ├── mocks/                      Mock data for development
│   │   └── types/                      TypeScript types
│   ├── package.json                    Dependencies
│   ├── vite.config.ts                  Build configuration
│   ├── README.md                       Frontend documentation
│   └── DEMO.md                         Test cases and demo guide
│
├── frontend/                           Legacy frontend (deprecated)
│
├── models/
│   ├── hierarchical_classifier.pkl     Hierarchical (SVM + RF, F1=0.7141)
│   ├── best_binary_model.pkl           SVM Binary classifier
│   ├── best_multiclass_model.pkl       Random Forest multiclass
│   └── model_metadata.pkl              Performance metrics
│
├── data/
│   ├── raw/                            UCI heart disease dataset
│   └── processed/                      Train/test splits, preprocessing artifacts
│
├── docs/
│   ├── CMPE 257_ Proposal.pdf          Project proposal
│   ├── archive/                        Archived documentation
│   └── development_guide/              Development resources
│
├── results/                            Model evaluation results and visualizations
│
├── requirements.txt                    Python dependencies
└── .gitignore
```

---

## Methodology

### Dataset
- **Source**: UCI Heart Disease (4 medical centers)
- **Size**: 920 patients
- **Features**: 14 clinical attributes → 18 after engineering
- **Classes**: 3 severity groups (0: No Disease, 1: Mild Disease, 2: Severe Disease)
- **Original Challenge**: Extreme class imbalance (329:299:108 distribution)
- **Solution**: Hierarchical approach (Binary detection → Severity classification)

### Preprocessing Pipeline

1. **Missing Value Handling**
   - KNN Imputation (k=5) for ALL numeric features
   - Label encoding for categorical features (sex, cp, fbs, restecg, exang, slope, thal)
   - High missingness features: ca (66%), thal (53%), slope (34%)

2. **Feature Engineering**
   - `age_group`: Binned into 5 categories [0-40, 40-50, 50-60, 60-70, 70-100]
   - `bp_category`: Blood pressure categories [0-120, 120-140, 140-160, 160-300]
   - `chol_category`: Cholesterol levels [0-200, 200-240, 240-600]
   - `hr_reserve`: thalch - (220 - age)
   - `cv_risk_score`: age/100 + trestbps/200 + chol/300 + oldpeak/10

3. **Class Imbalance**
   - **Binary**: No SMOTE applied (natural class balance: 407 disease vs 329 no disease)
   - **Multi-class**: BorderlineSMOTE (borderline-1, k_neighbors=5) to balance 3 classes

4. **Scaling**: StandardScaler (fit on train only)

### Models Developed

**Binary Classification**:
- Models: Logistic Regression, Random Forest, XGBoost, SVM (RBF), Gradient Boosting
- Best Model: SVM (RBF kernel) with F1 = 0.8530
- No hyperparameter tuning applied to binary models

**Multi-class Classification**:
- Models: Random Forest, XGBoost, XGBoost Ordinal, Gradient Boosting, SVM, Logistic Regression, KNN
- Best Model: Random Forest (tuned) with F1 = 0.6991
- Hyperparameter tuning: RandomizedSearchCV (20 iterations, 5-fold CV)
- All using BorderlineSMOTE (borderline-1, k_neighbors=5) for class balancing

**Hierarchical Classification** (FINAL):
- **Stage 1**: SVM Binary (F1 = 0.8530)
- **Stage 2**: Random Forest Multi-class (n_estimators=200, max_depth=10)
- **Overall**: F1 = 0.7141 (21.9% improvement over baseline)

---

## Frontend Features

- **Single-page assessment form** with 4 sections (Demographics, Symptoms, Vitals, Diagnostics)
- **Real-time validation** using React Hook Form
- **Color-coded results** (Green/Orange/Red-Pink for 3 severity levels)
- **Probability visualization** with Recharts bar charts showing hierarchical probabilities
- **Action items** personalized by risk level
- **Responsive design** (mobile-friendly)
- **Medical disclaimer** and terms & conditions

### Tech Stack
- React 19.2.0 + TypeScript 5.9.3
- Vite 7.2.4 (build tool)
- TailwindCSS 4.1.17
- React Hook Form 7.67.0
- Axios 1.13.2
- Recharts 3.5.1

---

## API Endpoints

### POST /api/predict
Predicts heart disease severity level.

**Request**:
```json
{
  "age": 65,
  "sex": "male",
  "cp": "typical angina",
  "trestbps": 160,
  "chol": 280,
  "fbs": true,
  "restecg": "ST-T abnormality",
  "thalch": 120,
  "exang": true,
  "oldpeak": 2.5,
  "slope": "downsloping",
  "ca": "2",
  "thal": "reversible defect"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "prediction": 2,
    "confidence": 0.78,
    "probabilities": {"0": 0.05, "1": 0.17, "2": 0.78},
    "risk_category": "Severe-Critical",
    "risk_color": "#E91E63",
    "action_items": ["Contact cardiologist IMMEDIATELY", ...]
  }
}
```

### GET /api/health
Health check endpoint.

### GET /api/model-info
Model information and metadata (approach, F1-score, class names).

See [src/api/README.md](src/api/README.md) for full API documentation.

---

## Documentation

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Project overview (this file) | Everyone |
| [frontendRedesign/README.md](frontendRedesign/README.md) | Frontend-specific docs | Frontend developers |
| [src/api/README.md](src/api/README.md) | Backend API docs | Backend developers |

---

## Multi-class Classification Progress

### Hierarchical Approach

We implemented **Hierarchical Classification** to improve severity prediction:

**Methodology**:
- **Stage 1**: SVM (RBF kernel) Binary Classifier - Disease vs No Disease
  - F1 = 0.8530, Accuracy = 0.8533
  - Trained on 736 samples (407 disease, 329 no disease)
  - No SMOTE applied due to natural balance
- **Stage 2**: Random Forest Multi-class - Severity classification (0, 1, 2)
  - Applied only to cases predicted as disease in Stage 1
  - Tuned with RandomizedSearchCV (n_estimators=200, max_depth=10, min_samples_split=5)
  - Trained with BorderlineSMOTE (borderline-1, k_neighbors=5)
- **Prediction Flow**: Binary → Multi-class for disease cases only

**Current Performance**: **71.41% F1-score, 71.74% Accuracy**
- Class 0 (No Disease): Precision=0.79, Recall=0.87, F1=0.83 (82 test samples)
- Class 1 (Mild Disease): Precision=0.72, Recall=0.64, F1=0.68 (75 test samples)
- Class 2 (Severe Disease): Precision=0.48, Recall=0.48, F1=0.48 (27 test samples)

### Gap to Target

Current F1 (71.41%) vs target (75%) = **-3.59% gap**

**Achievements**:
1. **21.9% improvement** over direct multi-class baseline (0.5863 → 0.7141)
2. **2.15% improvement** over best direct Random Forest multi-class (0.6991 → 0.7141)
3. Mimics clinical workflow (detection → severity assessment)
4. Tested 7 multi-class models and XGBoost Ordinal classification

**Remaining challenges**:
1. Class 2 (Severe) has lower performance (F1=0.48) due to only 108 training samples (11.7% of dataset)
2. Significant missing data: ca (66%), thal (53%), slope (34%)
3. Small dataset (920 total samples, 184 test samples)

**Context**: Published research on UCI heart disease achieves 55-65% F1 for multi-class, making our 71.41% **significantly above state-of-the-art**.

**Success**: Binary classification **exceeded target by 13.7%** (85.3% vs 75%), and hierarchical approach brought multi-class **within 3.6% of target**.

---

## Future Improvements

### Short-term (1-2 weeks)
- SHAP explanations for model interpretability
- Confusion matrix per-class analysis
- Cost-sensitive learning with medical costs

### Medium-term (1-2 months)
- Full backend with PostgreSQL + JWT auth
- User dashboard for assessment history
- Further optimization (ensemble methods, cost-sensitive learning)

### Long-term (3-6 months)
- Cloud deployment (Vercel + Railway)
- External validation on different datasets
- Mobile app (React Native)
- Research paper on ordinal medical classification

---

## Resources

### Datasets
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Kaggle Mirror](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

### Technical References
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- [React TypeScript Guide](https://react-typescript-cheatsheet.netlify.app/)

---

## Team

- **Lam Nguyen** - Data preprocessing & feature engineering
- **James Pham** - Model development & training
- **Le Duy Vu** - Full-stack implementation (frontend + backend)
- **Vi Thi Tuong Nguyen** - Evaluation & documentation

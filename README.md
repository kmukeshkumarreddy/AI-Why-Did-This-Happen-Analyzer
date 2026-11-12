# AI "Why Did This Happen?" Analyzer

> **"Go Beyond Predictions. Discover the Reasons Behind the Results."**

An intelligent data investigation platform that combines **statistical hypothesis testing**, **interpretable machine learning**, and **Explainable AI (XAI)** to uncover the hidden mechanisms, patterns, and root causes behind unexpected outcomes in enterprise datasets.

---

## 1. Problem Statement

Traditional machine learning and enterprise analytics predominantly answer:
> **"What will happen?"** *(e.g., will this customer churn? will this transaction fail?)*

However, business leaders, risk officers, and engineers rarely need just a probability score. When things go wrong—such as when **churn jumps from 10% to 25%**, or **conversion rates crater after a pricing release**—the most critical question is:
> **"Why did it happen?"**

Current solutions require hours of manual SQL slicing, ad-hoc bivariate charts, or opaque black-box models that cannot explain the multi-factor combinations that drove the shift.

---

## 2. Proposed Solution & Vision

The **AI "Why Did This Happen?" Analyzer** acts as an automated data detective:
1. **Identifies Anomalous / Target Cohorts:** Isolates the outcome of interest (e.g., churned vs. retained, high-delay flights, abnormal refund rates).
2. **Decomposes Statistical Significance:** Runs two-sample t-tests, Mann-Whitney U, ANOVA, and Chi-Square tests to prove which differences are real vs. random noise.
3. **Discovers Non-Linear Multivariate Patterns:** Trains high-capacity tree ensembles (Random Forest, XGBoost).
4. **Calculates Exact Attribution with XAI:** Utilizes SHAP (Shapley Additive exPlanations) and LIME to calculate individual and global feature impact.
5. **Synthesizes Root-Cause Reports:** Translates complex mathematical attributions into actionable plain-English diagnostics and executive dossiers.

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Language & Runtime** | Python 3.12+ (Windows / Linux / macOS) |
| **Frontend Framework** | Streamlit (Custom Enterprise Dark Theme) |
| **Data Processing** | Pandas, NumPy, OpenPyXL |
| **Statistical Analysis** | SciPy, Statsmodels |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Explainable AI (XAI)**| SHAP, LIME |
| **Data Visualization** | Plotly, Matplotlib, Seaborn |
| **Quality & Testing** | Pytest |

---

## 4. Modular Project Architecture

```
why-did-this-happen-analyzer/
│
├── app.py                      # Main Streamlit application entry point & router
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules for data, cache, and venvs
│
├── config/                     # Centralized project configuration
│   ├── __init__.py             # Config package exports
│   ├── settings.py             # AppSettings dataclass, theme config, directory paths
│   └── constants.py            # App labels, navigation enums, algorithm types
│
├── modules/                    # Core business logic & analytical pipeline
│   ├── __init__.py             # Module namespace
│   ├── data_loader.py          # Tabular file ingestion (.csv, .xlsx, .parquet)
│   ├── data_profiler.py        # Automated schema & data quality auditing
│   ├── preprocessing.py        # Encoders, scalers, and missing value handlers
│   ├── eda.py                  # Descriptive stats, KDE distributions, correlations
│   ├── outcome_detection.py    # Target variable discovery & cohort splitters
│   │
│   ├── statistics/             # Statistical hypothesis testing (t-test, ANOVA, Chi2)
│   │   └── __init__.py
│   ├── machine_learning/       # Predictive model training (Random Forest, XGBoost)
│   │   └── __init__.py
│   ├── explainability/         # Explainable AI engines (SHAP, LIME)
│   │   └── __init__.py
│   ├── anomaly/                # Unsupervised anomaly isolation (Isolation Forest)
│   │   └── __init__.py
│   ├── investigation/          # Multi-factor causal attribution & synthesis
│   │   └── __init__.py
│   ├── visualization/          # Interactive Plotly & Matplotlib figure builders
│   │   └── __init__.py
│   └── reporting/              # Executive report generation & PDF/HTML export
│       └── __init__.py
│
├── components/                 # Reusable Streamlit UI components
│   ├── __init__.py
│   ├── sidebar.py              # Navigation menu, status badges, environment info
│   ├── dashboard.py            # Hero banner, paradigm shift cards, placeholders
│   └── metric_cards.py         # Custom HTML/CSS enterprise metric cards
│
├── data/                       # Local file storage (gitignored except .gitkeep)
│   ├── uploads/                # User uploaded datasets
│   ├── processed/              # Cleaned & transformed feature caches
│   └── samples/                # Built-in demonstration datasets
│
├── tests/                      # Automated unit and integration tests
│   ├── __init__.py
│   ├── test_config.py          # Configuration & constants test suite
│   └── test_utils.py           # Utility & validator test suite
│
└── utils/                      # Helper tools and validation logic
    ├── __init__.py
    ├── helpers.py              # Formatting helpers, system info, custom CSS injection
    └── validators.py           # File extension, size, and dataframe integrity checks
```

---

## 5. Getting Started & Installation

### Prerequisites
- **Python 3.11+** (Python 3.12 recommended)
- **Git**
- Windows PowerShell, CMD, or bash shell

### 1. Clone or Open Workspace
```powershell
cd "c:\AI Why Did This Happen Analyzer"
```

### 2. Create and Activate Virtual Environment
```powershell
# Using Python 3.12 (Windows py launcher)
py -3.12 -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# (Or activate on CMD)
# .\.venv\Scripts\activate.bat
```

### 3. Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. How to Run the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

The application will be accessible at:
- **Local URL:** `http://localhost:8501`
- **Network URL:** Displayed in your terminal

---

## 7. Running the Automated Tests

Execute the test suite using pytest:
```powershell
pytest -v
```

---

## 8. Multi-Phase Development Roadmap

| Phase | Title | Focus & Deliverables | Status |
|:---:|---|---|:---:|
| **Phase 1** | **Architecture & Environment Setup** | Modular directory layout, settings, constants, enterprise UI theme, test suite | ✅ **Completed** |
| **Phase 2** | **Dataset Upload & Understanding** | Multi-format loaders, automated health audits, profiling KPIs, target selection | ✅ **Completed** |
| **Phase 3** | **Data Cleaning & Preprocessing** | Missing value imputations, categorical encoding, feature scaling | ✅ **Completed** |
| **Phase 4** | **Exploratory Data Analysis (EDA)** | Distribution curves, correlation heatmaps, bivariate scatter matrices | ✅ **Completed** |
| **Phase 5** | **Unexpected Outcome Detection** | Empirical anomaly isolation (Z-Score, MAD, IQR), distribution shifts, category swings, time series transitions, segment disaggregation | ✅ **Completed** |
| **Phase 6** | **Statistical Investigation Engine** | Formal hypothesis testing (t-test, Mann-Whitney U, ANOVA, Chi-Square) | ⏳ Next Phase |

---

## 9. License & Standards
Developed under enterprise software engineering practices with strict modularity, comprehensive typing, clean separation of concerns, and automated testing.

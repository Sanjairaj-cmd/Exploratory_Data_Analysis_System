# 📊 Exploratory Data Analysis and Data Insights System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An automated, intelligent, end-to-end **Exploratory Data Analysis (EDA) & Data Insights System** built with Python and Streamlit. This professional platform ingests raw CSV datasets, audits data quality, performs missing value imputation & duplicate cleanup, generates univariate/bivariate distributions, detects IQR & Z-score outliers, derives automated natural-language data science insights, and exports standalone HTML/Markdown reports and cleaned datasets.

---

## 🚀 Key Features

### 1. 📂 Dataset Upload & Ingestion
- Upload custom **CSV datasets** or select from built-in sample datasets (*Employee Analytics* & *Housing Prices*).
- Auto-detect row count, column attributes, data types, memory usage, numerical vs. categorical variables.

### 2. 📋 Comprehensive Data Overview
- Interactive KPI cards for total records, columns, duplicates, missing cells, and numerical/categorical splits.
- First 10 (head) and last 10 (tail) records inspection.
- Column data type breakdown and unique value sample previews.
- Full numerical summary (mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis, IQR) & categorical frequency tables.

### 3. 🧹 Interactive Data Cleaning Engine
- Missing data detection with percentage bar chart visualizations.
- Column-specific missing value imputation strategies (**Mean, Median, Mode, Drop Rows, Drop Column, Forward Fill, Backward Fill, Constant**).
- Duplicate records audit & one-click deduplication.
- Interactive column data type conversion (`numeric`, `category`, `string`, `datetime`).
- Dynamic session state updates so cleaned data propagates across all tabs in real-time.

### 4. 📈 Univariate Analysis & Distribution Diagnostics
- Detailed single-variable metrics: Mean, Median, Mode, Variance, Standard Deviation, Min, Max, Range, IQR, Skewness, Kurtosis.
- Dual visual distribution plots (**Histogram with KDE curve + Boxplot**) for numerical columns.
- Horizontal count plots with percentage badges for categorical columns.

### 5. 🔀 Bivariate Relationship Explorer
- **Numerical vs. Numerical**: Interactive scatter plots with regression trendlines and optional hue grouping.
- **Numerical vs. Categorical**: Grouped statistical summary tables and side-by-side **Box plots / Violin plots**.
- **Categorical vs. Categorical**: Contingency tables & cross-tabulations.

### 6. 🔗 Feature Correlation Diagnostics
- Pearson and Spearman rank correlation matrix calculations.
- Color-annotated heatmaps using Seaborn.
- Filterable table of top correlated feature pairs above user-adjusted threshold sliders.

### 7. 🎯 Outlier Detection & Treatment Engine
- Dual detection algorithms: **Interquartile Range (IQR)** with custom factor slider and **Z-Score** with threshold slider.
- Column-wise outlier count and percentage summary matrix.
- Annotated box plots illustrating lower and upper outlier cutoff thresholds.
- One-click outlier treatment options: **Winsorization (Capping)** or **Row Removal**.

### 8. 💡 Automated Natural-Language Insights
- Automated pattern detection evaluating skewness, severe missingness (>20%), multicollinearity risks, and heavy outlier tails.
- Dynamically generated bulleted executive summaries and data quality warning badges.
- Step-by-step actionable data science preprocessing & modeling recommendations.

### 9. 📄 Report Generation & Export System
- One-click export of cleaned dataset as a downloadable CSV.
- Downloadable standalone **HTML EDA Summary Report** styled with CSS cards, tables, and badges.
- Downloadable **Markdown EDA Report** for documentation or GitHub wikis.

---

## 🛠️ Technology Stack

- **Core Engine:** Python 3.9+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Analytics & Algorithms:** `scikit-learn`, `tabulate`
- **Interactive UI Framework:** `streamlit`

---

## 📂 Project Architecture

```
eda_insights_system/
├── app.py                     # Main Streamlit web application dashboard
├── requirements.txt           # Python dependency requirements
├── README.md                  # Project documentation
├── data/                      # Sample datasets directory
│   ├── sample_employee_data.csv
│   └── housing_prices.csv
├── notebooks/                 # Jupyter walkthrough notebooks
│   └── eda_exploration.ipynb
├── src/                       # Modular python backend engine
│   ├── __init__.py
│   ├── data_loader.py         # CSV reader and schema metadata engine
│   ├── cleaner.py             # Missing imputation, deduplication, type casting
│   ├── analyzer.py            # Descriptive statistics & correlation engine
│   ├── outliers.py            # IQR & Z-score outlier detection and treatment
│   ├── visualizer.py          # Seaborn & Matplotlib chart generation routines
│   ├── insights.py            # Automated heuristic insight generator
│   └── reporter.py            # Standalone HTML & Markdown report exporter
├── reports/                   # Output folder for generated reports
└── visualizations/            # Output folder for exported charts
```

---

## 💻 Installation & Local Setup

### Prerequisites
Ensure Python 3.9 or higher is installed on your system.

### 1. Clone or Open Project in VS Code
```bash
cd C:\Users\Ragur\.gemini\antigravity\scratch\eda_insights_system
```

### 2. Create & Activate a Virtual Environment (Recommended)
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

Launch the Streamlit dashboard by executing the following command in terminal:

```bash
streamlit run app.py
```

The system will automatically launch in your default web browser at:
`http://localhost:8501`

---

## 🧩 Module Explanations

| Module | Description & Functionality |
| :--- | :--- |
| **`app.py`** | Main UI driver featuring custom CSS, sidebar navigation, reactive session state, metric cards, and interactive parameter controls. |
| **`src/data_loader.py`** | Handles CSV reading with encoding fallbacks, schema identification, column categorization, and memory calculation. |
| **`src/cleaner.py`** | Provides functions for missing value imputation (`mean`, `median`, `mode`, `ffill`, `bfill`), row deduplication, and safe data type casting. |
| **`src/analyzer.py`** | Computes descriptive stats (skewness, kurtosis, IQR), bivariate group statistics, Pearson/Spearman correlation matrices, and top pair extraction. |
| **`src/outliers.py`** | Implements IQR ($Q_1 - 1.5 \times IQR$) and Z-score ($|Z| > 3.0$) detection methods, returning bounds, outlier masks, and capping/removal routines. |
| **`src/visualizer.py`** | Creates publication-ready Matplotlib/Seaborn figures for missing value bars, distribution histograms, violin plots, and correlation heatmaps. |
| **`src/insights.py`** | Generates dynamic natural language findings, alerts, and step-by-step preprocessing recommendations based on data metrics. |
| **`src/reporter.py`** | Compiles summary statistics, quality metrics, insights, and recommendations into standalone, styled HTML and Markdown documents. |

---

## 📝 License & Author Notes

This project is built for data science teams, analysts, and machine learning engineers to accelerate data preprocessing and discovery.
Created as part of the **Exploratory Data Analysis and Data Insights System** project.

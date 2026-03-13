# Optimizing Green Hydrogen Production: An AI-Based Approach

> MSc thesis using Random Forest with Bayesian Optimization and SHAP explainability to predict and optimize green hydrogen output from renewable energy sources.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)](https://shap.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)

---

## Research Question

How can an AI-driven Random Forest model, optimized through Bayesian Optimization, enhance the efficiency and predictive accuracy of green hydrogen production from renewable energy sources — while ensuring interpretability and applicability in real-world energy systems?

---

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Random Forest (Bayesian Optimized)** | **0.41** | **0.58** | **0.91** |
| Gradient Boosting | 0.50 | 0.65 | 0.86 |
| SVR | 1.28 | 1.69 | -0.14 |

Random Forest outperformed all models across every metric. SVR produced a negative R², confirming its inability to generalize in high-dimensional, multivariate electrolysis data — a finding consistent with Wei et al. (2025). Results validated via 5-fold cross-validation with low variance across folds.

---

## What This Research Does

1. **Compares 4 regression models** (Random Forest, Gradient Boosting, SVR, Linear Regression) under consistent evaluation metrics on a green hydrogen production dataset
2. **Optimizes the best model** (Random Forest) via Bayesian Optimization using Optuna — reducing computational cost vs. grid search while improving predictive accuracy
3. **Explains predictions with SHAP** — both global feature importance and local per-prediction explanations, identifying that electrolyzer efficiency, system efficiency, and solar irradiance are the top drivers of hydrogen output
4. **Validates robustness** via 5-fold cross-validation, residual analysis, and sensitivity testing
5. **Aligns with SDG 7** (Affordable and Clean Energy) and **SDG 13** (Climate Action)

---

## Key Findings

- **Electrolyzer efficiency** (0.85 correlation with H₂ output) is the single most impactful parameter — confirmed by both Pearson correlation and SHAP analysis
- **Bayesian Optimization** (Optuna) achieved better hyperparameter tuning than grid search at lower computational cost — relevant for energy-efficient AI training
- **SHAP dependence plots** revealed non-linear interactions between temperature and pressure, consistent with electrochemical process theory
- **SVR fails** on this type of data due to sensitivity to feature scaling, kernel selection, and multicollinearity — even after extensive Bayesian tuning
- The model is **transferable** to countries with high renewable potential (Chile, Morocco, India) due to its low computational requirements and open-source stack

---

## Dataset

- **Source:** [Renewable Hydrogen Production Dataset](https://www.kaggle.com/datasets/ziya07/renewable-hydrogen-production-dataset/data) (Kaggle, user "ziya07")
- **Records:** 2,535 simulated production instances
- **Target variable:** Hydrogen production (kg/day), range ~45–59 kg/day
- **Input features:** Solar irradiance (W/m²), wind speed (m/s), ambient temperature (°C), electrolyzer efficiency (%), specific electrical consumption (kWh/kg H₂), system load factor (%), PV power (kW), wind power (kW), desalination power (kW), system efficiency (%), feasibility score

**Preprocessing:**
- No missing values (synthetic dataset)
- Outlier removal via IQR filter (load_factor, energy)
- MinMaxScaler applied for SVR; tree models used raw features
- High-correlation features (>0.90) removed to reduce multicollinearity
- PCA evaluated but rejected to preserve SHAP interpretability
- 80/20 train-test split, stratified by output quantiles

---

## Methodology

Built on the **CRISP-DM** framework (Cross Industry Standard Process for Data Mining):

1. **Business Understanding** — Optimize green hydrogen production efficiency using interpretable AI
2. **Data Understanding** — Exploratory analysis: distributions, correlations, heatmaps
3. **Data Preparation** — Feature selection, scaling, outlier removal, stratified split
4. **Modeling** — 4 regression models trained, hyperparameters tuned via Bayesian Optimization
5. **Evaluation** — Holdout test set + 5-fold CV + SHAP + residual analysis + sensitivity testing
6. **Deployment Planning** — Industrial integration pathway proposed (not implemented)

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11 |
| ML Models | scikit-learn (RandomForestRegressor, GradientBoostingRegressor, SVR, LinearRegression) |
| Optimization | Optuna (Bayesian Optimization via BayesSearchCV) |
| Explainability | SHAP (summary plots, dependence plots, force plots) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Repository Structure

```
├── README.md
├── notebooks/
│   └── green_hydrogen_optimization.ipynb    # Full pipeline: EDA → modeling → evaluation → SHAP
├── data/
│   └── renewable_hydrogen_dataset_2535.csv   # Kaggle dataset
├── figures/
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── residual_plot_rf.png
│   ├── shap_summary.png
│   └── hydrogen_distribution.png
├── thesis/
│   └── THESIS_FINAL.pdf                      # Full MSc thesis document
└── requirements.txt
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/cristob777/msc-thesis-green-hydrogen.git
cd msc-thesis-green-hydrogen

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/green_hydrogen_optimization.ipynb
```

---

## Academic Context

- **Degree:** MSc in AI for Business — National College of Ireland
- **Supervisor:** Dr Muslim Jameel Syed
- **Submitted:** August 2025
- **Methodology:** CRISP-DM
- **SDG alignment:** SDG 7 (Affordable and Clean Energy), SDG 13 (Climate Action)

---

## Author

**Cristóbal Cáceres Ortúzar** — Solutions Engineer, based in Dublin, Ireland.

[![LinkedIn]https://www.linkedin.com/in/cristobal361/

---

## License

This repository is shared for academic and portfolio purposes. The thesis content is the intellectual property of the author and National College of Ireland.

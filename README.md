 UK Carbon Emissions Forecasting: A Comparative Analysis of Time Series and Machine Learning Approaches

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

  Project Overview

This project presents a comprehensive comparative analysis of forecasting algorithms applied to UK local authority greenhouse gas (GHG) emissions data. The study evaluates the performance of traditional time series methods (ARIMA/ARIMAX) against modern machine learning approaches (Random Forest, XGBoost) in predicting carbon emissions across different sectors.

 Key Objectives
- Compare forecasting accuracy between statistical and ML-based approaches
- Identify sector-specific emission trends and patterns
- Evaluate the impact of limited temporal data (20 observations) on model performance
- Implement sector-specific feature engineering for improved predictions

  Dataset

Source: UK Department for Business, Energy & Industrial Strategy (BEIS)  
Coverage: 326 Local Authorities across England  
Time Period: 2005-2024 (20 annual observations)  
Sectors Analyzed: 8 emission sectors with focus on top 3:
-  Transport (29% of total emissions)
-  Domestic (20% of total emissions)
-  Industry (15% of total emissions)

Dataset Link: [UK Local Authority GHG Emissions](https://www.gov.uk/government/statistics/uk-local-authority-and-regional-greenhouse-gas-emissions-national-statistics-2005-to-2021)

  Methodology

 Models Implemented

1. ARIMA (Baseline)
   - Automated order selection using AIC
   - Sector-specific parameterization
   - Performance: R² ≈ -0.0022 to -0.0043

2. Random Forest
   - 200 estimators, depth optimization
   - GridSearchCV hyperparameter tuning
   - Performance: R² ≈ -0.0056 to -0.0171

3. XGBoost (Manual)
   - Gradient boosting with regularization
   - Manual sector-specific tuning
   - Performance: R² ≈ -0.0006 to -0.0171

4. XGBoost (Optimized)
   - RandomizedSearchCV with 100 iterations
   - 11-parameter optimization space
   - Performance: R² ≈ -0.0017 to -0.0069

5. ARIMAX (Final Selection)
   - Sector-specific exogenous variables
   - Transport: COVID impact (2020-2021)
   - Domestic/Industry: Recession (2008-2009) + Brexit (2016-2020)
   - Performance: R² ≈ -0.0007 to -0.0022

 Train-Test Split
- Training Set: 70% (14 observations, 2005-2018)
- Test Set: 30% (6 observations, 2019-2024)

  Key Findings

 Model Performance
All models achieved near-zero R² scores (-0.0171 to -0.0006), indicating that:
- Limited temporal data (20 observations) is insufficient for robust predictive modeling
- Extensive hyperparameter tuning provides minimal benefit with scarce data
- Feature engineering alone cannot overcome fundamental data scarcity
- ARIMAX was selected for theoretical soundness and interpretability, not performance superiority

 Sector Insights
- Transport emissions show highest volatility with COVID-19 impact
- Domestic emissions influenced by recession cycles and policy changes
- Industry emissions demonstrate gradual declining trends

 Technical Learnings
- Sector-specific features (2-3 variables) outperform uniform feature sets (7 variables)
- Overfitting prevention requires careful feature selection with limited data
- Time series approaches maintain stability despite data constraints

  Project Structure

```
├── Carbon Emissions.csv                            Raw dataset
├── df_ce_long_cleaned.csv                         Processed long-format data
├── england_regions_emissions.csv                  Regional aggregation
├── EDA on Carbon Emissions.ipynb                  Initial exploration
├── EDA on Carbon Emissions Final.ipynb            Refined analysis
├── EDA on Carbon Emissions Final Extended.ipynb   Complete implementation
├── Machine Learning model- ARIMA.ipynb            ARIMAX modeling
├── Carbon_Emissions_Project_Report.md             Comprehensive report
└── README.md                                      This file
```

  Getting Started

 Prerequisites

```bash
Python 3.8+
Jupyter Notebook/Lab
```

 Required Libraries

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.4.0
statsmodels>=0.12.0
```

 Installation

```bash
 Clone the repository
git clone https://github.com/edwardleen95/Data-Science-FPR.git
cd Data-Science-FPR

 Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels jupyter
```

 Running the Analysis

```bash
 Launch Jupyter Notebook
jupyter notebook

 Open and run notebooks in sequence:
 1. EDA on Carbon Emissions Final Extended.ipynb  (Main analysis)
 2. Machine Learning model- ARIMA.ipynb            (ARIMAX implementation)
```

  Results Summary

| Model | Transport R² | Domestic R² | Industry R² |
|-------|-------------|------------|-------------|
| ARIMA | -0.0022 | -0.0007 | -0.0043 |
| Random Forest | -0.0171 | -0.0134 | -0.0056 |
| XGBoost (Manual) | -0.0171 | -0.0013 | -0.0006 |
| XGBoost (Tuned) | -0.0069 | -0.0042 | -0.0017 |
| ARIMAX (Selected) | -0.0022 | -0.0007 | -0.0043 |

Note: Near-zero/negative R² values indicate that no model successfully captured predictive patterns due to limited temporal observations. ARIMAX was selected for its theoretical framework and interpretability.

  Academic Context

This project was completed as part of the Data Science Program at the University of Hertfordshire, Semester 3. It demonstrates:
- Rigorous model comparison methodology
- Honest reporting of data limitations
- Sector-specific feature engineering
- Comprehensive hyperparameter optimization
- Statistical and ML modeling proficiency

  Future Work

- Data augmentation: Incorporate monthly/quarterly data for increased temporal resolution
- External features: Weather patterns, economic indicators, policy implementation dates
- Ensemble methods: Hybrid ARIMAX-XGBoost approaches
- Spatial analysis: Geographic clustering of similar emission patterns
- Deep learning: LSTM/GRU networks for sequence modeling with augmented data

  Author

Edward Lee  
University of Hertfordshire  
Data Science Program - Semester 3

  License

This project is licensed under the MIT License - see the LICENSE file for details.

  Acknowledgments

- UK Department for Business, Energy & Industrial Strategy for providing open emissions data
- University of Hertfordshire for academic guidance
- Open-source community for excellent ML/statistics libraries

  Contact

For questions or collaboration opportunities, please reach out via GitHub issues or email.

---

 If you find this project useful, please consider giving it a star!

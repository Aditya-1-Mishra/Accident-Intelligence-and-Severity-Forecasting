# Accident Intelligence and Severity Forecasting
This project focuses on predicting the potential severity of road accidents using machine learning techniques and various accident-related factors such as road, traffic, and environmental conditions.

The repository currently contains exploratory data analysis (EDA), feature engineering, and research conducted as part of the model development process. The project is under active development.

## important-feature-identification

Before building any model, I wanted to know which features actually matter. So I ran permutation importance across three targets — severity, distance, and duration — each treated separately. For severity, I tested LightGBM, CatBoost, and XGBoost to find the best base model, settled on CatBoost, and pulled the top 25 features from it. These are the same features used in the main classification notebook. For distance and duration, I used XGBoost regressors with log transformation on the skewed targets, then cross-validated the duration features against Random Forest to make the selection more reliable. Finally, I took the intersection across all three to find features that matter universally — not just for one target.

## severity-classification-optuna.ipynb

This is the main classification notebook. I experimented with four strategies to handle the severe class imbalance — no sampling, full random undersampling, sample weights, and moderate undersampling (manually capping only the dominant classes). Moderate undersampling came out on top and was used for tuning. I first ran GridSearchCV across 216 combinations, then followed up with Optuna for smarter search — which pushed the macro F1 from 0.43 → 0.54 across 60 trials.

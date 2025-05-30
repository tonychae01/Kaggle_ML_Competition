

# Kaggle_ML_Competition: Liver Cirrhosis Prediction 🏆

**Tony Chae – 3rd Place among 150 teams, Kaggle Cirrhosis Prediction Competition**

---
## 📄 Kaggle Competition Report

[![View PDF Report](https://img.shields.io/badge/View%20Report-PDF-blue?style=flat-square&logo=adobe)](Kaggle%20Report.pdf)

> **Note:** You can also click the badge above or use the link below  
> [**Open the full PDF report**](Kaggle%20Report.pdf)
>

## Key Contributions:
-	Identified seven clinical categorical variables (e.g., Ascites, Spiders) and continuous labs; compared SimpleImputer (mean/mode), IterativeImputer (MissForest), and uniform random sampling for missingness, then applied one hot encoding and min max normalization to harmonize 20 raw features into a consistent modeling matrix.
-	Engineered seven additional predictors by capturing key clinical interdependencies - grouping NA patterns across Ascites/Hepatomegaly/Spiders and encoding Bilirubin–Edema relationships—expanding the feature set to 27 dimensions and boosting separability across the three cirrhosis stages.
- Trained an ensemble of five XGBoost (multi:softprob) models on CUDA, averaging soft probabilities; optimized hyperparameters via Optuna (e.g., max_depth=12, min_child_weight=8, subsample=0.9666, colsample_bytree=0.1236, learning_rate=0.0262, n_estimators=687), achieving top 3 performance by log loss, and evaluated pseudo labeling (≥ 0.90 confidence) to augment training to ~22 k samples.

## Overview

This repository contains the code and supporting files for the liver cirrhosis classification project that earned **3rd place** in the [Kaggle Cirrhosis Prediction Competition](https://www.kaggle.com/competitions) under the “multi-class classification” challenge. We leverage advanced data-cleaning, feature-engineering, and ensemble modeling techniques to predict patient status across three classes with high accuracy and low log loss.

---
```
## Repository Structure

├── data/
│   ├── train.npy
│   ├── test_label.npy
│   
├── model/
│   ├── 01_data_exploration.ipynb
│   ├── 02 pseudo_data_gen_model.ipynb
│   └── 03 some models...ipynb (XGBOOST, CatBoost, RF)
├── other/
│   ├── best_param.pkl
│   
├── Final_Report.pdf
└── README.md


---

## Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/liver-cirrhosis-prediction.git
   cd liver-cirrhosis-prediction
   ```

2. **Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Data

- **Training set**: 15,000 samples with 20 features  
- **Test set**: 10,000 samples (unlabeled)  
- **Target classes**: 3 liver cirrhosis statuses  
- **Feature types**: categorical, continuous, integer 

---

## Methodology

### 1. Data Cleaning & Imputation  
- Identified categorical columns (`Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`, `Stage`) and continuous columns (e.g., `Age`, `Bilirubin`, `Albumin`)
- Addressed missing values using multiple strategies:
  - **Simple Imputer** (mean/mode)
  - **Iterative Imputer** (MissForest)
  - **Random Imputer** (uniform sampling)
- Experimented with combinations of one‑hot encoding and normalization for different model types 

### 2. Feature Engineering  
- Exploited domain knowledge on cirrhosis clinical signs:
  - Grouped N/A patterns among `Ascites`, `Hepatomegaly`, `Spiders`
  - Captured `Bilirubin`–`Edema` relationships
  - Encoded N/A patterns as new categorical features
- Augmented original 20 features to **27 engineered columns** 

### 3. Modeling & Ensembling  
- **Single best model**: XGBoost (`multi:softprob`) on GPU (CUDA)  
- **Ensemble strategy**: Trained **5 XGBoost** instances with different random seeds, averaged predicted probabilities  
- Explored stacking with RandomForest, LightGBM, SVM, CatBoost using a logistic-regression meta-learner (ultimately outperformed by the XGBoost ensemble) 

### 4. Hyperparameter Optimization  
- Evaluated **GridSearch**, **RandomizedSearch**, and **Optuna** (Tree‑structured Parzen Estimator)  
- Final tuned parameters for XGBoost:  
  ```python
  final_param = {
      'max_depth': 12,
      'min_child_weight': 8,
      'subsample': 0.9666,
      'colsample_bytree': 0.1236,
      'learning_rate': 0.0262,
      'n_estimators': 687
  }
  ```  

### 5. Pseudo‑Labeling (Exploratory)  
- Generated pseudo‑labels on test set using high‑confidence predictions (≥ 0.90 threshold)  
- Augmented training data from 15,000 → ~22,000 samples  
- Observed overfitting to known test labels; final submission **did not** incorporate pseudo‑labels 

---

## Usage

1. **Prepare data** in `data/`
2. **Clean & engineer features**  
   ```bash
   python src/data_cleaning.py --input data/train.csv --output data/clean_train.csv
   python src/feature_engineering.py --input data/clean_train.csv --output data/fe_train.csv
   ```
3. **Train & tune model**  
   ```bash
   python src/train.py --config config/xgb_optuna.yaml
   ```
4. **Evaluate & submit**  
   ```bash
   python src/evaluate.py --model outputs/best_xgb.pkl --test data/test.csv --output submission.csv
   ```

---

## Dependencies

- Python ≥ 3.8  
- pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna  
- CUDA-compatible GPU (optional for faster XGBoost)  

_All dependencies are listed in_ `requirements.txt`.

---

## Results

- **Final log loss**: 0.3657  
- **Rank**: 3rd place out of ~150 teams on the public leaderboard  

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
```

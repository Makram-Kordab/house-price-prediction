# House Price Prediction

Supervised machine learning project predicting residential property sale 
prices using the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Kaggle leaderboard RMSE: 0.12593**

## Project overview

Built an end-to-end regression pipeline that takes raw housing data through exploratory analysis, feature engineering, 
model training, and hyperparameter tuning to generate accurate price predictions.

## Technical approach

### Exploratory data analysis
- Analysed 1,460 properties across 80 features
- Identified right-skewed SalePrice distribution — applied log transformation to normalise target variable
- Mapped missing values: distinguished absent features (e.g. no garage) from unknown values, applying different 
  imputation strategies to each

### Feature engineering
- Engineered `TotalSF` (combined basement + floor areas), 
  `TotalBaths`, `HouseAge`, and `RemodAge` from existing columns
- Applied ordinal encoding to quality rating columns 
  (preserving natural order: Poor → Excellent)
- One-hot encoded 31 nominal categorical features
- Removed two high-leverage outliers identified during EDA
- Scaled all numeric features with RobustScaler to 
  reduce outlier influence

### Modelling
- Evaluated four algorithms: Linear Regression, Ridge, 
  Lasso, Gradient Boosting
- Applied 5-fold cross-validation throughout to ensure 
  reliable RMSE estimates
- Final predictions use a blended ensemble (30% Ridge, 
  70% Gradient Boosting)

### Hyperparameter tuning
- Ridge alpha tuned via GridSearchCV 
  — best alpha: 20
- Gradient Boosting tuned via RandomizedSearchCV 
  (40 iterations) — key parameters below

---

## Results

| Model | CV RMSE |
| Linear Regression | 0.1281 |
| Ridge (default) | 0.1155 |
| Gradient Boosting (default) | 0.1218 |
| Ridge (tuned) | 0.1154 |
| Gradient Boosting (tuned) | 0.1180 |
| Blended ensemble | 0.0871 |
| **Kaggle leaderboard** | **0.12593** |

---

## Best hyperparameters found

**Gradient Boosting:**
- n_estimators: 442
- learning_rate: 0.0303
- max_depth: 3
- max_features: 0.5385
- min_samples_leaf: 20
- subsample: 0.7484

**Ridge:**
- alpha: 20

---

## Tech stack

- Python 3.13
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Vs Code

---

## Project structure

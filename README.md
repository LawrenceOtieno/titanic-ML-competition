# Titanic: Machine Learning from Disaster 🚢

A complete, end-to-end machine learning pipeline built to predict passenger survival on the RMS Titanic. This project implements data preprocessing, targeted feature engineering, hyperparameter tuning, and cross-validation to generate robust, reproducible survival predictions.

---

## 📊 Results & Model Evaluation

Models were evaluated using **Stratified 5-Fold Cross-Validation** on the 891-row training dataset to ensure stable generalization and prevent overfitting. 

| Model | CV Accuracy | Notes |
| :--- | :--- | :--- |
| **XGBoost** | **0.8417** | **Best overall CV score (Selected Final Model)** |
| Random Forest | 0.8395 | Runner-up (Tuned via `GridSearchCV`) |
| Gradient Boosting | 0.8350 | Baseline ensemble variant |
| Logistic Regression | 0.8058 | Linear baseline performance |
| Support Vector Machine | 0.6891 | Performance impact from unscaled data |

> 📌 **Note:** Kaggle Public Leaderboard scores (~0.76) typically differ from local CV scores due to the relatively small 209-passenger evaluation subset on the platform. Trust local stratified CV for structural validation.

---

## ⚙️ Feature Engineering Highlights

To maximize model performance, the raw data is transformed in `src/preprocess.py` using several domain-specific techniques:

*   **Title Extraction:** Parses social titles (*Mr, Mrs, Master, etc.*) from the `Name` column to capture explicit social status, gender context, and age proxies.
*   **Targeted Age Imputation:** Instead of applying a global median, missing `Age` values are imputed using the median age of the passenger's specific `Title` group.
*   **Family Dynamics:** Combines `SibSp` (siblings/spouses) and `Parch` (parents/children) into a unified `FamilySize` feature, paired with an `IsAlone` boolean flag.
*   **Quantile Binning:** Applies quantile binning (`pd.qcut`) to `Fare` and standard binning to `Age` to mitigate the impact of outliers and help tree-based models split cleanly.
*   **Deck Extraction:** Extracts the deck letter from the `Cabin` string to capture socio-economic positioning within the ship's layout.

---

## 🛠️ Project Structure

```text
titanic-ml-competition/
│
├── data/
│   ├── raw/                  # Original train.csv and test.csv (git-ignored)
│   └── processed/            # Cleaned, engineered datasets
│
├── models/                   # Serialized, trained models (.pkl)
│
├── notebooks/                # Jupyter notebooks for Exploratory Data Analysis (EDA)
│
├── src/                      # Production source code pipeline
│   ├── preprocess.py         # Data cleaning, imputation, and feature engineering
│   ├── train.py              # Trains and tunes Random Forest via GridSearchCV
│   ├── train_ensemble.py     # Tests an ensemble VotingClassifier
│   ├── train_xgboost.py      # Trains the final selected XGBoost model
│   ├── experiments.py        # Validates feature-ablation variants
│   └── predict.py            # Loads a trained model and generates predictions
│
├── submissions/              # Generated output files for Kaggle submission
├── requirements.txt          # Reproducible Python environment dependencies
└── README.md                 # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone & Set Up Environment
Clone the repository and set up a clean Python virtual environment:
```bash
git clone https://github.com/YOUR_USERNAME/titanic-ml-competition.git
cd titanic-ml-competition
python -m venv venv
```

Activate the environment based on your operating system:
*   **Windows:**
    ```bash
    venv\Scripts\activate
    ```
*   **macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Download `train.csv` and `test.csv` from the [Kaggle Titanic Competition Page](https://kaggle.com) and place them inside the `data/raw/` directory.

### 4. Execute the Pipeline
Run the training script to evaluate the best model architecture (XGBoost):
```bash
cd src
python train_xgboost.py
```

Generate the final submission predictions:
```bash
python predict.py
```

The resulting file will be compiled and saved out to `submissions/submission.csv`.

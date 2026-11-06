import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from preprocess import load_and_combine, engineer_features, encode_and_split

FEATURES = ['Pclass','Sex','Age','SibSp','Parch','Fare',
            'Embarked','Title','FamilySize','IsAlone',
            'HasCabin','Deck','AgeBin','FareBin']

def main():
    print("Loading and engineering data...")
    df = load_and_combine('../data/raw/train.csv', '../data/raw/test.csv')
    df = engineer_features(df)
    X, y, X_test, test_df = encode_and_split(df, FEATURES)

    # XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    print("Evaluating XGBoost...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("\n=== XGBOOST RESULTS ===")
    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    if scores.mean() > 0.8395:
        print("XGBoost beat the Random Forest! Training on full data...")
        model.fit(X, y)
        joblib.dump(model, '../models/xgb_model.pkl')
        print("Model saved to models/xgb_model.pkl")
    else:
        print("Random Forest remains undefeated. We will stick with it.")

if __name__ == '__main__':
    main()
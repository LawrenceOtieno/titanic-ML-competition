import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from preprocess import load_and_combine, engineer_features, encode_and_split

# The features we want to feed into the model
FEATURES = ['Pclass','Sex','Age','SibSp','Parch','Fare',
            'Embarked','Title','FamilySize','IsAlone',
            'HasCabin','Deck','AgeBin','FareBin']

def main():
    print("Loading and engineering data...")
    df = load_and_combine('../data/raw/train.csv', '../data/raw/test.csv')
    df = engineer_features(df)
    X, y, X_test, test_df = encode_and_split(df, FEATURES)

    print("Training model (this will take 15-30 seconds)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Parameter grid to find the best settings
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [4, 5, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 0.5],
    }
    
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    gs.fit(X, y)
    
    print("\n=== RESULTS ===")
    print("Best parameters:", gs.best_params_)
    print("Best Cross-Validation Accuracy:", round(gs.best_score_, 4))
    
    # Save the model
    joblib.dump(gs.best_estimator_, '../models/rf_tuned.pkl')
    print("Model saved to models/rf_tuned.pkl")

if __name__ == '__main__':
    main()
import joblib
import pandas as pd
from preprocess import load_and_combine, engineer_features, encode_and_split

FEATURES = ['Pclass','Sex','Age','SibSp','Parch','Fare',
            'Embarked','Title','FamilySize','IsAlone',
            'HasCabin','Deck','AgeBin','FareBin']

def main():
    print("Loading model...")
    model = joblib.load('../models/xgb_model.pkl')
    
    print("Preparing test data...")
    df = load_and_combine('../data/raw/train.csv', '../data/raw/test.csv')
    df = engineer_features(df)
    _, _, X_test, test_df = encode_and_split(df, FEATURES)

    print("Making predictions...")
    preds = model.predict(X_test).astype(int)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'].astype(int),
        'Survived': preds
    })
    
    # Sanity checks to make sure we don't submit a broken file
    assert len(submission) == 418, f"Wrong number of rows: {len(submission)}"
    assert list(submission.columns) == ['PassengerId', 'Survived']
    
    # Save to the submissions folder
    submission.to_csv('../submissions/submission.csv', index=False)
    print(f"Success! Submission saved to submissions/submission.csv")
    print(f"Predicted Survival Rate: {preds.mean():.3f} (Expected ~0.36 to 0.40)")

if __name__ == '__main__':
    main()
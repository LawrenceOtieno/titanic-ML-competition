from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from preprocess import load_and_combine, engineer_features, encode_and_split

FEATURES = ['Pclass','Sex','Age','SibSp','Parch','Fare',
            'Embarked','Title','FamilySize','IsAlone',
            'HasCabin','Deck','AgeBin','FareBin']

def main():
    print("Loading and preparing data...")
    df = load_and_combine('../data/raw/train.csv', '../data/raw/test.csv')
    df = engineer_features(df)
    X, y, X_test, test_df = encode_and_split(df, FEATURES)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define the models we want to test
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        
        # We use the parameters GridSearch found were best for RF
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=4, 
            max_features=0.5, random_state=42
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
        
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, random_state=42)
    }

    print(f"{'Model':<25} | {'CV Accuracy':<12} | {'Std Dev'}")
    print("-" * 55)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{name:<25} | {scores.mean():.4f}       | ± {scores.std():.4f}")

if __name__ == '__main__':
    main()
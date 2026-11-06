from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from preprocess import load_and_combine, engineer_features, encode_and_split

def run_experiment(name, features, model):
    print(f"\n--- Running {name} ---")
    df = load_and_combine('../data/raw/train.csv', '../data/raw/test.csv')
    df = engineer_features(df)
    X, y, X_test, test_df = encode_and_split(df, features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()

if __name__ == '__main__':
    # The standard baseline (what we just submitted)
    base_features = ['Pclass','Sex','Age','SibSp','Parch','Fare',
                     'Embarked','Title','FamilySize','IsAlone',
                     'HasCabin','Deck','AgeBin','FareBin']
    
    base_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4, 
        max_features=0.5, random_state=42
    )

    # Experiment A: Drop Age
    feat_a = [f for f in base_features if f not in ['Age', 'AgeBin']]
    run_experiment("Experiment A: Drop Age", feat_a, base_model)

    # Experiment B: Shallower Trees
    model_b = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=2, 
        max_features=0.5, random_state=42
    )
    run_experiment("Experiment B: Shallower Trees", base_features, model_b)

    # Experiment C: Simpler Features (Drop Fare, Deck, HasCabin)
    feat_c = [f for f in base_features if f not in ['Fare', 'FareBin', 'Deck', 'HasCabin']]
    run_experiment("Experiment C: Simpler Features", feat_c, base_model)
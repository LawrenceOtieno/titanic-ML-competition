import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_combine(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Mark which rows are train (1) and which are test (0)
    train['is_train'] = 1
    test['is_train'] = 0
    test['Survived'] = np.nan  # Placeholder for the target
    
    # Combine them so we can process them together without data leakage
    df = pd.concat([train, test], ignore_index=True, sort=False)
    return df

def engineer_features(df):
    # 1. Extract Title from Name (Mr, Mrs, Miss, etc.)
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev',
                   'Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 3. Handle Age (impute based on the median age of their Title group)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda s: s.fillna(s.median()))
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,60,100], labels=False).fillna(2).astype(int)

    # 4. Handle Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

    # 5. Cabin / Deck
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('U')

    # 6. Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    return df

def encode_and_split(df, features):
    # Encode text columns to numbers (fitting ONLY on train data to avoid leakage)
    for col in ['Sex', 'Embarked', 'Title', 'Deck']:
        le = LabelEncoder()
        mask_train = df['is_train'] == 1
        mask_test = df['is_train'] == 0
        
        # Tell Pandas to allow mixed types (strings turning into integers)
        df[col] = df[col].astype(object) 

        le.fit(df.loc[mask_train, col])
        df.loc[mask_train, col] = le.transform(df.loc[mask_train, col])
        
        # Handle test data safely in case a category appears that wasn't in train
        known = set(le.classes_)
        df.loc[mask_test, col] = df.loc[mask_test, col].apply(
            lambda x: le.transform([x])[0] if x in known else 0)
        df[col] = df[col].astype(int)

    train_df = df[df['is_train'] == 1]
    test_df = df[df['is_train'] == 0]
    
    X = train_df[features]
    y = train_df['Survived'].astype(int)
    X_test = test_df[features]
    
    return X, y, X_test, test_df
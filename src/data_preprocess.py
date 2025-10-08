# src/data_preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessor():
    numeric_features = ['Age', 'Fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def prepare_train_test(df, target='Survived', test_size=0.2, random_state=42):
    X = df.copy()
    y = X.pop(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = load_data('../data/train.csv')
    print(df.head())

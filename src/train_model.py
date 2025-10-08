# src/train_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocess import load_data, build_preprocessor, prepare_train_test

def train_and_save(data_path='data/train.csv', model_path='models/rf_model.joblib'):
    df = load_data(data_path)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    df = df[features + ['Survived']]

    X_train, X_test, y_train, y_test = prepare_train_test(df)

    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, model_path)
    print('Saved model to', model_path)

if __name__ == '__main__':
    train_and_save()

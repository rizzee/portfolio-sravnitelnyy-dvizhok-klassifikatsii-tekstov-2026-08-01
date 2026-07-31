from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def load_dataset(path: Path) -> pd.DataFrame:
    # Load CSV with text and label columns
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series]:
    # Basic text preprocessing
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_col])
    y = df[label_col]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2) -> Tuple:
    # Split data into train/test sets
    return train_test_split(X, y, test_size=test_size)


def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    # Train and return Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train) -> LinearSVC:
    # Train and return SVM classifier
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> float:
    # Return accuracy score
    return model.score(X_test, y_test)

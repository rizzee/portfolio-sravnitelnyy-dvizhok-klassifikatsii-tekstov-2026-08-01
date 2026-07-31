import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from text_classification import (
    load_dataset,
    preprocess_data,
    train_test_split_data,
    train_naive_bayes,
    train_svm
)


class TestClassifiers(unittest.TestCase):
    def setUp(self):
        # Simple dataset for testing
        self.data = pd.DataFrame({
            'text': ['good movie', 'bad film', 'excellent show', 'terrible acting'],
            'label': [1, 0, 1, 0]
        })

    def test_load_dataset(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.csv'
            self.data.to_csv(path, index=False)
            loaded = load_dataset(path)
            self.assertEqual(len(loaded), 4)
            self.assertIn('text', loaded.columns)
            self.assertIn('label', loaded.columns)

    def test_preprocess_and_split(self):
        X, y = preprocess_data(self.data, text_col='text', label_col='label')
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.25)
        self.assertEqual(X_train.shape[0], 3)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(y_train.shape[0], 3)
        self.assertEqual(y_test.shape[0], 1)

    def test_train_models(self):
        X, y = preprocess_data(self.data, text_col='text', label_col='label')
        X_train, _, y_train, _ = train_test_split_data(X, y, test_size=0.25)
        nb = train_naive_bayes(X_train, y_train)
        svc = train_svm(X_train, y_train)
        self.assertIsInstance(nb, MultinomialNB)
        self.assertIsInstance(svc, LinearSVC)


if __name__ == '__main__':
    unittest.main()

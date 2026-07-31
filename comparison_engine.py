from typing import Dict, List

from sklearn.metrics import precision_score, recall_score, f1_score


def compare_classifiers(y_true: List[int], predictions: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    """
    Compare classifier predictions using precision, recall, and F1.
    Args:
        y_true: True labels
        predictions: Dictionary of classifier name to predicted labels
    Returns:
        Dictionary of classifier name to metrics
    """
    results = {}
    for name, y_pred in predictions.items():
        results[name] = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
    return results

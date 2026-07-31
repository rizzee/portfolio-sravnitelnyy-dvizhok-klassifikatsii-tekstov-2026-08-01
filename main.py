import argparse
from pathlib import Path

from text_classification import load_dataset, preprocess_data
from comparison_engine import compare_classifiers
from visualization import visualize_results


def main():
    parser = argparse.ArgumentParser(description='Compare text classification algorithms.')
    parser.add_argument('dataset', type=Path, help='Path to dataset file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    # Load and preprocess data
    data = load_dataset(args.dataset)
    X, y = preprocess_data(data)

    # Compare classifiers
    results = compare_classifiers(X, y)

    # Optional visualization
    if args.visualize:
        visualize_results(results)


if __name__ == '__main__':
    main()
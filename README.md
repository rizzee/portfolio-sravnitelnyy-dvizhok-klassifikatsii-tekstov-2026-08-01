# Text Classification Comparison Engine

Библиотека для сравнительного анализа точности различных алгоритмов классификации текстов (например, Naive Bayes и LinearSVC) на одном наборе данных. Инструмент помогает наглядно изучить разницу в метриках (precision, recall, F1) и понять логику работы разных моделей.

## Запуск

Сначала установите зависимости:
```bash
pip install -r requirements.txt
```

Запустите сравнение через CLI:
```bash
python main.py --dataset data/sample.csv
```

## Пример

```python
from comparison_engine import ClassifierComparator

# Пример использования логики сравнения
comparator = ClassifierComparator(data)
results = comparator.compare(['naive_bayes', 'linear_svc'])
print(results)
```

## Тесты

Для запуска автоматических тестов используйте:
```bash
python -m unittest discover -s tests -v
```

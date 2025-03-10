# -*- coding: utf-8 -*-
"""
Программа: MovieReviewAnalyzer
Описание: Анализатор отзывов о фильмах с использованием нейронной сети.
"""

import numpy as np
import math
from collections import Counter
import sys

# Функция загрузки данных из файлов
def load_data():
    """Загружает отзывы и метки из текстовых файлов.

    Returns:
        reviews (list): Список строк, содержащих отзывы.
        labels (list): Список меток ("POSITIVE" или "NEGATIVE").
    """
    with open('reviews.txt', 'r') as f:
        reviews = list(map(lambda x: x.strip(), f.readlines()))
    
    with open('labels.txt', 'r') as f:
        labels = list(map(lambda x: x.strip().upper(), f.readlines()))
    
    return reviews, labels

# Функция для создания словаря и преобразования данных в числовую форму
def preprocess_data(reviews):
    """Преобразует отзывы в набор уникальных слов и создает числовое представление.

    Args:
        reviews (list): Список строк с отзывами.

    Returns:
        input_dataset (list): Список списков индексов слов для каждого отзыва.
        vocab (list): Список уникальных слов.
        word2index (dict): Словарь, сопоставляющий слова с их индексами.
    """
    tokens = list(map(lambda x: set(x.split(" ")), reviews))

    vocab = set()
    for sent in tokens:
        vocab.update(sent)

    vocab = list(vocab)
    word2index = {word: i for i, word in enumerate(vocab)}

    input_dataset = []
    for sent in tokens:
        sent_indices = [word2index[word] for word in sent if word in word2index]
        input_dataset.append(list(set(sent_indices)))

    return input_dataset, vocab, word2index

# Функция сигмоид
def sigmoid(x):
    """Возвращает значение сигмоидальной функции.

    Args:
        x (float): Входное значение.

    Returns:
        float: Результат применения функции.
    """
    return 1 / (1 + np.exp(-x))

# Функция для инициализации весов
def initialize_weights(vocab_size, hidden_size):
    """Создает начальные веса для слоев нейронной сети.

    Args:
        vocab_size (int): Размер словаря (количество уникальных слов).
        hidden_size (int): Размер скрытого слоя.

    Returns:
        tuple: Матрицы весов для первого и второго слоя.
    """
    np.random.seed(1)
    weights_0_1 = 0.2 * np.random.random((vocab_size, hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1
    return weights_0_1, weights_1_2

# Функция обучения модели
def train_model(input_dataset, target_dataset, weights_0_1, weights_1_2, iterations=2, alpha=0.01):
    """Обучает модель на предоставленных данных.

    Args:
        input_dataset (list): Список входных данных (индексы слов).
        target_dataset (list): Список меток (0 или 1).
        weights_0_1 (np.ndarray): Матрица весов первого слоя.
        weights_1_2 (np.ndarray): Матрица весов второго слоя.
        iterations (int): Количество эпох обучения.
        alpha (float): Скорость обучения.
    """
    for iter in range(iterations):
        correct, total = 0, 0
        for i in range(len(input_dataset) - 1000):
            x, y = input_dataset[i], target_dataset[i]

            # Прямой проход
            layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
            layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

            # Обратное распространение ошибки
            layer_2_delta = layer_2 - y
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)

            # Обновление весов
            weights_0_1[x] -= layer_1_delta * alpha
            weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha

            # Вычисление точности
            if np.abs(layer_2_delta) < 0.5:
                correct += 1
            total += 1

            if i % 1000 == 999:
                print(f"Iter:{iter} Progress:{i / len(input_dataset):.2%} Accuracy:{correct / total:.2%}")

# Функция тестирования модели
def test_model(input_dataset, target_dataset, weights_0_1, weights_1_2):
    """Тестирует модель на отложенных данных.

    Args:
        input_dataset (list): Список входных данных (индексы слов).
        target_dataset (list): Список меток (0 или 1).
        weights_0_1 (np.ndarray): Матрица весов первого слоя.
        weights_1_2 (np.ndarray): Матрица весов второго слоя.
    """
    correct, total = 0, 0
    for i in range(len(input_dataset) - 1000, len(input_dataset)):
        x, y = input_dataset[i], target_dataset[i]

        # Прямой проход
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        if np.abs(layer_2 - y) < 0.5:
            correct += 1
        total += 1

    print(f"Test Accuracy: {correct / total:.2%}")

# Функция поиска похожих слов
def similar(word, weights_0_1, word2index):
    """Находит слова, близкие по смыслу к заданному.

    Args:
        word (str): Целевое слово.
        weights_0_1 (np.ndarray): Матрица весов первого слоя.
        word2index (dict): Словарь, сопоставляющий слова с их индексами.

    Returns:
        list: Список слов, наиболее похожих на заданное.
    """
    target_index = word2index[word]
    scores = Counter()

    for other_word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference * raw_difference
        scores[other_word] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)

# Основная программа
if __name__ == "__main__":
    # Загрузка данных
    reviews, labels = load_data()

    # Преобразование данных
    input_dataset, vocab, word2index = preprocess_data(reviews)
    target_dataset = [1 if label == 'POSITIVE' else 0 for label in labels]

    # Инициализация весов
    hidden_size = 100
    weights_0_1, weights_1_2 = initialize_weights(len(vocab), hidden_size)

    # Обучение модели
    train_model(input_dataset, target_dataset, weights_0_1, weights_1_2)

    # Тестирование модели
    test_model(input_dataset, target_dataset, weights_0_1, weights_1_2)

    # Пример использования функции similar
    print(similar('beautiful', weights_0_1, word2index))

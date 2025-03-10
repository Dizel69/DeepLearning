# matrix_neural_network.py

import numpy as np

# Определение функции активации ReLU
def relu(x):
    return np.maximum(0, x)

# Параметры обучения
alpha = 0.1  # скорость обучения
hidden_size = 4  # количество нейронов в скрытом слое

# Данные: состояние светофоров и действия "идти" или "стоять"
streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

# Начальные веса для простейшей нейронной сети
weights = np.array([0.5, 0.48, -0.7])

# Итерации обучения для одного светофора
input = streetlights[0]  # первый набор входных данных
goal_prediction = walk_vs_stop[0]  # целевой результат для первого светофора

for iteration in range(20):
    prediction = input.dot(weights)
    error = (goal_prediction - prediction) ** 2
    delta = prediction - goal_prediction
    weights -= alpha * (input * delta)
    print(f"Ошибка: {error:.6f} Предсказание: {prediction:.6f}")

# Демонстрация операций с векторами
vector_a = np.array([0, 1, 2, 1])
vector_b = np.array([2, 2, 2, 3])

print(f"{vector_a * vector_b}")  # поэлементное умножение
print(f"{vector_a + vector_b}")  # поэлементное сложение
print(f"{vector_a * 0.5}")  # умножение на скаляр
print(f"{vector_a + 0.5}")  # сложение с числом

# Полное обучение по всему набору данных
for iteration in range(40):
    total_error = 0
    for row, goal in zip(streetlights, walk_vs_stop):
        prediction = row.dot(weights)
        error = (goal - prediction) ** 2
        total_error += error
        delta = prediction - goal
        weights -= alpha * row * delta
        print(f"Предсказание: {prediction:.6f}")
    print(f"Суммарная ошибка: {total_error:.6f}\n")

# Инициализация слоёв для глубокой нейронной сети
np.random.seed(1)
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

layer_0 = streetlights[0]
layer_1 = relu(layer_0.dot(weights_0_1))
layer_2 = layer_1.dot(weights_1_2)

print(f"Глубокая сеть: Выходной слой {layer_2}")

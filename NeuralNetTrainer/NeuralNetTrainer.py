import sys
import numpy as np
from keras.datasets import mnist

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразование изображений и меток для обучения
images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])

# Перевод меток в one-hot кодировку
one_hot_labels = np.zeros((len(labels), 10))
for i, label in enumerate(labels):
    one_hot_labels[i][label] = 1
labels = one_hot_labels

# Преобразование тестовых данных

# Нормирование изображений до [0, 1]
test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, label in enumerate(y_test):
    test_labels[i][label] = 1

# Задаём случайное зерно для возпроизводимости
np.random.seed(1)

# ReLU и его производная 
relu = lambda x: (x >= 0) * x  # Функция ReLU возвращает x если x > 0, 0 иначе
relu2deriv = lambda x: x >= 0  # Производная ReLU: 1 для x > 0, 0 иначе

# Параметры модели
alpha = 0.005  # Скорость обучения
iterations = 350  # Количество итераций
hidden_size = 40  # Размер скрытого слоя
pixels_per_image = 784  # Число пикселей в изображении 28x28
num_labels = 10  # Количество классов

# Инициализация весов слоев
weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

# Основной цикл обучения
for iteration in range(iterations):
    total_error = 0.0
    correct_count = 0

    for i in range(len(images)):
        # Прямое распространение
        layer_0 = images[i:i + 1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        # Ошибка и точность
        total_error += np.sum((labels[i:i + 1] - layer_2) ** 2)
        correct_count += int(np.argmax(layer_2) == np.argmax(labels[i:i + 1]))

        # Обратное распространение ошибки
        layer_2_delta = labels[i:i + 1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        # Обновление весов
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if iteration % 10 == 0:
        print(f"Iteration: {iteration}, Error: {total_error:.4f}, Correct: {correct_count}/{len(images)}")

# Оценка на тестовых данных
test_correct_count = 0
for i in range(len(test_images)):
    layer_0 = test_images[i:i + 1]
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)
    test_correct_count += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

print(f"Test Accuracy: {test_correct_count / len(test_images) * 100:.2f}%")

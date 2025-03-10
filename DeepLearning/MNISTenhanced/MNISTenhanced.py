import numpy as np
from keras.datasets import mnist

# Установка семени для повторяемости результатов
np.random.seed(1)

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Подготовка тренировочных данных
images = x_train[:1000].reshape(1000, 28 * 28) / 255
labels = np.zeros((len(y_train[:1000]), 10))
for i, l in enumerate(y_train[:1000]):
    labels[i][l] = 1

# Подготовка тестовых данных
test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

# Функции активации и их производные
def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

# Гиперпараметры
alpha = 2
iterations = 300
pixels_per_image = 784
num_labels = 10
batch_size = 128

# Размеры для свёрточного слоя
input_rows = 28
input_cols = 28
kernel_rows = 3
kernel_cols = 3
num_kernels = 16

# Расчёт размера скрытого слоя после свёртки
hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

# Инициализация весов и свёрточных фильтров
kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

# Функция для выборки участка изображения
def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)

# Основной цикл обучения
for j in range(iterations):
    correct_cnt = 0

    for i in range(len(images) // batch_size):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        layer_0 = images[batch_start:batch_end].reshape(batch_size, 28, 28)

        # Разделение изображения на секции для свёртки
        sects = [
            get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
            for row_start in range(layer_0.shape[1] - kernel_rows)
            for col_start in range(layer_0.shape[2] - kernel_cols)
        ]

        expanded_input = np.concatenate(sects, axis=1)
        flattened_input = expanded_input.reshape(expanded_input.shape[0] * expanded_input.shape[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(expanded_input.shape[0], -1))

        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in range(batch_size):
            labelset = labels[batch_start + k:batch_start + k + 1]
            correct_cnt += int(np.argmax(layer_2[k:k + 1]) == np.argmax(labelset))

        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        kernels -= alpha * flattened_input.T.dot(layer_1_delta.reshape(kernel_output.shape))

    test_correct_cnt = 0
    for i in range(len(test_images)):
        layer_0 = test_images[i:i + 1].reshape(1, 28, 28)

        sects = [
            get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
            for row_start in range(layer_0.shape[1] - kernel_rows)
            for col_start in range(layer_0.shape[2] - kernel_cols)
        ]

        expanded_input = np.concatenate(sects, axis=1)
        flattened_input = expanded_input.reshape(expanded_input.shape[0] * expanded_input.shape[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(expanded_input.shape[0], -1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

    print(f"I:{j} Test-Acc:{test_correct_cnt / float(len(test_images)):.4f} Train-Acc:{correct_cnt / float(len(images)):.4f}")

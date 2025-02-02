import numpy as np
from keras.datasets import mnist

# Фиксируем зерно для воспроизводимости результатов
np.random.seed(42)

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Подготовка данных
train_images = x_train[:1000].reshape(1000, 28 * 28) / 255
train_labels = np.zeros((1000, 10))
for i, label in enumerate(y_train[:1000]):
    train_labels[i][label] = 1

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, label in enumerate(y_test):
    test_labels[i][label] = 1

# Определение функций активации

def tanh(x):
    """Гиперболический тангенс для активации"""
    return np.tanh(x)


def tanh2deriv(output):
    """Производная гиперболического тангенса"""
    return 1 - (output ** 2)


def softmax(x):
    """Функция softmax для нормализации выхода"""
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Параметры модели
learning_rate = 2
iterations = 300
hidden_size = 100
pixels_per_image = 784
num_labels = 10
batch_size = 100

# Инициализация весов
weights_input_hidden = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
weights_hidden_output = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

# Основной цикл обучения
for epoch in range(iterations):
    correct_train_count = 0

    for batch_start in range(0, len(train_images), batch_size):
        batch_end = batch_start + batch_size
        layer_0 = train_images[batch_start:batch_end]
        layer_1 = tanh(np.dot(layer_0, weights_input_hidden))

        # Применение dropout для регуляризации
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        layer_2 = softmax(np.dot(layer_1, weights_hidden_output))

        # Подсчет правильных ответов для текущей эпохи
        for i in range(batch_size):
            correct_train_count += int(np.argmax(layer_2[i]) == np.argmax(train_labels[batch_start + i]))

        # Обратное распространение ошибки
        layer_2_delta = (train_labels[batch_start:batch_end] - layer_2) / batch_size
        layer_1_delta = layer_2_delta.dot(weights_hidden_output.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        # Обновление весов
        weights_hidden_output += learning_rate * layer_1.T.dot(layer_2_delta)
        weights_input_hidden += learning_rate * layer_0.T.dot(layer_1_delta)

    correct_test_count = 0

    # Оценка точности на тестовом наборе
    for i in range(len(test_images)):
        layer_0 = test_images[i:i + 1]
        layer_1 = tanh(np.dot(layer_0, weights_input_hidden))
        layer_2 = np.dot(layer_1, weights_hidden_output)

        correct_test_count += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

    if epoch % 10 == 0:
        print(f"Эпоха {epoch}: Точность на тесте {correct_test_count / len(test_images):.4f}, \
              Точность на обучении {correct_train_count / len(train_images):.4f}")

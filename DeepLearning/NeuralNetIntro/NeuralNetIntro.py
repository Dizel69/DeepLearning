import numpy as np

# Простейшая нейронная сеть для выполнения предсказаний
class NeuralNetwork:
    def __init__(self, weights):
        # Инициализация весов нейронной сети
        self.weights = weights

    def single_layer_predict(self, input):
        """
        Простое предсказание с одним входом и одним весом.
        Возвращает произведение входного значения на вес.
        """
        return input * self.weights

    def multiple_input_predict(self, inputs, weights):
        """
        Предсказание для нескольких входов с использованием соответствующих весов.
        Проверяем, что длина входов и весов совпадает.
        Складываем произведения каждой пары.
        """
        assert len(inputs) == len(weights)  # Проверка равенства длины списков
        return sum(i * w for i, w in zip(inputs, weights))

    def matrix_vector_predict(self, vector, matrix):
        """
        Перемножение вектора на матрицу.
        Используем встроенную функцию numpy для оптимизации вычислений.
        """
        return np.dot(vector, matrix)

    def forward(self, input_vector):
        """
        Прямое распространение для нескольких слоев сети.
        Сначала вычисляем скрытый слой, затем окончательное предсказание.
        """
        hidden = np.dot(input_vector, self.weights[0])  # Перемножение входов и весов первого слоя
        prediction = np.dot(hidden, self.weights[1])    # Перемножение скрытого слоя и весов второго слоя
        return prediction

# Инициализация весов для разных типов сетей
weights_1d = 0.1  # Один вес для простейшего случая
weights_vector = [0.1, 0.2, 0.0]  # Веса для нескольких входов

# Веса для многослойной сети
ih_wgt = np.array([[0.1, 0.2, -0.1], [-0.1, 0.1, 0.9], [0.1, 0.4, 0.1]]).T
hp_wgt = np.array([[0.3, 1.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]).T
weights = [ih_wgt, hp_wgt]

# Пример входных данных
# toes — число пальцев у игроков
# wlrec — процент побед
# nfans — количество фанатов (в миллионах)
toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])
input_data = np.array([toes[0], wlrec[0], nfans[0]])  # Входные данные для первой игры

# Создаем экземпляр класса NeuralNetwork и выполняем предсказания
nn = NeuralNetwork(weights_1d)  # Нейронная сеть с одним весом
single_pred = nn.single_layer_predict(input_data[0])  # Предсказание для одного входа
print(f"Single Layer Prediction: {single_pred}")

nn_multi_input = NeuralNetwork(weights_vector)  # Нейронная сеть с несколькими весами
multi_pred = nn_multi_input.multiple_input_predict(input_data, weights_vector)  # Предсказание для нескольких входов
print(f"Multiple Input Prediction: {multi_pred}")

nn_matrix = NeuralNetwork(weights)  # Многослойная нейронная сеть
multi_layer_pred = nn_matrix.forward(input_data)  # Прямое распространение для многослойной сети
print(f"Multi-Layer Prediction: {multi_layer_pred}")

# Примеры операций с векторами и матрицами
# Создаем векторы для умножения
a = np.array([0, 1, 2, 3])
b = np.array([4, 5, 6, 7])
print(f"a * b: {a * b}")  # Поэлементное умножение векторов

# Примеры матриц
c = np.zeros((2, 4))  # Матрица из нулей размером 2x4
d = np.zeros((4, 3))  # Матрица из нулей размером 4x3
matrix_result = c.dot(d)  # Умножение матриц
print(f"Matrix dot product shape: {matrix_result.shape}")  # Размер результирующей матрицы

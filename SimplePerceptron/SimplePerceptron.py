def weighted_sum(inputs, weights):
    """
    Вычисление взвешенной суммы для входных данных и весов.
    """
    assert len(inputs) == len(weights), "Размеры входных данных и весов должны совпадать"
    return sum(i * w for i, w in zip(inputs, weights))


def neural_network(inputs, weights):
    """
    Простой нейрон с линейной функцией активации.
    """
    return weighted_sum(inputs, weights)


def scalar_multiply(scalar, vector):
    """
    Умножение вектора на скаляр.
    """
    return [scalar * element for element in vector]


def gradient_descent_example():
    """
    Демонстрация обучения нейронной сети с использованием градиентного спуска.
    """
    # Данные входов и истинные значения
    toes = [8.5, 9.5, 9.9, 9.0]
    wlrec = [0.65, 0.8, 0.8, 0.9]
    nfans = [1.2, 1.3, 0.5, 1.0]
    win_or_lose_binary = [1, 1, 0, 1]

    true_value = win_or_lose_binary[0]
    inputs = [toes[0], wlrec[0], nfans[0]]

    # Начальные веса и скорость обучения
    weights = [0.1, 0.2, -0.1]
    alpha = 0.01

    print("Градиентный спуск с несколькими итерациями:\n")
    for iteration in range(3):
        # Прогнозирование
        prediction = neural_network(inputs, weights)
        error = (prediction - true_value) ** 2
        delta = prediction - true_value

        # Обновление весов
        weight_deltas = scalar_multiply(delta, inputs)
        weights = [w - alpha * dw for w, dw in zip(weights, weight_deltas)]

        # Вывод промежуточных результатов
        print(f"Итерация {iteration + 1}")
        print(f"Прогноз: {prediction:.5f}, Ошибка: {error:.5f}, Дельта: {delta:.5f}")
        print(f"Веса: {weights}")
        print(f"Изменение весов: {weight_deltas}\n")

if __name__ == "__main__":
    gradient_descent_example()

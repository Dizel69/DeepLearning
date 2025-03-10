# Пример использования простой нейронной сети с ручным обновлением веса для минимизации ошибки

# Начальные параметры
knob_weight = 0.5  # начальный вес
input_value = 0.5  # входное значение
goal_pred = 0.8  # целевое значение предсказания

# Расчёт предсказания и ошибки
pred = input_value * knob_weight  # предсказание
error = (pred - goal_pred) ** 2  # квадратичная ошибка
print(f'Начальная ошибка: {error}')

# 1) Пустая сеть с одним весом
weight = 0.1  # начальное значение веса
lr = 0.01  # коэффициент обучения

# Простая нейросеть с одним входом и одним весом
def neural_network(input_val, weight_val):
    prediction = input_val * weight_val  # вычисление предсказания
    return prediction

# 2) Предсказание и оценка ошибки
number_of_toes = [8.5]  # пример данных: количество пальцев ног (условный пример)
win_or_lose_binary = [1]  # пример метки выигрыша или проигрыша

input_data = number_of_toes[0]
true_value = win_or_lose_binary[0]

pred = neural_network(input_data, weight)
error = (pred - true_value) ** 2
print(f'Ошибка для одного предсказания: {error}')

# 3) Сравнение предсказания с увеличением веса
p_up = neural_network(input_data, weight + lr)  # предсказание с увеличением веса
error_up = (p_up - true_value) ** 2  # ошибка при увеличении веса
print(f'Ошибка с увеличением веса: {error_up}')

# 4) Сравнение предсказания с уменьшением веса
p_down = neural_network(input_data, weight - lr)  # предсказание с уменьшением веса
error_down = (p_down - true_value) ** 2  # ошибка при уменьшении веса
print(f'Ошибка с уменьшением веса: {error_down}')

# Итеративное обновление веса с использованием маленьких шагов
weight = 0.5
input_value = 0.5
goal_prediction = 0.8
step_amount = 0.001

# Подбор веса методом поиска с шагом
for iteration in range(1101):
    prediction = input_value * weight
    error = (prediction - goal_prediction) ** 2
    print(f'Итерация {iteration} - Ошибка: {error}, Предсказание: {prediction}')

    # Проверка ошибки при увеличении и уменьшении веса
    up_prediction = input_value * (weight + step_amount)
    up_error = (goal_prediction - up_prediction) ** 2

    down_prediction = input_value * (weight - step_amount)
    down_error = (goal_prediction - down_prediction) ** 2

    # Обновление веса в сторону уменьшения ошибки
    if down_error < up_error:
        weight -= step_amount
    else:
        weight += step_amount

# Градиентный спуск с учётом величины шага
weight = 0.5
goal_pred = 0.8
input_value = 0.5

for iteration in range(20):
    pred = input_value * weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input_value  # направление и величина изменения веса
    weight -= direction_and_amount

    print(f'Итерация {iteration} - Ошибка: {error}, Предсказание: {pred}')

# Градиентный спуск с фиксированным коэффициентом обучения
alpha = 0.01
weight = 0.0

for iteration in range(4):
    pred = input_value * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input_value
    weight -= weight_delta * alpha
    print(f'Итерация {iteration} - Ошибка: {error}, Предсказание: {pred}')

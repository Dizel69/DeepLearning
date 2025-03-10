import numpy as np
import matplotlib.pyplot as plt

# Функция для отрисовки линии на графике
# slope - угловой коэффициент (наклон), y_intercept - точка пересечения с осью Y
# color - цвет линии, linewidth - толщина линии, starting и ending - диапазон значений X

def draw_line(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)  # Генерируем 1000 значений X в указанном диапазоне
    plt.plot(x, y_intercept + slope * x, linestyle='-', color=color, linewidth=linewidth)


# Функция для построения точек на графике (разброс значений)
# features - количество комнат, labels - соответствующие цены
def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    plt.scatter(X, y)
    plt.xlabel('Количество комнат')
    plt.ylabel('Цена')

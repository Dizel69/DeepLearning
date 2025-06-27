import numpy as np
import matplotlib.pyplot as plt

# === Новые функции с понятными именами ===

def scatter_labels(features, labels):
    """
    Строит на графике точки двух классов (метки 0 и 1).
    """
    coords = np.array(features)
    tags   = np.array(labels)
    pos = coords[tags == 1]
    neg = coords[tags == 0]
    if pos.size:
        plt.scatter(pos[:,0], pos[:,1], s=100, marker='^', edgecolor='k', facecolor='cyan', label='class 1')
    if neg.size:
        plt.scatter(neg[:,0], neg[:,1], s=100, marker='s', edgecolor='k', facecolor='red',  label='class 0')
    plt.xlabel('aack')
    plt.ylabel('beep')
    plt.legend()

def plot_linear(a, b, c, x_start=0, x_end=3, **kwargs):
    """
    Рисует прямую a*x + b*y + c = 0 на указанном интервале X.
    """
    x = np.linspace(x_start, x_end, 500)
    y = -(a*x + c)/b
    plt.plot(x, y, **kwargs)

# === Обратная совместимость со старыми вызовами ===

# Старое имя plot_points → scatter_labels
plot_points = scatter_labels

# Старое имя draw_line → plot_linear
draw_line   = plot_linear

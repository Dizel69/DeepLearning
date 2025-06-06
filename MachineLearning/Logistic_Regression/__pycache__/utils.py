import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Модуль utils.py
# Вспомогательные функции для визуализации точек и разделяющих прямых
# ================================================================

# Функция scatter_labels: отображает точки двух классов (0 и 1)
def scatter_labels(features, labels):
    pts = np.array(features)
    lbs = np.array(labels)
    class1 = pts[lbs == 1]
    class0 = pts[lbs == 0]
    if class1.size:
        plt.scatter(
            class1[:,0], class1[:,1],
            s=100, marker='^', edgecolor='k', facecolor='cyan',
            label='happy (1)'
        )
    if class0.size:
        plt.scatter(
            class0[:,0], class0[:,1],
            s=100, marker='s', edgecolor='k', facecolor='red',
            label='sad (0)'
        )
    plt.xlabel('aack')
    plt.ylabel('beep')
    plt.legend()

# Функция plot_linear_decision: строит прямую a*x + b*y + c = 0
def plot_linear(a, b, c, color='black', linewidth=2.0, linestyle='solid', starting=0, ending=3):
    x_vals = np.linspace(starting, ending, 1000)
    y_vals = -(a * x_vals + c) / b
    plt.plot(x_vals, y_vals, color=color, linewidth=linewidth, linestyle=linestyle)

# Обратная совместимость: старые имена plot_points и draw_line
plot_points = scatter_labels
draw_line   = plot_linear

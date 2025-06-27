import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# utils.py
# Вспомогательные функции для визуализации точек, классификаторов,
# деревьев решений и регрессоров.
# ================================================================

# === Функция scatter_points (раньше plot_points) ===
def scatter_points(features, labels, size=100):
    """
    Рисует точки двух классов на плоскости:
    - класс 1 отображается треугольниками cyan;
    - класс 0 отображается квадратами red.

    features: array-like (n_samples, 2)
    labels:   array-like (n_samples,), значения 0 или 1
    size:     размер маркеров
    """
    X = np.asarray(features)
    y = np.asarray(labels)
    pos = X[y == 1]
    neg = X[y == 0]
    if pos.size:
        plt.scatter(pos[:,0], pos[:,1], s=size, marker='^', edgecolor='k', facecolor='cyan', label='class 1')
    if neg.size:
        plt.scatter(neg[:,0], neg[:,1], s=size, marker='s', edgecolor='k', facecolor='red', label='class 0')
    plt.legend()

# alias для обратной совместимости
plot_points = scatter_points


# === Функция plot_classifier_decision_boundary (раньше plot_model) ===
def plot_classifier_decision_boundary(features, labels, model, size=100, step=0.2):
    """
    Визуализирует разделяющие регионы обученного классификатора на 2D признаках.

    features: array-like (n_samples, 2)
    labels:   array-like (n_samples,)
    model:    обученный классификатор с методом predict()
    size:     размер маркеров
    step:     шаг сетки для вычисления поверхности решений
    """
    X = np.asarray(features)
    y = np.asarray(labels)
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.contour(xx, yy, Z, colors='k', linewidths=1)
    scatter_points(X, y, size)
    plt.show()

# alias для обратной совместимости
plot_model = plot_classifier_decision_boundary


# === Функция show_tree (раньше display_tree) ===
def show_tree(decision_tree_model):
    """
    Визуализация структуры DecisionTreeClassifier без Graphviz.
    Использует matplotlib и sklearn.tree.plot_tree.
    """
    from sklearn.tree import plot_tree
    plt.figure(figsize=(8,6))
    plot_tree(decision_tree_model, filled=True, rounded=True, feature_names=getattr(decision_tree_model, 'feature_names_in_', None), class_names=True)
    plt.show()

# alias для обратной совместимости
display_tree = show_tree


# === Функция plot_regression_line ===
def plot_regression_line(model, X, y):
    """
    Строит линию регрессии для одномерного признака.

    model: регрессор с методом predict()
    X: array-like (n_samples,) или (n_samples,1)
    y: истинные значения целевой переменной
    """
    X = np.ravel(X)
    x_vals = np.linspace(X.min(), X.max(), 500)
    y_vals = model.predict(x_vals.reshape(-1,1))
    plt.scatter(X, y)
    plt.plot(x_vals, y_vals)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.show()

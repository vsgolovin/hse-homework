from typing import Callable
import math
import numpy as np
import matplotlib.pyplot as plt
from plot_settings import settings


def main():
    plt.rcParams.update(settings)

    data = np.array([[1, 4], [4, 0], [5, 5], [8, 0]])
    bezier = get_bezier_function(data)
    t = np.linspace(0, 1)
    bezier_curve = np.array([bezier(ti) for ti in t])
    x = np.polyval([4, -6, 9, 1], t)
    y = np.polyval([-19, 27, -12, 4], t)

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'b-')
    plt.plot(x, y, 'g--')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('figures/ex4.pdf')


def get_bezier_function(points: np.ndarray) -> Callable:
    n = len(points)
    coeffs = np.array([math.comb(n - 1, k) * points[k] for k in range(n)])

    def bezier(t):
        terms = np.array([(1 - t)**(n - 1 - i) * t**i
                          for i in range(n)]).reshape((-1, 1))
        return np.sum(terms * coeffs, axis=0)
    return bezier


if __name__ == '__main__':
    main()

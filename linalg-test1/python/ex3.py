import numpy as np
import matplotlib.pyplot as plt
from plot_settings import settings


def main():
    plt.rcParams.update(settings)

    x = np.array([-2, -1, 0, 1])
    y = np.array([-10, -10, 16, -20])
    p = np.array([-44/3, -31, 29/3, 16])
    x_curve = np.linspace(-2.5, 1.5)
    y_curve = np.polyval(p, x_curve)

    plt.figure()
    plt.plot(x, y, 'ro')
    plt.plot(x_curve, y_curve, 'b-')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('figures/ex3.pdf')


if __name__ == '__main__':
    main()

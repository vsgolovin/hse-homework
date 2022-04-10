import numpy as np
import matplotlib.pyplot as plt
from plot_settings import settings


def main():
    plt.rcParams.update(settings)

    x = np.linspace(0, 4)
    f = np.polyval([1, -4, 4, -5], x)
    g = np.polyval([2, -5, -3], x)

    plt.figure()
    plt.plot(x, f, label=r'$f(x)$')
    plt.plot(x, g, label=r'$g(x)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.savefig('figures/ex5.pdf')


if __name__ == '__main__':
    main()

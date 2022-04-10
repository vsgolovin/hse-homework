import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import chebyt
from plot_settings import settings


def h1(x):
    return np.sqrt(5 * x + 5)


def h2(u):
    return np.sqrt(15 * u + 20)


def main():
    u = np.linspace(-1, 1, 1000)[1:-1]
    f = h2(u)

    # calculate Chebyshev approximation coefficients
    T_arrays = np.zeros((4, len(u)))
    alphas = np.zeros(4)
    denom = np.sqrt(1 - u**2)
    for i in range(4):
        T_arrays[i] = np.polyval(chebyt(i), u)
        alphas[i] = simps(f * T_arrays[i] / denom, u) * 2 / np.pi
    alphas[0] /= 2
    print(alphas)

    # reconstruct g(x) from Chebyshev polynomials
    g = np.sum(alphas[:, np.newaxis] * T_arrays, axis=0)
    x = 3 * u + 3
    g_x = np.polyval([-0.0151627, 0.0205017, 0.9370461, 2.0590848], x)
    assert np.allclose(g, g_x, atol=1e-4)

    # plot results
    plt.rcParams.update(settings)
    plt.figure()
    plt.plot(x, f, label=r'$f(x)$')
    plt.plot(x, g, label=r'$g(x)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.savefig('figures/ex7.pdf')


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from test_functions import test_functions
from methods import minimize_ING


def main():
    for i, tf in enumerate(test_functions):
        a, b = tf.interval
        eps = 1e-4 * (b - a)
        res = minimize_ING(tf.f, a, b, r=2.0, atol=eps, full_output=True)
        x = np.linspace(a, b, 500)
        y = tf.f(x)

        plt.figure()
        plt.plot(x, y)
        plt.plot(res.x, res.y, 'r.')
        plt.title(f'x_min = {res.x_min:.3f}, nfev = {res.nfev}')
        plt.savefig(f'plots/{i + 1}.png', dpi=150)
        plt.close()


if __name__ == '__main__':
    main()

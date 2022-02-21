from collections import namedtuple
import numpy as np


def lipschitz_constant(f, a, b, npoints=50000):
    dx = (b - a) / npoints
    x = np.arange(a, b + dx / 2, dx)
    y = np.real(f(x))
    dy_dx = (y[2:] - y[:-2]) / (2 * dx)
    return np.max(np.abs(dy_dx))


TestFunction = namedtuple('TestFunction', ['f', 'interval', 'L'])

def f_01(x):
    return (1/6*x**6 - 52/25*x**5 + 39/80*x**4
            + 71/10*x**3 - 79/20*x**2 - x + 0.1)

def f_02(x):
    return np.sin(x) + np.sin(10 * x / 3)

def f_03(x):
    return -sum(k * np.sin((k + 1) * x + k) for k in range(1, 6))

def f_04(x):
    return -(16 * x**2 - 24 * x + 5) * np.exp(-x)

def f_05(x):
    return (3 * x - 1.4) * np.sin(18 * x)

def f_06(x):
    return -(x + np.sin(x)) * np.exp(-x**2)

def f_07(x):
    return np.sin(x) + np.sin(10 * x / 3) + np.log(x) - 0.84 * x + 3

def f_08(x):
    return -sum(k * np.cos((k + 1) * x + k) for k in range(1, 6))

def f_09(x):
    return np.sin(x) + np.sin(2 * x / 3)

def f_10(x):
    return -x * np.sin(x)

def f_11(x):
    return 2 * np.cos(x) + np.cos(2 * x)

def f_12(x):
    return np.sin(x)**3 + np.cos(x)**3

def f_13(x):
    return -(x**(2/3)) + np.cbrt(x**2 - 1)

def f_14(x):
    return -np.exp(-x) * np.sin(2 * np.pi * x)

def f_15(x):
    return (x**2 - 5 * x + 6) / (x**2 + 1)

def f_16(x):
    return 2 * (x - 3)**2 + np.exp(0.5 * x**2)

def f_17(x):
    return x**6 - 15 * x**4 + 27 * x**2 + 250

@np.vectorize
def f_18(x):
    if x <= 3:
        return (x - 2)**2
    return 2 * np.log(x - 2) + 1

def f_19(x):
    return -x + np.sin(3 * x) - 1

def f_20(x):
    return (np.sin(x) - x) * np.exp(-x**2)


test_functions = (
    TestFunction(f_01, (-1.5, 11), 13865),
    TestFunction(f_02, (2.7, 7.5), 4.3),
    TestFunction(f_03, (-10, 10), 68.42),
    TestFunction(f_04, (1.9, 3.9), 2.94),
    TestFunction(f_05, (0, 1.2), 35.5),
    TestFunction(f_06, (-10, 10), 2.0),
    TestFunction(f_07, (2.7, 7.5), 4.8),
    TestFunction(f_08, (-10, 10), 69.5),
    TestFunction(f_09, (3.1, 20.4), 1.67),
    TestFunction(f_10, (0, 10), 9.65),
    TestFunction(f_11, (-1.57, 6.28), 3.53),
    TestFunction(f_12, (0, 6.28), 2.13),
    TestFunction(f_13, (0.001, 0.99), 8.31),
    TestFunction(f_14, (0, 4), 6.29),
    TestFunction(f_15, (-5, 5), 6.38),
    TestFunction(f_16, (-3, 3), 294),
    TestFunction(f_17, (-4, 4), 2520),
    TestFunction(f_18, (0, 6), 4.0),
    TestFunction(f_19, (0, 6.5), 4.0),
    TestFunction(f_20, (-10, 10), 0.1)
)


if __name__ == '__main__':
    for i, tf in enumerate(test_functions):
        print(i + 1, tf.L, lipschitz_constant(tf.f, *tf.interval))

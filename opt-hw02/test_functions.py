from collections import namedtuple
import numpy as np


def lipschitz_constant(f, a, b, npoints=50000):
    dx = (b - a) / npoints
    x = np.arange(a, b + dx / 2, dx)
    y = np.real(f(x))
    dy_dx = (y[2:] - y[:-2]) / (2 * dx)
    return np.max(np.abs(dy_dx))


TestFunction = namedtuple('TestFunction', ['f', 'fdot', 'interval', 'L'])


def f_01(x):
    return (1/6*x**6 - 52/25*x**5 + 39/80*x**4
            + 71/10*x**3 - 79/20*x**2 - x + 0.1)

def fdot_01(x):
    return x**5 - 10.4 * x**4 + 1.95 * x**3 + 21.3 * x**2 - 7.9 * x - 1

def f_02(x):
    return np.sin(x) + np.sin(10 * x / 3)

def fdot_02(x):
    return np.cos(x) + 10 * np.cos(10 * x / 3) / 3

def f_03(x):
    return -sum(k * np.sin((k + 1) * x + k) for k in range(1, 6))

def fdot_03(x):
    return -sum(k * (k + 1) * np.cos((k + 1) * x + k) for k in range(1, 6))

def f_04(x):
    return -(16 * x**2 - 24 * x + 5) * np.exp(-x)

def fdot_04(x):
    return (16 * x**2 - 56 * x + 29) * np.exp(-x)

def f_05(x):
    return (3 * x - 1.4) * np.sin(18 * x)

def fdot_05(x):
    return 3 * np.sin(18 * x) + (3 * x - 1.4) * 18 * np.cos(18 * x)

def f_06(x):
    return -(x + np.sin(x)) * np.exp(-x**2)

def fdot_06(x):
    return (2 * x * (x + np.sin(x)) - np.cos(x) - 1) * np.exp(-x**2)

def f_07(x):
    return np.sin(x) + np.sin(10 * x / 3) + np.log(x) - 0.84 * x + 3

def fdot_07(x):
    return np.cos(x) + 10 / 3 * np.cos(10 * x / 3) + 1 / x - 0.84

def f_08(x):
    return -sum(k * np.cos((k + 1) * x + k) for k in range(1, 6))

def fdot_08(x):
    return -sum(-k * (k + 1) * np.sin((k + 1) * x + k) for k in range(1, 6))

def f_09(x):
    return np.sin(x) + np.sin(2 * x / 3)

def fdot_09(x):
    return np.cos(x) + 2 / 3 * np.cos(2 * x / 3)

def f_10(x):
    return -x * np.sin(x)

def fdot_10(x):
    return -np.sin(x) - x * np.cos(x)

def f_11(x):
    return 2 * np.cos(x) + np.cos(2 * x)

def fdot_11(x):
    return -2 * np.sin(x) - 2 * np.sin(2 * x)

def f_12(x):
    return np.sin(x)**3 + np.cos(x)**3

def fdot_12(x):
    a, b = np.sin(x), np.cos(x)
    return 3 * a * b * (a - b)

def f_13(x):
    return -(x**(2/3)) + np.cbrt(x**2 - 1)

def fdot_13(x):
    return -2 / (3 * np.cbrt(x)) + 2 * x / (3 * np.cbrt((x**2 - 1)**2))

def f_14(x):
    return -np.exp(-x) * np.sin(2 * np.pi * x)

def fdot_14(x):
    return np.exp(-x) * (np.sin(2 * np.pi * x)
                         - 2 * np.pi * np.cos(2 * np.pi * x))

def f_15(x):
    return (x**2 - 5 * x + 6) / (x**2 + 1)

def fdot_15(x):
    return 5 * (x**2 - 2 * x - 1) / (x**4 + 2 * x**2 + 1)

def f_16(x):
    return 2 * (x - 3)**2 + np.exp(0.5 * x**2)

def fdot_16(x):
    return 4 * (x - 3) + x * np.exp(0.5 * x**2)

def f_17(x):
    return x**6 - 15 * x**4 + 27 * x**2 + 250

def fdot_17(x):
    return 6 * x**5 - 60 * x**3 + 54 * x

@np.vectorize
def f_18(x):
    if x <= 3:
        return (x - 2)**2
    return 2 * np.log(x - 2) + 1

@np.vectorize
def fdot_18(x):
    if x <= 3:
        return 2 * (x - 2)
    return 2 / (x - 2)

def f_19(x):
    return -x + np.sin(3 * x) - 1

def fdot_19(x):
    return -1 + 3 * np.cos(3 * x)

def f_20(x):
    return (np.sin(x) - x) * np.exp(-x**2)

def fdot_20(x):
    return np.exp(-x**2) * (np.cos(x) - 1 - 2 * x * (np.sin(x) - x))


test_functions = (
    TestFunction(f_01, fdot_01, (-1.5, 11), 13865),
    TestFunction(f_02, fdot_02, (2.7, 7.5), 4.3),
    TestFunction(f_03, fdot_03, (-10, 10), 68.42),
    TestFunction(f_04, fdot_04, (1.9, 3.9), 2.94),
    TestFunction(f_05, fdot_05, (0, 1.2), 35.5),
    TestFunction(f_06, fdot_06, (-10, 10), 2.0),
    TestFunction(f_07, fdot_07, (2.7, 7.5), 4.8),
    TestFunction(f_08, fdot_08, (-10, 10), 69.5),
    TestFunction(f_09, fdot_09, (3.1, 20.4), 1.67),
    TestFunction(f_10, fdot_10, (0, 10), 9.65),
    TestFunction(f_11, fdot_11, (-1.57, 6.28), 3.53),
    TestFunction(f_12, fdot_12, (0, 6.28), 2.13),
    TestFunction(f_13, fdot_13, (0.001, 0.99), 8.31),
    TestFunction(f_14, fdot_14, (0, 4), 6.29),
    TestFunction(f_15, fdot_15, (-5, 5), 6.38),
    TestFunction(f_16, fdot_16, (-3, 3), 294),
    TestFunction(f_17, fdot_17, (-4, 4), 2520),
    TestFunction(f_18, fdot_18, (0, 6), 4.0),
    TestFunction(f_19, fdot_19, (0, 6.5), 4.0),
    TestFunction(f_20, fdot_20, (-10, 10), 0.1)
)


if __name__ == '__main__':
    for i, tf in enumerate(test_functions):
        print(i + 1, tf.L, lipschitz_constant(tf.f, *tf.interval))

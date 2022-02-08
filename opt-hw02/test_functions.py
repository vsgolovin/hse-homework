from math import sin


def f1(x):
    return (1/6 * x**6 - 52/25 * x**5 + 39/80 * x**4 + 71/10 * x**3
            - 79/20 * x**2 - x + 0.1)

def f2(x):
    return sin(x) + sin(10/3 * x)


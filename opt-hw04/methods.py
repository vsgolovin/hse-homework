import numpy as np
import test_functions as tf


def main():
    x0 = [0, 0]
    sol = dfp(tf.rosenbrock, tf.rosenbrock_derivatives, x0,
              h_method='min', grad_min=1e-4, full_output=True)
    print(len(sol))
    print(sol[-1])


def golden_section_search(f, a, b, atol=None):
    """
    Найти минимум функции `f` на интервале [`a`, `b`] с точностью `atol`
    методом золотого сечения. Точность по умолчанию `1e-4 * (b - a)`.
    """
    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4
    # коэффициент для разбиения отрезка
    phi = (np.sqrt(5) - 1) / 2

    # начальное разбиение
    c = b - (b - a) * phi
    f_c = f(c)
    d, f_d = None, None

    while (b - a) > atol:
        # новая точка
        if c is None:
            c = b - (b - a) * phi
            f_c = f(c)
        else:
            assert d is None
            d = a + (b - a) * phi
            f_d = f(d)

        # выбор нового интервала
        if f_c < f_d:
            b, d = d, c
            f_d = f_c
            c = None
        else:
            a, c = c, d
            f_c = f_d
            d = None

    return a + (b - a) / 2


def choose_h(f, x, grad, h_method, atol=None):
    if h_method == 'min':
        # поиск h из условия минимума
        h = 1.0
        f_x = f(*x)
        while True:
            h = golden_section_search(
                lambda h: f(*(x_i - h * grad_x_i
                            for x_i, grad_x_i in zip(x, grad))),
                a=0,
                b=h,
                atol=atol
            )
            if f(*(x - h * grad)) < f_x:
                break
    elif isinstance(h_method, (int, float)):
        # постоянное значение h
        h = h_method
    else:
        # поиск h через двойное неравенство
        assert len(h_method) == 2
        alpha = h_method[0]
        beta = h_method[1]
        h_min, h_max = 0, 1
        grad_norm = np.linalg.norm(grad)
        f_x = f(*x)
        h = 1

        # двоичный поиск
        if atol is None:
            atol = 1e-4
        while h_max - h_min > atol:
            decrease = f_x - f(*(x - grad * h))
            dot_product = grad_norm**2 * h
            if alpha * dot_product > decrease:
                h_max = h
            elif beta * dot_product < decrease:
                h_min = h
            else:
                break
            h = (h_min + h_max) / 2

    return h


def broyden(f, fdot, x0, h_method=1e-3, grad_min=1e-4, restart_period=1,
            maxiter=10000, full_output=True):
    # инициализация
    n = len(x0)                      # размерность пространства
    x = np.array(x0, dtype='float')  # текущее решение
    H = np.eye(n)                    # (аппроксимация Гессиана)^-1
    grad = np.array(fdot(*x))        # градиент
    solutions = [x.copy()]           # все решения
    num_iterations = 0               # число итераций после рестарта

    while np.linalg.norm(grad) >= grad_min:
        # рестарт
        if num_iterations == n * restart_period:
            H = np.eye(n)
            num_iterations = 0

        delta = np.dot(H, grad)
        h = choose_h(f, x, delta, h_method)

        # обновляем решение
        delta *= -h
        x += delta
        solutions.append(x.copy())
        g_new = np.array(fdot(*x))
        gamma = g_new - grad
        grad = g_new

        # обновляем H
        residual = delta - np.dot(H, gamma)
        H += np.outer(residual, residual) / np.dot(residual, gamma)
        num_iterations += 1

        # проверяем полное число итераций
        if len(solutions) >= maxiter:
            raise Exception(
                f'Решение не сошлось после {maxiter} итераций.')

    if full_output:
        return solutions
    return x


def dfp(f, fdot, x0, h_method=1.0, grad_min=1e-4, maxiter=10000,
        full_output=True):
    # инициализация
    n = len(x0)                      # размерность пространства
    x = np.array(x0, dtype='float')  # текущее решение
    H = np.eye(n)                    # (аппроксимация Гессиана)^-1
    grad = np.array(fdot(*x))        # градиент
    solutions = [x.copy()]           # все решения

    while np.linalg.norm(grad) >= grad_min:
        delta = np.dot(H, grad)
        h = choose_h(f, x, delta, h_method)

        # обновляем решение
        delta *= -h
        x += delta
        solutions.append(x.copy())
        g_new = np.array(fdot(*x))
        gamma = g_new - grad
        grad = g_new

        # обновляем H
        u = np.dot(H, gamma)
        H += (1 / np.dot(gamma, delta) * np.outer(delta, delta)
              - 1 / np.dot(u, gamma) * np.outer(np.dot(H, gamma), u))

        # проверяем полное число итераций
        if len(solutions) >= maxiter:
            raise Exception(
                f'Решение не сошлось после {maxiter} итераций.')

    if full_output:
        return solutions
    return x


def bfgs(f, fdot, x0, h_method=1.0, grad_min=1e-4, maxiter=10000,
         full_output=True):
    # инициализация
    n = len(x0)                      # размерность пространства
    x = np.array(x0, dtype='float')  # текущее решение
    H = np.eye(n)                    # (аппроксимация Гессиана)^-1
    grad = np.array(fdot(*x))        # градиент
    solutions = [x.copy()]           # все решения

    while np.linalg.norm(grad) >= grad_min:
        delta = np.dot(H, grad)
        h = choose_h(f, x, delta, h_method)

        # обновляем решение
        delta *= -h
        x += delta
        solutions.append(x.copy())
        g_new = np.array(fdot(*x))
        gamma = g_new - grad
        grad = g_new

        # обновляем H
        u = np.dot(H, gamma)
        m = np.dot(delta, gamma)
        G = np.outer(u, delta)
        H += ((m + np.dot(gamma, u)) / m**2 * np.outer(delta, delta)
              - (G + G.T) / m)

        # проверяем полное число итераций
        if len(solutions) >= maxiter:
            raise Exception(
                f'Решение не сошлось после {maxiter} итераций.')

    if full_output:
        return solutions
    return x


if __name__ == '__main__':
    main()

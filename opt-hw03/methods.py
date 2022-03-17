import numpy as np


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


def gradient_descent(f, grad_f, x0, h_method='min', grad_min=1e-4,
                     return_solutions=False):
    """
    Найти минимум функции `f` с градентом `grad_f` с помощью метода
    градиентного спуска. Параметр `h_method` определяет способ выбора
    коэффициента `h` (x <- x - h * grad_f).
    """
    x = np.array(x0, dtype='float64')  # текущее решение
    solutions = [x.copy()]             # все решения
    grad = np.array(grad_f(*x))        # градиент в текущей точке
    grad_norm = np.linalg.norm(grad)
    f_x = f(*x)

    # обновляем ответ, пока норма градиента не станет достаточно малой
    while grad_norm > grad_min:
        if h_method == 'min':
            # поиск h из условия минимума
            h = 1.0
            while True:
                h = golden_section_search(
                    lambda h: f(*(x_i - h * grad_x_i
                                for x_i, grad_x_i in zip(x, grad))),
                    a=0,
                    b=h
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
            h = 1

            # двоичный поиск
            while True:
                decrease = f_x - f(*(x - grad * h))
                dot_product = grad_norm**2 * h
                if alpha * dot_product > decrease:
                    h_max = h
                elif beta * dot_product < decrease:
                    h_min = h
                else:
                    break
                h = h_min + (h_max - h_min) / 2

        x -= h * grad
        f_x = f(*x)
        grad = np.array(grad_f(*x))
        grad_norm = np.linalg.norm(grad)
        solutions.append(x.copy())

    if return_solutions:
        return solutions
    return x

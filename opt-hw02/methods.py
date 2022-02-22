from collections import namedtuple


OptimizeResult = namedtuple('OptimizeResult', ['x_min', 'x', 'y', 'nfev'])
MAXFEV = 3000


def minimize_NA(f, a, b, L, atol=None, maxfev=MAXFEV, full_output=False):
    """
    Найти на на отрезке [`a`, `b`] минимум функции `f` с помощью метода
    ломаных (недифференцируемая целевая функция, априорно заданная оценка
    константы Липшица `L`).
    """
    def new_point(i):
        return (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)

    def characteristic(i):
        return (y[i] + y[i + 1]) / 2 - L * (x[i + 1] - x[i]) / 2

    # точность по умолчанию
    if atol is None:
        atol = 1e-4 * (b - a)

    x = [a, b]               # точки разбиения интервала
    y = [f(a), f(b)]         # значения функции в этих точках
    x_m = [new_point(0)]     # точки внутри интервалов
    F = [characteristic(0)]  # характеристики интервалов

    while True:
        # найдём интервал с минимальной характеристикой
        i_min = 0  # номер интервала
        F_min = F[0]
        for i in range(1, len(F)):
            if F[i] < F_min:
                i_min = i
                F_min = F[i]
        i = i_min

        # проверим условие остановки по точности
        if x[i + 1] - x[i] <= atol:
            break

        # проверяем условие остановки по числу итераций
        if len(y) > maxfev:  # слишком много итераций -- останавливаем расчёт
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')

        # условие не выполнено => разбиваем интервал на 2 части
        x.insert(i + 1, x_m[i])
        y.insert(i + 1, f(x_m[i]))

        # меняем характеристику старого интервала на характеристики двух новых
        x_m[i] = new_point(i)
        x_m.insert(i + 1, new_point(i + 1))
        F[i] = characteristic(i)
        F.insert(i + 1, characteristic(i + 1))

    # возвращаем результат
    x_min = x[i] if y[i] < y[i + 1] else x[i + 1]
    if not full_output:
        return x_min
    x_full = []
    y_full = []
    for i in range(len(y) - 1):
        x_full.append(x[i])
        x_full.append(x_m[i])
        y_full.append(y[i])
        y_full.append(F[i])
    x_full.append(x[-1])
    y_full.append(y[-1])
    return OptimizeResult(x_min, x_full, y_full, len(y))


def minimize_NG(f, a, b, r, atol=None, maxfev=MAXFEV, full_output=False):
    """
    Найти на отрезке [`a`, `b`] минимум функции `f` с помощью геометрического
    метода с адаптивным оцениваением глобальной константы Липшица. Точность
    метода и максимальное количество вызовов функции задаются параметрами
    `atol` и `maxfev`, соответственно.
    """
    def new_point(i):
        return (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)

    def characteristic(i):
        return (y[i] + y[i + 1]) / 2 - L * (x[i + 1] - x[i]) / 2

    def difference(i):
        return abs(y[i + 1] - y[i]) / (x[i + 1] - x[i])

    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4

    x = [a, b]                     # границы интервалов
    y = [f(a), f(b)]               # значения функции в x
    diff = [difference(0)]         # абсолютные приращения

    # начальная оценка константы Липшица
    H = diff[0]
    L = 1 if H < 1e-7 else H * r
    F = [characteristic(0)]

    while True:
        # выбор подынтервала
        i_best = 0
        F_best = F[0]
        for i in range(1, len(F)):
            if F[i] < F_best:
                i_best = i
                F_best = F[i]
        i = i_best

        # условия остановки
        if x[i + 1] - x[i] <= atol:
            break
        if len(y) >= maxfev:
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')

        # новое испытание
        x_new = new_point(i)
        x.insert(i + 1, x_new)
        y.insert(i + 1, f(x_new))
        diff[i] = difference(i)
        diff.insert(i + 1, difference(i + 1))

        # оценка глобальной константы Липшица
        k = i if diff[i] > diff[i + 1] else i + 1
        if diff[k] > H:  # обновляем все значения F
            H = diff[k]
            L = 1 if H < 1e-7 else H * r
            F = [characteristic(i) for i in range(len(y) - 1)]
        else:  # обновляем F только на новых интервалах
            F[i] = characteristic(i)
            F.insert(i + 1, characteristic(i + 1))

    # найдём минимум
    i_min = 0
    y_min = y[0]
    for i, yi in enumerate(y):
        if yi < y_min:
            i_min = i
            y_min = yi
    x_min = x[i_min]

    # возвращаем результат
    if not full_output:
        return x_min
    return OptimizeResult(x_min, x, y, len(y))


def minimize_ING(f, a, b, r, atol=None, maxfev=MAXFEV, full_output=False):
    """
    Найти на отрезке [`a`, `b`] минимум функции `f` с помощью информационно-
    статистического метода с адаптивным оцениваением глобальной константы
    Липшица. Точность метода и максимальное количество вызовов функции задаются
    параметрами `atol` и `maxfev`, соответственно.
    """
    def characteristic(i):
        return (mu * (x[i + 1] - x[i])
                + ((y[i + 1] - y[i])**2) / (mu * (x[i + 1] - x[i]))
                - 2 * (y[i + 1] + y[i]))

    def difference(i):
        return abs(y[i + 1] - y[i]) / (x[i + 1] - x[i])

    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4

    x = [a, b]
    y = [f(a), f(b)]
    diff = [difference(0)]

    # начальная оценка глобальной константы Липшица
    H = diff[0]
    mu = 1 if H < 1e-7 else H * r
    R = [characteristic(0)]

    while True:
        # выбор подынтервала
        i_best = 0
        R_best = R[0]
        for i in range(1, len(R)):
            if R[i] > R_best:
                i_best = i
                R_best = R[i]
        i = i_best

        # условия остановки
        if x[i + 1] - x[i] <= atol:
            break
        if len(y) >= maxfev:
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')

        # новое испытание
        x_new = (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * mu)
        x.insert(i + 1, x_new)
        y.insert(i + 1, f(x_new))
        diff[i] = difference(i)
        diff.insert(i + 1, difference(i + 1))

        # оценка глобальной константы Липшица
        k = i if diff[i] > diff[i + 1] else i + 1
        if diff[k] > H:  # обновляем все значения R
            H = diff[k]
            mu = 1 if H < 1e-7 else H * r
            R = [characteristic(i) for i in range(len(y) - 1)]
        else:  # обновляем R только на новых интервалах
            R[i] = characteristic(i)
            R.insert(i + 1, characteristic(i + 1))

    # найдём минимум
    i_min = 0
    y_min = y[0]
    for i, yi in enumerate(y):
        if yi < y_min:
            i_min = i
            y_min = yi
    x_min = x[i_min]

    # возвращаем результат
    if not full_output:
        return x_min
    return OptimizeResult(x_min, x, y, len(y))


def minimize_ING2(f, a, b, r, atol=None, maxfev=MAXFEV, full_output=False):
    """
    Найти на отрезке [`a`, `b`] минимум функции `f` с помощью информационно-
    статистического метода с адаптивным оцениваением глобальной константы
    Липшица. Точность метода и максимальное количество вызовов функции задаются
    параметрами `atol` и `maxfev`, соответственно.
    """
    def characteristic(i):
        return -4 * ((y[i] + y[i + 1]) / 2 - L * (x[i + 1] - x[i]) / 2)

    def difference(i):
        return abs(y[i + 1] - y[i]) / (x[i + 1] - x[i])

    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4

    x = [a, b]
    y = [f(a), f(b)]
    diff = [difference(0)]

    # начальная оценка глобальной константы Липшица
    H = diff[0]
    mu = 1 if H < 1e-7 else H * r
    L = 0.5 * (mu + 1 / mu * ((y[1] - y[0]) / (b - a))**2)
    R = [characteristic(0)]

    while True:
        # выбор подынтервала
        i_best = 0
        R_best = R[0]
        for i in range(1, len(R)):
            if R[i] > R_best:
                i_best = i
                R_best = R[i]
        i = i_best

        # условия остановки
        if x[i + 1] - x[i] <= atol:
            break
        if len(y) >= maxfev:
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')

        # новое испытание
        x_new = (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)
        x.insert(i + 1, x_new)
        y.insert(i + 1, f(x_new))
        diff[i] = difference(i)
        diff.insert(i + 1, difference(i + 1))

        # оценка глобальной константы Липшица
        k = i if diff[i] > diff[i + 1] else i + 1
        if diff[k] > H:  # обновляем все значения R
            H = diff[k]
            mu = 1 if H < 1e-7 else H * r
            L = 0.5 * (mu
                       + 1 / mu * ((y[k + 1] - y[k]) / (x[k + 1] - x[k]))**2)
            R = [characteristic(i) for i in range(len(y) - 1)]
        else:  # обновляем R только на новых интервалах
            R[i] = characteristic(i)
            R.insert(i + 1, characteristic(i + 1))

    # найдём минимум
    i_min = 0
    y_min = y[0]
    for i, yi in enumerate(y):
        if yi < y_min:
            i_min = i
            y_min = yi
    x_min = x[i_min]

    # возвращаем результат
    if not full_output:
        return x_min
    return OptimizeResult(x_min, x, y, len(y))


def minimize_NL(f, a, b, r, xi, atol=None, maxfev=MAXFEV, full_output=False):
    """
    Найти на отрезке [`a`, `b`] минимум функции `f` с помощью геометрического
    метода с адаптивым оцениваением локальных констант Липшица. Точность метода
    и максимальное количество вызовов функции задаются параметрами `atol` и
    `maxfev`, соответственно.
    """
    def new_point(i):
        return (x[i + 1] + x[i]) / 2 - (y[i + 1] - y[i]) / (2 * mu[i])

    def characteristic(i):
        return (y[i + 1] + y[i]) / 2 - mu[i] * (x[i + 1] - x[i]) / 2

    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4

    x = [a, b]
    y = [f(a), f(b)]

    # оценка локальной константы Липшица и характеристики первого интервала
    dx = [b - a]
    diff = [abs(y[1] - y[0]) / dx[0]]
    X_max = dx[0]
    lam = [diff[0]]
    lam_max = lam[0]
    gamma = [lam_max * dx[0] / X_max]
    mu = [r * max(lam[0], gamma[0], xi)]
    R = [characteristic(0)]

    while True:
        # выбор подынтервала
        i_best = 0
        R_best = R[0]
        for i in range(1, len(R)):
            if R[i] < R_best:
                R_best = R[i]
                i_best = i
        i = i_best

        # условия остановки
        if x[i + 1] - x[i] <= atol:
            break
        if len(y) >= maxfev:
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')

        # новое испытание
        x.insert(i + 1, new_point(i))
        y.insert(i + 1, f(x[i + 1]))

        # оценка локальных констант Липшица и характеристик
        update_gamma = False  # обновлять ли все значения gamma
        if dx[i] == X_max:    # удалим самый длинный интервал
            X_max = None
            update_gamma = True
        dx[i] = x[i + 1] - x[i]
        dx.insert(i + 1, x[i + 2] - x[i + 1])
        k = i if dx[i] > dx[i + 1] else i + 1
        if X_max is None:
            X_max = max(dx)
        elif dx[k] > X_max:
            X_max = dx[k]
            update_gamma = True

        # обновляем lam
        diff[i] = abs(y[i + 1] - y[i]) / dx[i]
        diff.insert(i + 1, abs(y[i + 2] - y[i + 1]) / dx[i + 1])
        lam.insert(i + 1, None)
        for j in range(max(0, i - 1), min(i + 3, len(lam))):
            j1 = max(0, j - 1)
            j2 = min(j + 2, len(lam))
            lam[j] = max(diff[j1:j2])
            if lam[j] > lam_max:
                lam_max = lam[j]
                update_gamma = True

        if update_gamma:  # обновить все значения
            gamma = [lam_max * dx_i / X_max for dx_i in dx]
            mu = [r * max(lam[i], gamma[i], xi) for i in range(len(lam))]
            R = [characteristic(i) for i in range(len(y) - 1)]
        else:  # обновить только значения, измененные новыми интервалами
            gamma.insert(i + 1, None)
            mu.insert(i + 1, None)
            R.insert(i + 1, None)
            for j in range(max(0, i - 1), min(i + 3, len(lam))):
                gamma[j] = lam_max * dx[j] / X_max
                mu[j] = r * max(lam[j], gamma[j], xi)
                R[j] = characteristic(j)

    # найдём минимум
    i_min = 0
    y_min = y[0]
    for i, yi in enumerate(y):
        if yi < y_min:
            i_min = i
            y_min = yi
    x_min = x[i_min]

    # возвращаем результат
    if not full_output:
        return x_min
    return OptimizeResult(x_min, x, y, len(y))

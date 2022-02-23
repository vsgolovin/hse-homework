from collections import namedtuple


OptimizeResult = namedtuple('OptimizeResult', ['x_min', 'x', 'y', 'nfev'])
MAXFEV = 3000


def argmax(iterable):
    """
    Возвращает индекс максимального элемента.
    """
    i_max = None
    x_max = None
    for i, x in enumerate(iterable):
        if x_max is None or x > x_max:
            x_max = x
            i_max = i
    return i_max


def argmin(iterable):
    """
    Возвращает индекс максимального элемента.
    """
    i_min = None
    x_min = None
    for i, x in enumerate(iterable):
        if x_min is None or x < x_min:
            x_min = x
            i_min = i
    return i_min


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
        i = argmin(F)

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
        i = argmin(F)

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
    x_min = x[argmin(y)]

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
        i = argmax(R)

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
    x_min = x[argmin(y)]

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
        i = argmax(R)

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
    x_min = x[argmin(y)]

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
        i = argmin(R)

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
        full_update = False  # обновлять ли все значения
        if dx[i] == X_max:   # разбиваем самый длинный интервал
            X_max = None
            full_update = True
        dx[i] = x[i + 1] - x[i]
        dx.insert(i + 1, x[i + 2] - x[i + 1])
        k = i if dx[i] > dx[i + 1] else i + 1
        if X_max is None:
            X_max = max(dx)
        elif dx[k] > X_max:
            X_max = dx[k]
            full_update = True

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
                full_update = True

        if full_update:  # обновить все значения
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
    x_min = x[argmin(y)]

    # возвращаем результат
    if not full_output:
        return x_min
    return OptimizeResult(x_min, x, y, len(y))


def minimize_INL(f, a, b, r, xi, atol=None, maxfev=MAXFEV, full_output=False):
    """
    Найти на отрезке [`a`, `b`] минимум функции `f` с помощью информационно-
    статистического метода с адаптивным оцениваением локальной константы
    Липшица. Точность метода и максимальное количество вызовов функции задаются
    параметрами `atol` и `maxfev`, соответственно.
    """
    def characteristic(i):
        return (mu[i] * dx[i] + (y[i + 1] - y[i])**2 / (dx[i] * mu[i])
                - 2 * (y[i + 1] + y[i]))

    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4

    x = [a, b]
    y = [f(a), f(b)]

    # оценка локальной константы Липшица и характеристики первого интервала
    dx = [b - a]
    X_max = dx[0]
    diff = [abs(y[1] - y[0]) / dx[0]]
    diff_max = diff[0]
    lam = [diff[0]]
    gamma = [diff_max]
    mu = [r * max(lam[0], gamma[0], xi)]
    R = [characteristic(0)]

    while True:
        # выбор подынтервала
        i = argmax(R)

        # условия остановки
        if x[i + 1] - x[i] <= atol:
            break
        if len(y) >= maxfev:
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')

        # новое испытание
        x_new = 0.5 * (x[i + 1] + x[i] - (y[i + 1] - y[i]) / mu[i])
        x.insert(i + 1, x_new)
        y.insert(i + 1, f(x_new))

        # оценка локальных констант Липшица и характеристик
        full_update = False  # обновлять ли все значения

        # обновляем dx
        if dx[i] == X_max:
            X_max = None
            full_update = True
        dx[i] = x[i + 1] - x[i]
        dx.insert(i + 1, x[i + 2] - x[i + 1])
        k = i if dx[i] > dx[i + 1] else i + 1
        if X_max is None:
            X_max = max(dx)
        elif dx[k] > X_max:
            X_max = dx[k]
            full_update = True

        # обновляем diff
        if diff[i] == diff_max:
            full_update = True
            diff_max = None
        diff[i] = abs(y[i + 1] - y[i]) / dx[i]
        diff.insert(i + 1, abs(y[i + 2] - y[i + 1]) / dx[i + 1])
        k = i if diff[i] > diff[i + 1] else i + 1
        if diff_max is None:
            diff_max = max(diff)
        elif diff[k] > diff_max:
            diff_max = diff[k]
            full_update = True

        # обновляем lam
        lam.insert(i + 1, None)
        for j in range(max(0, i - 1), min(i + 3, len(lam))):
            j1 = max(0, j - 1)
            j2 = min(j + 2, len(lam))
            lam[j] = max(diff[j1:j2])

        if full_update:  # обновить все значения
            gamma = [diff_max * dx_i / X_max for dx_i in dx]
            mu = [r * max(lam[i], gamma[i], xi) for i in range(len(lam))]
            R = [characteristic(i) for i in range(len(y) - 1)]
        else:  # обновить только значения, измененный новыми интервалами
            gamma.insert(i + 1, None)
            mu.insert(i + 1, None)
            R.insert(i + 1, None)
            for j in range(max(0, i - 1), min(i + 3, len(lam))):
                gamma[j] = diff_max * dx[j] / X_max
                mu[j] = r * max(lam[j], gamma[j], xi)
                R[j] = characteristic(j)

    # найдём минимум
    x_min = x[argmin(y)]

    # возвращаем результат
    if not full_output:
        return x_min
    return OptimizeResult(x_min, x, y, len(y))

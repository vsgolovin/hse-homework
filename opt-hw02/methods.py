from collections import namedtuple


OptimizeResult = namedtuple('OptimizeResult', ['x_min', 'x', 'y', 'nfev'])


def minimize_NA(f, a, b, L, atol=1e-5, maxfev=30000, full_output=False):
    """
    Найти на на отрезке [`a`, `b`] минимум функции `f` с помощью метода
    ломаных (недифференцируемая целевая функция, априорно заданная оценка
    константы Липшица `L`).
    """
    def new_point(i):
        return (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)

    def characteristic(i):
        return (y[i] + y[i + 1]) / 2 - L * (x[i + 1] - x[i]) / 2

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
        x_m.pop(i)
        F.pop(i)
        x_m.insert(i, new_point(i))
        x_m.insert(i + 1, new_point(i + 1))
        F.insert(i, characteristic(i))
        F.insert(i + 1, characteristic(i + 1))

    # возвращаем результат
    x_min = (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)
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


def minimize_ING(f, a, b, r, atol=None, maxfev=30000, full_output=False):
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

    # точность по умолчанию
    if atol is None:
        atol = (b - a) * 1e-4

    x = [a, b]
    y = [f(a), f(b)]

    while True:
        # оценка глобальной константы Липшица
        H = max(abs(y[i + 1] - y[i]) / (x[i + 1] - x[i])
                for i in range(len(x) - 1))
        mu = 1 if H < 1e-7 else H * r

        # выбор подынтервала
        R = [characteristic(i) for i in range(len(y) - 1)]
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

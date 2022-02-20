from collections import namedtuple


OptimizeResult = namedtuple('OptimizeResult', ['x_min', 'x', 'y', 'nfev'])


def minimize_na(f, a, b, L, atol=1e-5, maxfev=100000, full_output=False):
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

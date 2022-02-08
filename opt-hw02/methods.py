def minimize_na(f, a, b, L, atol=1e-5, maxfev=100000, return_nfev=False):
    """
    Найти на на отрезке [`a`, `b`] минимум функции `f` с помощью метода
    ломаных (недифференцируемая целевая функция, априорно заданная оценка
    константы Липшица `L`).
    """
    def characteristic(i):
        return (y[i] + y[i + 1]) / 2 - L * (x[i + 1] - x[i]) / 2

    x = [a, b]               # точки разбиения интервала
    y = [f(a), f(b)]         # значения функции в этих точках
    nfev = 2                 # количество вызовов функции `f`
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

        # проверим условие остановки
        if x[i + 1] - x[i] <= atol:
            break

        # условие не выполнено => разбиваем интервал на 2 части
        if nfev > maxfev:  # слишком много итераций -- останавливаем расчёт
            raise Exception(
                f'Решение не сошлось после {maxfev} вызовов целевой функции.')
        x_mid = (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)
        y_mid = f(x_mid)
        nfev += 1

        # меняем характеристику старого интервала на характеристики двух новых
        F.pop(i)
        x.insert(i + 1, x_mid)
        y.insert(i + 1, y_mid)
        F.insert(i, characteristic(i))
        F.insert(i + 1, characteristic(i + 1))

    # возвращаем результат
    x_min = (x[i] + x[i + 1]) / 2 - (y[i + 1] - y[i]) / (2 * L)
    if return_nfev:
        return x_min, nfev
    return x_min

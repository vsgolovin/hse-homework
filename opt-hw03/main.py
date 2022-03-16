import numpy as np
import matplotlib.pyplot as plt
from methods import gradient_descent
import test_functions as tf


# различные методы выбора h
x0 = (0, 0)
sol = gradient_descent(tf.quadratic, tf.quadratic_derivatives, x0,
                       return_solutions=True)
print('Квадратичная функция / условие минимума:')
print(f'x = {sol[-1]}, число итераций: {len(sol)}\n')

sol = gradient_descent(tf.quadratic, tf.quadratic_derivatives, x0,
                       h_method=2e-3, return_solutions=True)
print('Квадратичная функция / h = const:')
print(f'x = {sol[-1]}, число итераций: {len(sol)}\n')

sol = gradient_descent(tf.quadratic, tf.quadratic_derivatives, x0,
                       h_method=(0.1, 0.9), return_solutions=True)
print('Квадратичная функция / линейное приближение:')
print(f'x = {sol[-1]}, число итераций: {len(sol)}\n')

sol = gradient_descent(tf.rosenbrock, tf.rosenbrock_derivatives, x0,
                       return_solutions=True)
print('Функция Розенброка / условие минимума:')
print(f'x = {sol[-1]}, число итераций: {len(sol)}\n')

sol = gradient_descent(tf.rosenbrock, tf.rosenbrock_derivatives, x0,
                       h_method=2e-3, return_solutions=True)
print('Функция Розенброка / h = const:')
print(f'x = {sol[-1]}, число итераций: {len(sol)}\n')

sol = gradient_descent(tf.rosenbrock, tf.rosenbrock_derivatives, x0,
                       h_method=(0.1, 0.9), return_solutions=True)
print('Функция Розенброка / линейное приближение:')
print(f'x = {sol[-1]}, число итераций: {len(sol)}\n')


# визуализация
mesh = np.meshgrid(np.linspace(-0.5, 2), np.linspace(-1, 2))

# квадратичная функция
sol = gradient_descent(tf.quadratic, tf.quadratic_derivatives, x0,
                       h_method=(0.1, 0.9), return_solutions=True)
fig = plt.figure('Quadratic function')
ax = fig.add_subplot()
cn = ax.contourf(*mesh, tf.quadratic(*mesh), levels=20)
cb = fig.colorbar(cn)
sol = np.array(sol)
ax.plot(sol[:, 0], sol[:, 1], 'r-', lw=0.5)


# функция Розенброка
sol = gradient_descent(tf.rosenbrock, tf.rosenbrock_derivatives, x0,
                       h_method='min', return_solutions=True)
fig = plt.figure('Rosenbrock function')
ax = fig.add_subplot()
cn = ax.contourf(*mesh, tf.quadratic(*mesh), levels=20)
cb = fig.colorbar(cn)
sol = np.array(sol)
ax.plot(sol[:, 0], sol[:, 1], 'r-', lw=0.5)

plt.show()

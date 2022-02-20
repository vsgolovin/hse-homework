import numpy as np
import matplotlib.pyplot as plt
from test_functions import f1
from methods import minimize_na


# minimizer settings
a = -1.5
b = 11
L = 13900
eps = 1e-4 * (b - a)

# solve the optimization problem
ans = minimize_na(f1, a, b, L, atol=eps, full_output=True)
print(f'Минимум: {ans.x_min:.3f}'
      + f', количество вызовов целевой функции: {ans.nfev}')

# plot results
x = np.linspace(a, b, 200)
y = [f1(xi) for xi in x]
plt.figure()
plt.plot(x, y)
plt.plot(ans.x, ans.y, 'r-', lw=0.5)
plt.show()

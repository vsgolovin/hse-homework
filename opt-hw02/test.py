from test_functions import f1
from methods import minimize_na


a = -1.5
b = 11
# L = 7536
L = 8500
eps = 1e-4 * (b - a)


ans, nfev = minimize_na(f1, a, b, L, atol=eps, return_nfev=True)
print(ans, nfev)

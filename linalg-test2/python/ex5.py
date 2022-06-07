import numpy as np
import numpy.linalg as la


A = np.array([[-24., 7, -6], [-84, 24, -20], [36, -10, 10]])
print(f'A = \n{A}')

lam, _ = la.eig(A)
print(f'eigenvalues of A:\n{lam}')

lhs = np.array([[36., 6, 1], [4, 2, 1], [4, 1, 0]])
rhs = np.exp([8.0, 4.0, 4.0])
p = la.solve(lhs, rhs)
print(f'polynomial coefficients: \n{p}')

sol = p[0] * (A @ A) + p[1] * A + p[2] * np.eye(3)
print(f'exp(A+2) = \n{sol}')

# exponential through Taylor series
ans = np.zeros_like(A)
for n in range(100):
    s = np.eye(3)
    for i in range(n):
        s = s @ A / (i + 1)
    ans += s
ans = ans * np.exp(2)
print(f'brute force:\n{ans}')

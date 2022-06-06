import numpy as np
import numpy.linalg as la


def main(A: np.ndarray, b: np.ndarray, eps: float):
    P = np.zeros_like(A)
    b_orig = b.copy()
    for i, row in enumerate(A):
        P[i] = -row / row[i]
        b[i] /= row[i]
        P[i, i] = 0.0
    sigma = np.sqrt(np.max(la.eig(P.T @ P)[0]))
    P_1 = la.norm(P, 1)
    P_2 = la.norm(P, 2)
    assert abs(sigma - P_2) < 1e-7
    num_iter_1 = np.log((1 - P_1) * eps / la.norm(b, 1)) / np.log(P_1)
    num_iter_2 = np.log((1 - P_2) * eps / la.norm(b, 2)) / np.log(P_2)

    print(f'P = \n{P}')
    print(f'||P||_1 = {la.norm(P, 1)}')
    print(f'||P||_2 = {la.norm(P, 2)}')
    print(f'b = {b}')
    print(f'|b|_1 = {la.norm(b, 1)}')
    print(f'|b|_2 = {la.norm(b, 2)}')
    print(f'k_1 >= {num_iter_1}')
    print(f'k_2 >= {num_iter_2}')

    print('\nIteration method')
    x = np.zeros(len(P))
    print(f'x_0 = {x}')

    num_iter = int(np.ceil(min(num_iter_1, num_iter_2)))
    for k in range(1, 101):
        x = P @ x + b
        if k <= num_iter or k == 100:
            print(f'x_{k} = {x}')

    print('\n Analytic solution')
    print(f'x = {la.solve(A, b_orig)}')


if __name__ == '__main__':
    A = np.array([[22., 5, 4], [2, 26, 2], [1, 4, 23]])
    b = np.array([2., 9, 1])
    eps = 0.01
    main(A, b, eps)

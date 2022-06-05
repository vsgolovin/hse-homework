import numpy as np
import numpy.linalg as la


def main(A: np.ndarray, ord: int, c: float):
    kappa = la.norm(A, ord=ord) * la.norm(la.inv(A), ord=ord)
    eps = c * A.shape[0]
    delta_A = eps / la.norm(A, ord=ord)
    delta_Ainv = kappa * delta_A / (1 - kappa * delta_A)

    print(f'A^-1 = 1/10*\n{10*la.inv(A)}')
    print(f'||A|| = {la.norm(A, ord=ord)}')
    print(f'||A^-1|| = {la.norm(la.inv(A), ord=ord)}')
    print(f'kappa = {kappa}')
    print(f'||eps_A|| = {eps}')
    print(f'delta_A = {eps / la.norm(A, ord=ord)}')
    print(f'delta_Ainv = {delta_Ainv}')


if __name__ == '__main__':
    A = np.array([[2, -3], [8, -7]])
    norm_ord = 1
    c = 0.01
    main(A, norm_ord, c)

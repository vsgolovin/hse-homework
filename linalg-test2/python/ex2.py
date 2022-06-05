import numpy as np
import numpy.linalg as la


def main(A: np.ndarray, b: np.ndarray, xhat: np.ndarray, verbose: bool = True):
    # approximate A and b
    Ahat = np.round(A)
    bhat = np.round(b)
    eps_A = Ahat - A
    eps_b = bhat - b
    x = la.solve(Ahat, bhat)
    assert np.allclose(x, xhat)
    A_inv = la.inv(A)

    if verbose:
        print(f'A_hat = \n{Ahat}')
        print(f'b_hat = {bhat}')
        print(f'A^-1 = \n{A_inv}')

    # calculate delta_x
    for ord in (1, 2):
        print(f'\nL{ord} norm:')

        def norm(x):
            return la.norm(x, ord=ord)

        kappa = norm(A) * norm(A_inv)
        delta_A = norm(eps_A) / norm(A)
        delta_b = norm(eps_b) / norm(b)
        delta_x = kappa / (1 - kappa * delta_A) * (delta_A + delta_b)

        if verbose:
            print(f'||A|| = {norm(A)}')
            print(f'||A_hat|| = {norm(Ahat)}')
            print(f'||A^-1|| = {norm(A_inv)}')
            print(f'kappa(A) = {kappa}')
            print(f'||eps_A|| = {norm(eps_A)}')
            print(f'delta_A = {delta_A}')
            print(f'|b| = {norm(b)}')
            print(f'|eps_b| = {norm(eps_b)}')
            print(f'delta_b = {delta_b}')
        print(f'delta_x = {delta_x}')


if __name__ == '__main__':
    A = np.array([[4.99, 0.02], [-4.83, -8.01]])
    b = np.array([5.0, -12.98])
    x_hat = np.array([1.0, 1.0])
    main(A, b, x_hat)

from copy import deepcopy
import numpy as np
from numpy import linalg as la


def main(verbose: bool = True):
    R = 2
    A = np.array([
        [-8., -42, 9, 40],
        [10, 48, -54, -2],
        [-67, -12, 36, -22]
    ])

    # 1. compute SVD: A = U @ S @ V.T
    # V columns are A.T @ A eigenvectors
    lam, V = la.eig(A.T @ A)
    # sort by eigenvalues
    inds = np.argsort(lam)[::-1]
    lam = lam[inds]
    V = V[:, inds]

    # find singular values
    S = np.zeros_like(A)
    np.fill_diagonal(S, np.sqrt(lam))

    # find U from A @ V = U @ S
    U = np.zeros((A.shape[0], A.shape[0]))
    for j in range(A.shape[0]):
        U[:, j] = (A @ V[:, j]) / S[j, j]

    # check decomposition
    assert np.allclose(U @ S @ V.T,  A)

    if verbose:
        print(f'A^T A = \n{A.T @ A}')
        print(f'V =  1/11 * \n{np.round(V*11).astype(int)}')
        print(f'Sigma = \n{S}')
        print(f'U = \n{U}')

    # 2. compute low-rank approximation of A
    S1 = deepcopy(S)
    S1[R:, R:] = 0.0
    A1 = U @ S1 @ V.T
    err = S[R, R]

    # check results
    A1_ans = low_rank_approx(A, R)
    assert np.allclose(A1, A1_ans)
    err_ans = la.norm(A - A1, ord=2)
    assert np.allclose(err, err_ans)

    print(f'A1 = \n{A1}')
    print(f'||A - A1||_2 = {err}')


def low_rank_approx(A: np.ndarray, r: int) -> np.ndarray:
    U, s, V = la.svd(A)
    S = np.zeros_like(A)
    np.fill_diagonal(S, s)
    S[r:, r:] = 0.0
    return U @ S @ V


if __name__ == '__main__':
    main()

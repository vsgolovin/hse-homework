import copy
import numpy as np


def main(A: np.ndarray):
    print(f'A = \n{A}')
    A_original = copy.deepcopy(A)

    T = np.array([[1., 0, 0],
                  [0, 1/np.sqrt(2), -1/np.sqrt(2)],
                  [0, 1/np.sqrt(2), 1/np.sqrt(2)]])
    A = T.T @ A @ T
    print('\n1st iteration')
    print(f'T_ij = \n{T}')
    print(f'A_1 = \n{A}')
    print(f'T_1 = \n{T}')

    phi = np.arctan(-2 * np.sqrt(2) / 7) / 2
    T_ij = np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    A = T_ij.T @ A @ T_ij
    T = T @ T_ij
    print('\n2nd iteration')
    print(f'T_ij = \n{T_ij}')
    print(f'A_2 = \n{A}')
    print(f'T_2 = \n{T}')

    print('\nCheck results:')
    lam, V = np.linalg.eig(A_original)
    print(f'eigenvalues: {lam}')
    print(f'eigenvectors: \n{V}')


if __name__ == '__main__':
    A = np.array([[4., 1, 1], [1, 8, 3], [1, 3, 8]])
    main(A)

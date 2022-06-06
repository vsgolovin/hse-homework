import numpy as np


def main(M: np.ndarray, beta: float):
    P = np.zeros(M.shape)
    for j in range(M.shape[1]):
        P[:, j] = M[:, j] / sum(M[:, j])
    print(f'P = \n{P}')

    P1 = (1 - beta) * P + beta * np.ones(M.shape) / (len(M))
    print(f'P1 = \n{P1}')

    x = np.ones(len(M)) / len(M)
    print_at = set(range(1, 6)).union(set(range(10, 101, 10)))
    for i in range(1, 101):
        x = P1 @ x
        if i in print_at:
            print(f'x_{i} = {x}')


if __name__ == '__main__':
    M = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 1, 1]
    ])
    beta = 0.15
    main(M, beta)

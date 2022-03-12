import cv2
import numpy as np
import matplotlib.pyplot as plt

n_components = 16


# создаем случайный центроид (для инициализации)
def random_centroid():
    return np.random.choice(np.arange(0, 255), size=3)


# функция вычисления расстояния между точками
def euclidean_distances(x, y):
    r = x-y
    return np.sqrt((r**2).sum(axis=-1))


# вычисление текущих значений центроидов - вычисление mu_j
def cluster_means(X, y, n_components):
    # напишите здесь ваш код
    mu = np.zeros((n_components, X.shape[-1]))
    for n in range(n_components):
        mask = y == n
        if mask.any():
            mu[n, :] = X[mask].mean(axis=0)
    return mu


# вычисление ближайшего центроида для i-ой точки - вычисление c^{(i)}
def update_encoder(X, mu):
    # напишите здесь ваш код
    distances = []
    for centroid in mu:
        distances.append(euclidean_distances(X, centroid))
    y = np.argmin(np.array(distances), axis=0)
    return y


# итерации алгоритма К-средних
def kmeans(X, iterations=10):
    n = X.shape[0]*X.shape[1]
    y = np.random.choice(n_components, n).reshape(X.shape[0], X.shape[1])
    for _ in range(iterations):
        mu = cluster_means(X, y, n_components)
        y = update_encoder(X, mu)
    return mu


if __name__ == '__main__':
    X = cv2.imread('data/boat.jpg')
    mu = kmeans(X, 10)
    clusters = update_encoder(X, mu)
    compressed = cv2.cvtColor(clusters.astype('uint8'), cv2.COLOR_GRAY2BGR)
    for i in range(n_components):
        ix = clusters == i
        compressed[ix] = mu[i]
    plt.imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
    plt.show()

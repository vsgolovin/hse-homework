import numpy as np
import dtree


x_1 = np.array([0, 0, 1, 0, 1, 0, 1, 1])
x_2 = np.array([1, 1, 1, 1])
x_3 = np.array([0, 0, 1, 0])


def test_gini():
    results = np.empty(3)
    results[:] = [dtree.gini(x) for x in (x_1, x_2, x_3)]
    ans = np.array([0.5, 0.0, 0.375])
    assert np.allclose(results, ans)


def test_entropy():
    results = np.array([dtree.entropy(x) for x in (x_1, x_2, x_3)])
    ans = np.array([1.0, 0.0, 0.811])
    assert np.allclose(results, ans, rtol=0, atol=1e-3)

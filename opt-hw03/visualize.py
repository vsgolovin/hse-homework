import numpy as np
import matplotlib.pyplot as plt
from test_functions import quadratic, rosenbrock


mesh = np.meshgrid(np.linspace(-3, 5), np.linspace(-2, 3))

fig = plt.figure('Quadratic function')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(*mesh, quadratic(*mesh))

fig2 = plt.figure('Rosenbrock function')
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_surface(*mesh, rosenbrock(*mesh))
plt.show()

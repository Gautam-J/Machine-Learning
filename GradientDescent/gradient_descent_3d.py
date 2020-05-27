import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def f(x, y):
    '''
    Himmelblau's function (four identical local minima)

    Local Maximum:
        (-0.270845, -0.923039) = 181.617

    Local Minima:
        (3, 2) = 0
        (-2.805118, 3.131312) = 0
        (-3.779310, -3.283186) = 0
        (3.584428, -1.848126) = 0
    '''
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def grad_f(x, y):
    # partial derivatives of the above function
    dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
    return np.array([dx, dy])


# ------------------------- Gradient Descent Algorithm ------------------------
history = {'x': [], 'y': []}  # record training history
x = np.random.uniform(-6, 6, 2)  # random starting position
alpha = 1e-3  # learning rate
max_iters = 100  # maximum number of iterations

for _ in range(max_iters):
    # calculate gradients
    gradients = grad_f(x[0], x[1])
    # gradient descent
    x = x - alpha * gradients
    # record training history
    history['x'].append(x[0])
    history['y'].append(x[1])

# -------------------------------VISUALIZATION---------------------

x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)  # all possible combinations of x and y
Z = f(X, Y)
x_his = np.array(history['x'])
y_his = np.array(history['y'])

fig = plt.figure(figsize=(10, 7))

# surface plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='jet', alpha=0.5)

ax1.set_title(f'Iterations: {max_iters} lr={alpha}')

# contour plot
ax2 = fig.add_subplot(122)
levels = np.linspace(0, 500, 30)
ax2.contour(X, Y, Z, levels, cmap='jet', alpha=0.5)
ax2.contourf(X, Y, Z, levels, cmap='jet', alpha=0.5)

for i in range(len(x_his)):
    ax1.scatter(x_his[i], y_his[i], f(x_his[i], y_his[i]), c='r', marker='*', alpha=0.5)
    ax2.scatter(x_his[i], y_his[i], c='r', marker='*', alpha=0.3)
    plt.pause(0.001)

plt.show()

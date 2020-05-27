import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use('seaborn')


def f(x):
    # minima at x = -1.59791
    return x**4 + 2 * x**3 + x + 4


def grad_f(x):
    # derivative of the above function
    return 4 * x**3 + 6 * x**2 + 1


# ------------------------- Gradient Descent Algorithm ------------------------

history = {'x': []}  # Keeping record of parameters
x = 1.0  # random starting position
alpha = 1e-2  # Learning Rate (set it to 0.2 to see model not converging)
max_iters = 150  # Maximum number of iterations

for _ in range(max_iters):
    # calculate gradients
    gradients = grad_f(x)
    # gradient descent
    x = x - alpha * gradients
    # record training history
    history['x'].append(x)

# -------------------------------VISUALIZATION--------------------------

x = np.linspace(-2.5, 1, 1000)
y = f(x)

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111)

ax1.plot(x, y, c='b', label='function', alpha=0.6)
line1, = ax1.plot([], [], label='optimization', c='r', marker='.', alpha=0.4)
ax1.set_title(f'Iterations: {max_iters} lr: {alpha}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

scatter_x, scatter_y = [], []


def init():
    line1.set_data([], [])
    return line1,


def animate(i):
    x = history['x'][i]
    scatter_x.append(x)
    scatter_y.append(f(x))

    line1.set_data(scatter_x, scatter_y)
    return line1,


ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(history['x']), blit=True,
                              interval=10, repeat=False)

# saves the animation as .mp4 (takes time, comment if needed)
mywriter = animation.FFMpegWriter(fps=30)
ani.save(f'Gradient Descent 2D.mp4', writer=mywriter)
# show the animation
plt.show()

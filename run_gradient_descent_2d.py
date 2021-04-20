import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from GradientDescent.gradient_descent_2d import GradientDescent2D

plt.style.use('seaborn')

lr = 3e-2
gd = GradientDescent2D(alpha=lr, max_iterations=150, start_point=1.0, epsilon=1e-3)
gd.run()

# -------------------------------VISUALIZATION--------------------------

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(111)

# plot the original function
x = np.linspace(-2.5, 1, 1000)
y = gd.f(x)
ax1.plot(x, y, c='b', label='function', alpha=0.6)

# destructure history object
history = gd.getHistory()
gradientHistory = history['grads']
xHistory = history['x']
yHistory = [gd.f(i) for i in history['x']]
totalIterations = len(xHistory) - 1

scatter_x, scatter_y = [], []
line1, = ax1.plot(scatter_x, scatter_y, label='optimization', c='r', marker='.', alpha=0.4)
ax1.set_title(f'Iterations: {totalIterations} lr: {lr}')
ax1.set_xlabel('X')
ax1.set_ylabel('f(x)')
ax1.legend()



def init():
    line1.set_data([], [])
    return line1,


def animate(i):
    x = xHistory[i]
    y = yHistory[i]
    scatter_x.append(x)
    scatter_y.append(y)

    line1.set_data(scatter_x, scatter_y)
    return line1,


ani = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=totalIterations + 1, blit=True,
                            interval=200, repeat=False)

# show the animation
plt.show()

# plot the distribution of gradients
plt.xlabel('Gradients')
plt.title('Historgram of Gradients')
plt.show()

# saves the animation as .mp4 (takes time, comment if needed)
# mywriter = animation.FFMpegWriter(fps=30)
# ani.save(f'Gradient Descent 2D.mp4', writer=mywriter)

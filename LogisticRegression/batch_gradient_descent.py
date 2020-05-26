import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_blobs
from sklearn.metrics import log_loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


plt.style.use('seaborn')
np.random.seed(42)

x, y = make_blobs(n_samples=100, centers=2, n_features=2,
                  cluster_std=2.25, random_state=42)

ones = np.ones((len(x), 1))
x = np.concatenate((ones, x), axis=1)
y = y.reshape(-1, 1)

theta = np.random.random((x.shape[-1], 1))
theta = np.array([2, 2, 2]).reshape(-1, 1)  # if you need to set manually

alpha = 1e-2
max_iter = 1000
m = len(y)

history = {
    'theta': [],
    'cost': []
}

for i in range(max_iter):
    # compute matrix multiplication
    lr = np.dot(x, theta)
    # get hypothesis
    hypothesis = sigmoid(lr)
    # compute Log loss
    cost = ((1 / m) * np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis))) * -1
    # calculate gradients
    gradients = np.sum(((hypothesis - y) * x), keepdims=True, axis=0).T
    # update theta
    theta = theta - (alpha * (1 / m) * gradients)
    # record training history
    history["theta"].append(theta.squeeze().tolist())
    history['cost'].append(cost)

    if not i % 50 or i == max_iter - 1:
        print(f'Iteration: {i} Cost: {cost}')

y_pred = sigmoid(np.dot(x, theta))
print('\nTheta:', theta.squeeze().tolist())
print('Log loss:', log_loss(y, y_pred))

line_x = np.linspace(-5, 5, 50)

line_ys = []
for i in range(len(history['cost'])):
    theta = history['theta'][i]
    line_y = -(theta[0] + theta[1] * line_x) / theta[2]
    line_ys.append(line_y)

x1, x2 = x[:, 1], x[:, 2]

x1_0 = [x1[i] for i in range(len(x1)) if y[i][0] == 0]
x1_1 = [x1[i] for i in range(len(x1)) if y[i][0] == 1]
x2_0 = [x2[i] for i in range(len(x2)) if y[i][0] == 0]
x2_1 = [x2[i] for i in range(len(x2)) if y[i][0] == 1]

name = 'Batch'

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(x1_0, x2_0, color='r')
ax1.scatter(x1_1, x2_1, color='b')
line1, = ax1.plot([], [], '--', alpha=0.6, color='k')
ax1.set_title(f'{name} Gradient Descent')
ax1.set_xlabel('X1 data')
ax1.set_ylabel('X2 data')

line2, = ax2.plot([], color='g')
ax2.set_title(f'{name} - Training History Cost')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost (Log Transformed)')

x_cost, y_cost = [], []


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(i):
    y_value = line_ys[i]
    cost = np.log(history['cost'][i])

    x_cost.append(i)
    y_cost.append(cost)

    xmin, xmax = ax2.get_xlim()
    if i >= xmax:
        ax2.set_xlim(xmin, 2 * xmax)
        ax2.figure.canvas.draw()
    elif i < xmin:
        ax2.set_xlim(xmin / 2, xmax)
        ax2.figure.canvas.draw()

    ymin, ymax = ax2.get_ylim()

    if cost >= ymax:
        ax2.set_ylim(ymin, cost + 0.5)
        ax2.figure.canvas.draw()
    elif cost < ymin:
        ax2.set_ylim(cost * 2, ymax)
        ax2.figure.canvas.draw()

    line1.set_data(line_x, y_value)
    line2.set_data(x_cost, y_cost)

    return line1, line2


ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(line_ys), blit=True,
                              interval=1, repeat=False)

# saves the animation as .mp4 (takes time, comment if needed)
mywriter = animation.FFMpegWriter(fps=60)
ani.save(f'{name} Gradient Descent.mp4', writer=mywriter)
# show the animation
plt.show()

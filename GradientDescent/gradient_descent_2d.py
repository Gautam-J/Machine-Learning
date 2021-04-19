import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation

# plt.style.use('seaborn')


class GradientDescent2D:

    def __init__(self, alpha=1e-2, max_iterations=150, start_point=1.0):
        self.alpha = alpha
        self.maxIterations = max_iterations
        self.history = {
            'x': [],
            'grads': [],
        }

        self.x = None
        self.setStartPoint(start_point)

    def setStartPoint(self, arg):
        if type(arg) == int or type(arg) == float:
            self.x = arg
        elif arg == 'random':
            self.x = np.random.uniform(-2.5, 1)
        else:
            raise ValueError('start_point must be either "random" or a numerical value')

    @staticmethod
    def f(x):
        # minima at x = -1.59791
        return x**4 + 2 * x**3 + x + 4

    @staticmethod
    def grad_f(x):
        # derivative of the above function
        return 4 * x**3 + 6 * x**2 + 1

    def run(self):

        for _ in range(self.maxIterations):
            gradient = self.grad_f(self.x)
            self.x = self.x - (self.alpha * gradient)

            self.history['x'].append(self.x)
            self.history['grads'].append(gradient)


if __name__ == '__main__':
    gd = GradientDescent2D()
    gd.run()

# -------------------------------VISUALIZATION--------------------------

# x = np.linspace(-2.5, 1, 1000)
# y = f(x)

# fig = plt.figure(figsize=(10, 7))
# ax1 = fig.add_subplot(111)

# ax1.plot(x, y, c='b', label='function', alpha=0.6)
# line1, = ax1.plot([], [], label='optimization', c='r', marker='.', alpha=0.4)
# ax1.set_title(f'Iterations: {max_iters} lr: {alpha}')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.legend()

# scatter_x, scatter_y = [], []


# def init():
#     line1.set_data([], [])
#     return line1,


# def animate(i):
#     x = history['x'][i]
#     scatter_x.append(x)
#     scatter_y.append(f(x))

#     line1.set_data(scatter_x, scatter_y)
#     return line1,


# ani = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=len(history['x']), blit=True,
#                               interval=10, repeat=False)

# # saves the animation as .mp4 (takes time, comment if needed)
# mywriter = animation.FFMpegWriter(fps=30)
# ani.save(f'Gradient Descent 2D.mp4', writer=mywriter)
# # show the animation
# plt.show()

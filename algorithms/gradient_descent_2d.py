import numpy as np


class GradientDescent2D:

    def __init__(self, alpha=3e-2, max_iterations=150, start_point=1.0,
                 epsilon=1e-3, random=False):
        self.alpha = alpha
        self.maxIterations = max_iterations
        self.epsilon = epsilon
        self.history = {
            'x': [],
            'grads': [],
        }

        self.prevX = None
        self.x = None
        self.setStartPoint(start_point, random)
        self.printStats()

    def printStats(self):
        print('=' * 80)
        print('[INFO]\t\tHyperparameters for Gradient Descent')
        print('=' * 80)
        print(f'[INFO] Learning Rate: {self.alpha}')
        print(f'[INFO] Maximum Iterations: {self.maxIterations}')
        print(f'[INFO] Starting Point of x: {self.x}')
        print(f'[INFO] Epsilon for checking convergence: {self.epsilon}')
        print('=' * 80)

    def setStartPoint(self, arg, random):
        if random:
            self.x = np.random.uniform(-2.5, 1)
        else:
            self.x = arg

    def isConverged(self):
        return abs(self.x - self.prevX) <= self.epsilon

    def run(self):
        # log the starting point of x
        self.history['x'].append(self.x)

        for i in range(self.maxIterations):
            # keeping track of prev X for checking convergence
            self.prevX = self.x

            gradient = self.grad_f(self.x)
            self.x = self.x - (self.alpha * gradient)

            # log metrics
            self.history['x'].append(self.x)
            self.history['grads'].append(gradient)

            if self.isConverged():
                print('[INFO] Gradient Descent converged at iteration', i + 1)
                break

    def getHistory(self):
        return self.history

    @staticmethod
    def f(x):
        # minima at x = -1.59791
        return x**4 + 2 * x**3 + x + 4

    @staticmethod
    def grad_f(x):
        # derivative of the above function
        return 4 * x**3 + 6 * x**2 + 1

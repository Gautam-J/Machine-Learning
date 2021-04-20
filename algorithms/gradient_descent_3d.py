import numpy as np

from .gradient_descent_2d import GradientDescent2D


class GradientDescent3D(GradientDescent2D):

    def __init__(self, alpha=3e-3, max_iterations=150,
                 start_point=[0.62, -6.0], epsilon=1e-3, random=False):
        super().__init__(alpha=alpha, max_iterations=max_iterations,
                         start_point=start_point, epsilon=epsilon,
                         random=random)

        self.history = {
            "x": [],
            'y': [],
            'gradsX': [],
            'gradsY': []
        }

    def setStartPoint(self, arg, random):
        if random:
            self.x = np.random.uniform(-6, 6, 2)
        else:
            self.x = np.array(arg)

    def isConverged(self):
        return (abs(self.x - self.prevX) <= self.epsilon).all()

    def run(self):
        # log the starting points
        self.history['x'].append(self.x[0])
        self.history['y'].append(self.x[1])

        for i in range(self.maxIterations):
            # keeping track of prev X for checking convergence
            self.prevX = self.x

            gradients = self.grad_f(self.x)
            self.x = self.x - (self.alpha * gradients)

            # log metrics
            self.history['x'].append(self.x[0])
            self.history['y'].append(self.x[1])
            self.history['gradsX'].append(gradients[0])
            self.history['gradsY'].append(gradients[1])

            if self.isConverged():
                print('[INFO] Gradient Descent converged at iteration', i + 1)
                break

    @staticmethod
    def f(x, y):
        '''
        Himmelblau's function (four identical local minima)

        Local Minima:
            (3, 2) = 0
            (-2.805118, 3.131312) = 0
            (-3.779310, -3.283186) = 0
            (3.584428, -1.848126) = 0
        '''
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    @staticmethod
    def grad_f(X):
        x = X[0]
        y = X[1]

        # partial derivatives of the above function
        partialX = 2 * (2 * x * (x**2 + y - 11) + x + y**2 - 7)
        partialY = 2 * (x**2 + 2 * y * (x + y**2 - 7) + y - 11)

        return np.array([partialX, partialY])

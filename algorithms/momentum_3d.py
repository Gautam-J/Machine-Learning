from .gradient_descent_3d import GradientDescent3D


class Momentum3D(GradientDescent3D):

    def __init__(self, alpha=3e-3, max_iterations=150,
                 start_point=[0.62, -6.0], epsilon=1e-3, random=False,
                 momentum=0.3):

        self.momentum = momentum

        super().__init__(alpha=alpha, max_iterations=max_iterations,
                         start_point=start_point, epsilon=epsilon,
                         random=random)

    def printStats(self):
        print('=' * 80)
        print('[INFO]\t\tHyperparameters for Momentum 2D')
        print('=' * 80)
        print(f'[INFO] Learning Rate: {self.alpha}')
        print(f'[INFO] Maximum Iterations: {self.maxIterations}')
        print(f'[INFO] Starting Point of x: {self.x}')
        print(f'[INFO] Epsilon for checking convergence: {self.epsilon}')
        print(f'[INFO] Momentum: {self.momentum}')
        print('=' * 80)

    def run(self):
        # log the starting points
        self.history['x'].append(self.x[0])
        self.history['y'].append(self.x[1])
        change = 0.0

        for i in range(self.maxIterations):
            # keeping track of prev X for checking convergence
            self.prevX = self.x

            gradients = self.grad_f(self.x)
            newChange = self.alpha * gradients + self.momentum * change
            self.x = self.x - (newChange)
            change = newChange

            # log metrics
            self.history['x'].append(self.x[0])
            self.history['y'].append(self.x[1])
            self.history['gradsX'].append(gradients[0])
            self.history['gradsY'].append(gradients[1])

            if self.isConverged():
                print('[INFO] Gradient Descent using Momentum converged at iteration', i + 1)
                break

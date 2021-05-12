from .gradient_descent_2d import GradientDescent2D


class Momentum2D(GradientDescent2D):

    def __init__(self, alpha=3e-2, max_iterations=150, start_point=1.0,
                 epsilon=1e-3, momentum=0.3, random=False):

        self.momentum = momentum

        super().__init__(alpha=alpha, max_iterations=max_iterations,
                         start_point=start_point, epsilon=epsilon, random=random)

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
        # log the starting point of x
        self.history['x'].append(self.x)
        change = 0.0

        for i in range(self.maxIterations):
            # keeping track of prev X for checking convergence
            self.prevX = self.x

            gradient = self.grad_f(self.x)
            newChange = self.alpha * gradient + self.momentum * change
            self.x = self.x - (newChange)
            change = newChange

            # log metrics
            self.history['x'].append(self.x)
            self.history['grads'].append(gradient)

            if self.isConverged():
                print('[INFO] Gradient Descent using Momentum converged at iteration', i + 1)
                break

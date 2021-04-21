import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.linear_regression import LinearRegression

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak Linear Regression.')

    parser.add_argument('--lr', type=float, default=3e-2,
                        help='Learning rate. Defaults to 0.03')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum epochs for gradient descent to run. Defaults to 10')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-2,
                        help='Epsilon for checking convergence. Defaults to 0.01')
    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-l', '--length', type=int, default=5,
                        help="Length of the animation in seconds. Defaults to 5")
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help="Batch size to be taken for mini-batch gradient descent. Defaults to 10")
    parser.add_argument('--scale', type=bool, default=True,
                        help="Flag to perform mean normalization and feature scaling. Defaults to True")
    parser.add_argument('-m', '--n_samples', type=int, default=200,
                        help="Number of training examples. Defaults to 200")
    parser.add_argument('--noise', type=float, default=0.7,
                        help="Noise of the dataset. Defaults to 0.7")

    return parser.parse_args()


def getCircularDataset(n_samples=200, noise=1):
    center = np.random.uniform(low=-10, high=10)

    x = np.random.normal(size=(n_samples, 1))
    y = center**2 - x**2

    if noise:
        noiseData = np.random.normal(scale=noise, size=(n_samples, 1))
        y += noiseData

    return x, y


def plotAndSaveGraphs(lr, args, scaler):
    # TODO: Animate

    history = lr.getHistory()
    costHistory = history['cost']

    plt.plot(costHistory)
    plt.show()

    fullData = np.linspace(lr.x[:, 1].min(), lr.x[:, 1].max(), 100).reshape(-1, 1)
    squared = np.concatenate((fullData, (fullData[:, 0]**2).reshape(-1, 1)), axis=1)

    if scaler:
        squared = scaler.transform(squared)

    fullDataWithOnes = np.concatenate((np.ones((squared.shape[0], 1)), squared), axis=1)

    plt.scatter(lr.x[:, 1], lr.y)
    plt.plot(fullData, lr.getPrediction(fullDataWithOnes, lr.getThetaByNormalEquations()))
    plt.show()

    # TODO: Add distribution of gradients as a 1 x 3 fig


def main():
    args = getArguments()
    print('[DEBUG]', args)

    x, y = getCircularDataset(n_samples=args.n_samples,
                              noise=args.noise)

    nf = x[:, 0]**2
    x = np.concatenate((x, nf.reshape(-1, 1)), axis=1)

    if args.scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        scaler = None

    lr = LinearRegression(x,
                          y.reshape(-1, 1),
                          alpha=args.lr,
                          max_epochs=args.max_epochs,
                          epsilon=args.epsilon,
                          batch_size=args.batch_size)

    bestTheta = lr.getThetaByNormalEquations()
    bestPredictions = lr.getPrediction(lr.x, bestTheta)
    bestCost = lr.getCost(bestPredictions, lr.y)

    print(f'[DEBUG] Best Theta: {bestTheta.tolist()}')
    print(f'[DEBUG] Best Cost: {bestCost}')

    lr.runGradientDescent()
    optimizedTheta = lr.theta
    optimizedPredictions = lr.getPrediction(lr.x, optimizedTheta)
    optimizedCost = lr.getCost(optimizedPredictions, lr.y)

    print(f'[DEBUG] Optimized Theta: {optimizedTheta.tolist()}')
    print(f'[DEBUG] Optimized Cost: {optimizedCost}')

    plotAndSaveGraphs(lr, args, scaler)


if __name__ == "__main__":
    main()

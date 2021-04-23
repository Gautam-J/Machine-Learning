import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.preprocessing import StandardScaler

from models.linear_regression import LinearRegression

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak Linear Regression.')

    parser.add_argument('--lr', type=float, default=3e-2,
                        help='Learning rate. Defaults to 0.03')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum epochs for gradient descent to run. Defaults to 10')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-3,
                        help='Epsilon for checking convergence. Defaults to 0.001')
    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-l', '--length', type=int, default=5,
                        help="Length of the animation in seconds. Defaults to 5")
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help="Batch size to be taken for mini-batch gradient descent. Defaults to 10")
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


def animate(i, dataset, costDataset, line, c_line):
    x = dataset[0]
    preds = dataset[1][i]
    line.set_data(x, preds)
    c_line.set_data(costDataset[:, :i])
    return line, c_line


def plotAndSaveGraphs(lr, args, scaler):

    # destructure history object
    history = lr.getHistory()
    thetaHistory = np.array(history['theta'])
    costHistory = history['cost']
    totalIterations = len(costHistory) - 1
    costDataset = np.array([np.arange(1, totalIterations + 2), costHistory])

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'linear_regression')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    fullData = np.linspace(lr.x[:, 1].min(), lr.x[:, 1].max(), 100).reshape(-1, 1)
    squared = np.concatenate((fullData, (fullData[:, 0]**2).reshape(-1, 1)), axis=1)
    squared = scaler.transform(squared)

    fullDataWithOnes = np.concatenate((np.ones((squared.shape[0], 1)), squared), axis=1)

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(121)

    sns.scatterplot(x=lr.x[:, 1].reshape(-1,), y=lr.y.reshape(-1,),
                    ax=ax1, label='Datapoint')

    hypotheses = []
    for theta in thetaHistory:
        theta = np.array(theta).reshape(-1, 1)
        fullDataPrediction = lr.getPrediction(fullDataWithOnes, theta)
        hypotheses.append(fullDataPrediction)

    dataset = np.array([fullData, hypotheses], dtype=object)

    line = ax1.plot(dataset[0], dataset[1][-1], c='r', label='Hypothesis',
                    alpha=0.6)[0]

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Training Dataset')
    ax1.legend()

    ax2 = fig.add_subplot(122)

    # plot training cost history
    c_line = ax2.plot(costDataset[0], costDataset[1], label='Cost')[0]
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Cost')
    ax2.set_title(f'Iterations: {totalIterations} lr: {args.lr} batch_size: {args.batch_size}')
    ax2.legend()

    lengthOfVideo = args.length
    nFrames = totalIterations + 1
    interval = lengthOfVideo * 1000 / nFrames
    fps = (1 / (interval / 1000))

    print('=' * 80)
    print('[INFO]\t\tParameters for Animation')
    print('=' * 80)
    print(f'[INFO] Duration of video: {lengthOfVideo} seconds')
    print(f'[DEBUG] Total number of frames: {nFrames}')
    print(f'[DEBUG] Interval for each frame: {interval}')
    print(f'[DEBUG] FPS of video: {fps}')
    print('=' * 80)

    ani = animation.FuncAnimation(fig, animate, fargs=(dataset, costDataset, line, c_line),
                                  frames=nFrames, blit=False,
                                  interval=interval, repeat=True)

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearLinearRegression.mp4')
        print('[INFO] Saving animation...')
        startTime = time.time()
        ani.save(fileName, fps=fps)
        timeDifference = time.time() - startTime
        print(f'[INFO] Animation saved to {fileName}. Took {timeDifference:.2f} seconds.')
        plt.close()
    else:
        plt.show()

    # plot distribution of theta
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(131)

    sns.kdeplot(x=thetaHistory[:, 0].reshape(-1,), fill=True, ax=ax1, label='Theta1 Values')
    ax1.set_xlabel('Theta1 values')
    ax1.set_title('Distribution of theta1')
    ax1.legend()

    ax2 = fig.add_subplot(132)

    sns.kdeplot(x=thetaHistory[:, 1].reshape(-1,), fill=True, ax=ax2, label='Theta2 Values')
    ax2.set_xlabel('Theta2 values')
    ax2.set_title('Distribution of theta2')
    ax2.legend()

    ax3 = fig.add_subplot(133)

    sns.kdeplot(x=thetaHistory[:, 2].reshape(-1,), fill=True, ax=ax3, label='Theta3 Values')
    ax3.set_xlabel('Theta3 values')
    ax3.set_title('Distribution of theta3')
    ax3.legend()

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearDistributionOfGradients.png')
        plt.savefig(fileName)
        print(f'[INFO] Distribution of gradients saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print('[DEBUG]', args)

    x, y = getCircularDataset(n_samples=args.n_samples,
                              noise=args.noise)

    nf = x[:, 0]**2
    x = np.concatenate((x, nf.reshape(-1, 1)), axis=1)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

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

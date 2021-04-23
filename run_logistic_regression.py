import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from models.logistic_regression import LogisticRegression

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak Logistic Regression.')

    parser.add_argument('--lr', type=float, default=3e-2,
                        help='Learning rate. Defaults to 0.03')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum epochs for gradient descent to run. Defaults to 50')
    parser.add_argument('-e', '--epsilon', type=float, default=3e-4,
                        help='Epsilon for checking convergence. Defaults to 0.0003')
    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-l', '--length', type=int, default=5,
                        help="Length of the animation in seconds. Defaults to 5")
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help="Batch size to be taken for mini-batch gradient descent. Defaults to 10")
    parser.add_argument('-m', '--n_samples', type=int, default=100,
                        help="Number of training examples. Defaults to 100")
    parser.add_argument('--noise', type=float, default=2.25,
                        help="Noise of the dataset. Defaults to 2.25")

    return parser.parse_args()


def animate(i, dataset, costDataset, line, c_line):
    x = dataset[0]
    preds = dataset[1][i]
    line.set_data(x, preds)
    c_line.set_data(costDataset[:, :i])
    return line, c_line


def plotAndSaveGraphs(lr, args, scaler):
    history = lr.getHistory()
    costHistory = np.array(history['cost'])
    thetaHistory = np.array(history['theta'])
    totalIterations = len(costHistory) - 1
    costDataset = np.array([np.arange(1, totalIterations + 2), costHistory])

    x = lr.x[:, 1:]
    y = lr.y

    y = y.reshape(-1, 1)
    x1, x2 = x[:, 0], x[:, 1]

    x1_0 = np.array([x1[i] for i in range(len(x1)) if y[i][0] == 0])
    x1_1 = np.array([x1[i] for i in range(len(x1)) if y[i][0] == 1])
    x2_0 = np.array([x2[i] for i in range(len(x2)) if y[i][0] == 0])
    x2_1 = np.array([x2[i] for i in range(len(x2)) if y[i][0] == 1])

    theta0, theta1, theta2 = lr.history['theta'][-1]

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'logistic_regression')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(121)

    sns.scatterplot(x=x1_0, y=x2_0, label='0', ax=ax1)
    sns.scatterplot(x=x1_1, y=x2_1, label='1', ax=ax1)

    hypotheses = []
    fullData = np.linspace(x[:, 0].min(), x[:, 0].max(), args.n_samples)

    for theta in thetaHistory:
        h = - (theta[0] + fullData * theta[1]) / theta[2]
        hypotheses.append(h)

    dataset = np.array([fullData, hypotheses], dtype=object)

    line = ax1.plot(dataset[0], dataset[1][-1], c='r', label='Hypothesis',
                    alpha=0.6)[0]

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Training Dataset')
    ax1.legend()

    ax2 = fig.add_subplot(122)

    # plot training history
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
        fileName = os.path.join(pathToDirectory, 'DecisionBoundary.mp4')
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
    for i in range(1, 4):
        ax = fig.add_subplot(2, 2, i)
        sns.kdeplot(x=thetaHistory[:, i - 1].reshape(-1,), fill=True,
                    ax=ax, label=f'Theta{i} values')
        ax.set_xlabel(f'Theta{i} values')
        ax.legend()

    fig.suptitle('Distribution of Theta')

    if args.save:
        fileName = os.path.join(pathToDirectory, 'DistributionOfGradients.png')
        plt.savefig(fileName)
        print(f'[INFO] Distribution of gradients saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print('[DEBUG]', args)

    x, y = make_blobs(n_samples=args.n_samples,
                      centers=2,
                      n_features=2,
                      cluster_std=args.noise,
                      random_state=42)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    lr = LogisticRegression(x=x,
                            y=y.reshape(-1, 1),
                            alpha=args.lr,
                            max_epochs=args.max_epochs,
                            epsilon=args.epsilon,
                            batch_size=args.batch_size)

    lr.runGradientDescent()

    print(f'[DEBUG] Optimized Theta: {lr.history["theta"][-1]}')
    print(f'[DEBUG] Optimized Cost: {lr.history["cost"][-1]}')

    plotAndSaveGraphs(lr, args, scaler)


if __name__ == "__main__":
    main()

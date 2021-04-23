import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_circles
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
    parser.add_argument('--noise', type=float, default=0.1,
                        help="Noise of the dataset. Defaults to 0.1")

    return parser.parse_args()


def plotAndSaveGraphs(lr, args, scaler):

    history = lr.getHistory()
    costHistory = np.array(history['cost'])
    thetaHistory = np.array(history['theta'])

    x = lr.x
    y = lr.y
    x1, x2 = x[:, 1], x[:, 2]

    x1_0 = np.array([x1[i] for i in range(len(x1)) if y[i][0] == 0])
    x1_1 = np.array([x1[i] for i in range(len(x1)) if y[i][0] == 1])
    x2_0 = np.array([x2[i] for i in range(len(x2)) if y[i][0] == 0])
    x2_1 = np.array([x2[i] for i in range(len(x2)) if y[i][0] == 1])

    xs = np.linspace(lr.x[:, 1].min(), lr.x[:, 1].max(), 500)
    ys = np.linspace(lr.x[:, 2].min(), lr.x[:, 2].max(), 500)
    X, Y = np.meshgrid(xs, ys)
    temp = np.c_[X.ravel(), Y.ravel(), (X**2).ravel(), (Y**2).ravel()]
    temp = scaler.transform(temp)
    fullData = np.c_[np.ones((temp.shape[0], 1)), temp]

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'logistic_regression')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    # plot training data as 2D plot
    plt.scatter(x1_0, x2_0, label='0', c='k', alpha=0.7)
    plt.scatter(x1_1, x2_1, label='1', c='g', alpha=0.7)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Training Data')

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearTrainingData.png')
        plt.savefig(fileName)
        print(f'[INFO] Training Data Plot saved to {fileName}')
        plt.close()
    else:
        plt.show()

    # plot decision boundary
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(projection='3d')

    ax1.scatter(x1_0, x2_0, np.array(x1_0)**2 + np.array(x2_0)**2, label='0')
    ax1.scatter(x1_1, x2_1, np.array(x1_1)**2 + np.array(x2_1)**2, label='1')
    ax1.plot_surface(X, Y, lr.getLinearPrediction(fullData, thetaHistory[-1]).reshape(X.shape), alpha=0.8, cmap='coolwarm')

    ax1.legend()
    ax1.view_init(30, -40)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('X1^2 + X2^2')

    lengthOfVideo = args.length
    nFrames = 360
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

    ani = animation.FuncAnimation(fig, lambda i: ax1.view_init(30, i),
                                  frames=nFrames, blit=False,
                                  interval=interval, repeat=True)

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearDecisionBoundary.mp4')
        print('[INFO] Saving animation...')
        startTime = time.time()
        ani.save(fileName, fps=fps)
        timeDifference = time.time() - startTime
        print(f'[INFO] Animation saved to {fileName}. Took {timeDifference:.2f} seconds.')
        plt.close()
    else:
        plt.show()

    # plot training history
    plt.plot(costHistory[::(args.n_samples // args.batch_size)], label='Cost')[0]
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.title(f'Epochs: {args.max_epochs} lr: {args.lr} batch_size: {args.batch_size}')
    plt.legend()

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearCost.png')
        plt.savefig(fileName)
        print(f'[INFO] Plot saved to {fileName}')
        plt.close()
    else:
        plt.show()

    # plot distribution of theta
    fig = plt.figure(figsize=(16, 9))
    for i in range(1, 6):
        ax = fig.add_subplot(2, 3, i)
        sns.kdeplot(x=thetaHistory[:, i - 1].reshape(-1,), fill=True,
                    ax=ax, label=f'Theta{i} values')
        ax.set_xlabel(f'Theta{i} values')
        ax.legend()

    fig.suptitle('Distribution of Theta')

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearDistributionOfGradients.png')
        plt.savefig(fileName)
        print(f'[INFO] Distribution of gradients saved to {fileName}')
        plt.close()
    else:
        plt.show()

    plt.scatter(x1_0, x2_0, label='0', c='k', alpha=0.7)
    plt.scatter(x1_1, x2_1, label='1', c='g', alpha=0.7)
    plt.contour(X, Y, lr.getLinearPrediction(fullData, thetaHistory[-1]).reshape(X.shape), [0], cmap='coolwarm')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Training Data with Decision Boundary')

    if args.save:
        fileName = os.path.join(pathToDirectory, 'NonLinearTrainingDataWithDecisionBoundary.png')
        plt.savefig(fileName)
        print(f'[INFO] Training Data with Decision Boundary Plot saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print('[DEBUG]', args)

    x, y = make_circles(n_samples=args.n_samples,
                        noise=args.noise,
                        factor=0.3,
                        random_state=42)

    x1Squared = x[:, 0]**2
    x2Squared = x[:, 1]**2

    x = np.concatenate((x, x1Squared.reshape(-1, 1)), axis=1)
    x = np.concatenate((x, x2Squared.reshape(-1, 1)), axis=1)
    y = y.reshape(-1, 1)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    lr = LogisticRegression(x=x,
                            y=y,
                            alpha=3e-2,
                            max_epochs=1000,
                            epsilon=1e-3,
                            batch_size=100)
    lr.runGradientDescent()

    print(f'[DEBUG] Optimized Cost: {lr.history["cost"][-1]}')
    print(f'[DEBUG] Optimized Theta: {lr.history["theta"][-1]}')

    plotAndSaveGraphs(lr, args, scaler)


if __name__ == '__main__':
    main()

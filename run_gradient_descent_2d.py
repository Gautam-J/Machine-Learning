import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from algorithms.gradient_descent_2d import GradientDescent2D
from algorithms.momentum_2d import Momentum2D

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak gradient descent.')

    parser.add_argument('--lr', type=float, default=3e-2,
                        help='Learning rate. Set to 0.2 to see gradient descent NOT converging. Defaults to 0.03')
    parser.add_argument('--max_iterations', type=int, default=150,
                        help='Maximum iterations for gradient descent to run. Defaults to 150')
    parser.add_argument('--start_point', type=float, default=1.0,
                        help='Starting point for gradient descent. Defaults to 1.0')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-3,
                        help='Epsilon for checking convergence. Defaults to 0.001')
    parser.add_argument('-r', '--random', action='store_true',
                        help='Flag to initialize a random starting point')
    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-l', '--length', type=int, default=5,
                        help="Length of the animation in seconds. Defaults to 5")
    parser.add_argument('--use-momentum', action='store_true',
                        help='Flag to use momentum in gradient descent')
    parser.add_argument('--momentum', type=float, default=0.3,
                        help='Momentum for gradient descent. Only used when use-momentum is True. Defaults to 0.3')

    return parser.parse_args()


def animate(i, dataset, line):
    line.set_data(dataset[:, :i])
    return line


def plotAndSaveGraphs(gd, args):
    fig = plt.figure(figsize=(16, 9))

    # plot the original function
    ax = fig.add_subplot(111)
    x = np.linspace(-2.5, 1, 1000)
    y = gd.f(x)
    ax.plot(x, y, c='b', label='function', alpha=0.6)

    # destructure history object
    history = gd.getHistory()
    gradientHistory = history['grads']
    xHistory = history['x']
    yHistory = gd.f(np.array(xHistory))
    dataset = np.array([xHistory, yHistory])
    totalIterations = len(xHistory) - 1

    line = ax.plot(dataset[0], dataset[1], label='optimization', c='r', marker='.', alpha=0.4)[0]
    ax.set_title(f'Iterations: {totalIterations} lr: {args.lr}')
    ax.set_xlabel('X')
    ax.set_ylabel('f(x)')
    ax.legend()

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

    ani = animation.FuncAnimation(fig, animate, fargs=(dataset, line),
                                  frames=nFrames, blit=False,
                                  interval=interval, repeat=True)

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'gradient_descent')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    # save animation
    if args.save:
        fileName = os.path.join(pathToDirectory, 'GradientDescent2D.mp4')
        print('[INFO] Saving animation...')
        startTime = time.time()
        ani.save(fileName, fps=fps)
        timeDifference = time.time() - startTime
        print(f'[INFO] Animation saved to {fileName}. Took {timeDifference:.2f} seconds.')
        plt.close()
    else:
        plt.show()

    sns.kdeplot(x=gradientHistory, fill=True)
    plt.xlabel('Gradients')
    plt.title('Distribution of Gradients')

    # save distribution of gradients
    if args.save:
        fileName = os.path.join(pathToDirectory, 'DistributionOfGradients2D.png')
        plt.savefig(fileName)
        print(f'[INFO] Distribution of gradients saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print('[DEBUG]', args)

    if args.use_momentum:
        gd = Momentum2D(alpha=args.lr,
                        max_iterations=args.max_iterations,
                        start_point=args.start_point,
                        random=args.random,
                        epsilon=args.epsilon,
                        momentum=args.momentum)
    else:
        gd = GradientDescent2D(alpha=args.lr,
                               max_iterations=args.max_iterations,
                               start_point=args.start_point,
                               random=args.random,
                               epsilon=args.epsilon)

    gd.run()

    print(f'[DEBUG] Value of x: {gd.x}')
    print('[DEBUG] Expected value: -1.59791')

    plotAndSaveGraphs(gd, args)


if __name__ == "__main__":
    main()

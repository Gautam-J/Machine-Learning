import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from algorithms.momentum_3d import Momentum3D
from algorithms.gradient_descent_3d import GradientDescent3D

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak gradient descent.')

    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate. Set to 0.2 to see gradient descent NOT converging. Defaults to 0.03')
    parser.add_argument('--max_iterations', type=int, default=150,
                        help='Maximum iterations for gradient descent to run. Defaults to 150')
    parser.add_argument('--start_x', type=float, default=0.62,
                        help='Starting X point for gradient descent. Defaults to 0.62')
    parser.add_argument('--start_y', type=float, default=-6.0,
                        help='Starting Y point for gradient descent. Defaults to -6.0')
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


def animate(i, dataset, line, c_line):
    line.set_data(dataset[0:2, :i])
    line.set_3d_properties(dataset[2, :i])
    c_line.set_data(dataset[0:2, :i])
    return line, c_line


def plotAndSaveGraphs(gd, args):
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=30, azim=130)

    x = np.linspace(-6, 6, 25)
    y = np.linspace(-6, 6, 25)
    X, Y = np.meshgrid(x, y)  # all possible combinations of x and y
    Z = gd.f(X, Y)
    ax1.plot_surface(X, Y, Z, cmap='gray', alpha=0.8)

    levels = np.linspace(0, 500, 30)
    ax2 = fig.add_subplot(122)
    ax2.contourf(X, Y, Z, levels, cmap='jet', alpha=0.5)

    # destructure history object
    history = gd.getHistory()
    xHistory = np.array(history['x'])
    yHistory = np.array(history['y'])
    zHistory = gd.f(xHistory, yHistory)
    dataset = np.array([xHistory, yHistory, zHistory])
    xGradHistory = history['gradsX']
    yGradHistory = history['gradsY']
    totalIterations = len(xHistory) - 1

    line = ax1.plot(dataset[0], dataset[1], dataset[2], label='optimization', c='r', marker='.', alpha=0.4)[0]
    ax1.set_title(f'Iterations: {totalIterations} lr: {args.lr}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x, y)')
    ax1.legend()

    c_line = ax2.plot(dataset[0], dataset[1], label='optimization', c='r', marker='.', alpha=0.4)[0]

    lengthOfVideo = args.length
    nFrames = totalIterations + 1
    interval = lengthOfVideo * 1000 / nFrames
    fps = (1 / (interval / 1000))

    print('=' * 80)
    print('[INFO]\t\tParameters for Animation')
    print('=' * 80)
    print(f'[INFO] Duration of video: {lengthOfVideo} seconds')
    print(f'[DEBUG] Total number of frames: {nFrames}')
    print(f'[DEBUG] Interval for each frame: {interval:.2f}')
    print(f'[DEBUG] FPS of video: {fps}')
    print('=' * 80)

    ani = animation.FuncAnimation(fig, animate, frames=nFrames, blit=False,
                                  interval=interval, repeat=True,
                                  fargs=(dataset, line, c_line))

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'gradient_descent')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    # save animation
    if args.save:
        fileName = os.path.join(pathToDirectory, 'GradientDescent3D.mp4')
        print('[INFO] Saving animation...')
        startTime = time.time()
        ani.save(fileName, fps=fps)
        timeDifference = time.time() - startTime
        print(f'[INFO] Animation saved to {fileName}. Took {timeDifference:.2f} seconds.')
        plt.close()
    else:
        plt.show()

    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(121)
    sns.kdeplot(x=xGradHistory, fill=True, ax=ax1)
    ax1.set_xlabel('Gradients of x')
    ax1.set_title('Distribution of Gradients in x-axis')

    ax2 = fig.add_subplot(122)
    sns.kdeplot(x=yGradHistory, fill=True, ax=ax2)
    ax2.set_xlabel('Gradients of y')
    ax2.set_title('Distribution of Gradients in y-axis')

    # save distribution of gradients
    if args.save:
        fileName = os.path.join(pathToDirectory, 'DistributionOfGradients3D.png')
        plt.savefig(fileName)
        print(f'[INFO] Distribution of gradients saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print('[DEBUG]', args)

    start_point = [args.start_x, args.start_y]

    if args.use_momentum:
        gd = Momentum3D(alpha=args.lr,
                        max_iterations=args.max_iterations,
                        start_point=start_point,
                        random=args.random,
                        epsilon=args.epsilon,
                        momentum=args.momentum)
    else:
        gd = GradientDescent3D(alpha=args.lr,
                               max_iterations=args.max_iterations,
                               start_point=start_point,
                               random=args.random,
                               epsilon=args.epsilon)

    gd.run()

    print(f'[DEBUG] Value of x: {gd.x}')
    print('[DEBUG] Expected values:')
    print('[DEBUG]\t\t[3.584428, -1.848126]')
    print('[DEBUG]\t\t[-3.779310, -3.283186]')
    print('[DEBUG]\t\t[-2.805118, 3.131312]')
    print('[DEBUG]\t\t[3, 2]')

    plotAndSaveGraphs(gd, args)


if __name__ == '__main__':
    main()

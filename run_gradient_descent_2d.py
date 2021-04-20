import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from algorithms.gradient_descent_2d import GradientDescent2D

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak gradient descent.')

    parser.add_argument('--lr', type=float, default=3e-2,
                        help='Learning rate. Set to 0.2 to see gradient descent NOT converging')
    parser.add_argument('--max_iterations', type=int, default=150,
                        help='Maximum iterations for gradient descent to run')
    parser.add_argument('--start_point', type=float, default=1.0,
                        help='Starting point for gradient descent')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-3,
                        help='Epsilon for checking convergence')
    parser.add_argument('-r', '--random', action='store_true',
                        help='Flag to initialize a random starting point. If set to True, start_point will be discarded')

    return parser.parse_args()


def main():
    args = getArguments()

    gd = GradientDescent2D(alpha=args.lr,
                        max_iterations=args.max_iterations,
                        start_point=args.start_point,
                        random=args.random,
                        epsilon=args.epsilon)
    gd.run()

    print(f'[DEBUG] Value of x: {gd.x:.2f}')
    print('[DEBUG] Expected value: -1.59791')

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)

    # plot the original function
    x = np.linspace(-2.5, 1, 1000)
    y = gd.f(x)
    ax1.plot(x, y, c='b', label='function', alpha=0.6)

    # destructure history object
    history = gd.getHistory()
    gradientHistory = history['grads']
    xHistory = history['x']
    yHistory = [gd.f(i) for i in history['x']]
    totalIterations = len(xHistory) - 1

    scatter_x, scatter_y = [], []
    line1, = ax1.plot(scatter_x, scatter_y, label='optimization', c='r', marker='.', alpha=0.4)
    ax1.set_title(f'Iterations: {totalIterations} lr: {args.lr}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('f(x)')
    ax1.legend()


    def init():
        line1.set_data([], [])
        return line1,


    def animate(i):
        x = xHistory[i]
        y = yHistory[i]
        scatter_x.append(x)
        scatter_y.append(y)

        line1.set_data(scatter_x, scatter_y)
        return line1,

    lengthOfVideo = 5
    nFrames = totalIterations + 1
    interval = lengthOfVideo  * 1000 / nFrames
    fps = (1 / (interval / 1000))

    print('=' * 80)
    print('[INFO]\t\tParameters for Animation')
    print('=' * 80)
    print(f'[INFO] Duration of video: {lengthOfVideo} seconds')
    print(f'[DEBUG] Total number of frames: {nFrames}')
    print(f'[DEBUG] Interval for each frame: {interval:.2f}')
    print(f'[DEBUG] FPS of video: {fps}')
    print('=' * 80)

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=nFrames, blit=True,
                                interval=interval, repeat=False)

    # make directories
    pathToDirectory = os.path.join('visualizations', 'gradient_descent')
    if not os.path.exists(pathToDirectory):
        os.makedirs(pathToDirectory)

    # save animation
    fileName = os.path.join(pathToDirectory, 'GradientDescent2D.mp4')
    print('[INFO] Saving animation...')
    startTime = time.time()
    ani.save(fileName, fps=fps)
    timeDifference = time.time() - startTime
    print(f'[INFO] Animation saved to {fileName}. Took {timeDifference:.2f} seconds.')
    plt.close()

    # save distribution of gradients
    fileName = os.path.join(pathToDirectory, 'DistributionOfGradients2D.png')
    sns.kdeplot(x=gradientHistory, fill=True)
    plt.xlabel('Gradients')
    plt.title('Distribution of Gradients')
    plt.savefig(fileName)
    print(f'[INFO] Distribution of gradients saved to {fileName}')
    plt.close()


if __name__ == "__main__":
    main()

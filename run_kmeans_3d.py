import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from models.k_means import KMeans

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak K Nearest Neighbors.')

    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-m', '--n_samples', type=int, default=200,
                        help="Number of training examples. Defaults to 200")
    parser.add_argument('--noise', type=float, default=2.25,
                        help="Noise of the dataset. Defaults to 2.25")
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of classes in the dataset. Defaults to 4')
    parser.add_argument('-l', '--length', type=int, default=5,
                        help="Length of the animation in seconds. Defaults to 5")

    return parser.parse_args()


def plotAndSaveGraphs(km, args):
    x = km.x
    y = [km.getPrediction(sample) for sample in x]
    dataset = {c: np.array([x[i] for i in range(len(x)) if y[i] == c]) for c in range(km.k)}

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'k_means')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(projection='3d')

    c = ['r', 'g', 'b', 'k', 'c', 'm']

    # scatter all available data color coded according to the class
    for i in range(km.k):
        ax1.scatter(km.centroids[i][0], km.centroids[i][1], km.centroids[i][2], label=f'Centroid {i + 1}', c=c[i], s=100)
        ax1.scatter(dataset[i][:, 0], dataset[i][:, 1], dataset[i][:, 2], label=f'Class {i + 1}', c=c[i], alpha=0.5)

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title(f'K Means - {km.k} Clusters Classified')
    ax1.legend()
    ax1.view_init(30, -40)

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
        fileName = os.path.join(pathToDirectory, 'Clusters3D.mp4')
        print('[INFO] Saving animation...')
        startTime = time.time()
        ani.save(fileName, fps=fps)
        timeDifference = time.time() - startTime
        print(f'[INFO] Animation saved to {fileName}. Took {timeDifference:.2f} seconds.')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print(args)

    x, _ = make_blobs(n_samples=args.n_samples,
                      centers=args.n_classes,
                      cluster_std=args.noise,
                      n_features=3,
                      random_state=7)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    km = KMeans(x, k=args.n_classes)
    km.setCentroids()

    plotAndSaveGraphs(km, args)


if __name__ == '__main__':
    main()

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from models.k_nearest_neighbors import KNearestNeighbors

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak K Nearest Neighbors.')

    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-m', '--n_samples', type=int, default=100,
                        help="Number of training examples. Defaults to 100")
    parser.add_argument('--noise', type=float, default=2.25,
                        help="Noise of the dataset. Defaults to 2.25")
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset. Defaults to 2')
    parser.add_argument('-l', '--length', type=int, default=5,
                        help="Length of the animation in seconds. Defaults to 5")

    return parser.parse_args()


def plotAndSaveGraphs(knn, args, scaler, X):
    dataset = knn.dataset

    newData1 = np.random.uniform(X[:, 0].min(), X[:, 0].max(), 1)
    newData2 = np.random.uniform(X[:, 1].min(), X[:, 1].max(), 1)
    newData3 = np.random.uniform(X[:, 2].min(), X[:, 2].max(), 1)
    newData = np.c_[newData1, newData2, newData3]
    newData = scaler.transform(newData)

    predictedClass = knn.getPrediction(newData)
    print(f'[DEBUG] Predicted Class: {predictedClass}')

    distances = knn.distances
    crucial_features = [i[2] for i in sorted(distances, key=lambda x:x[0])[:knn.k]]

    # make directories
    if args.save:
        pathToDirectory = os.path.join('visualizations', 'k_nearest_neighbors')
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(projection='3d')

    # # scatter all available data color coded according to the class
    for i in np.unique(knn.y):
        ax1.scatter(dataset[i][:, 0], dataset[i][:, 1], dataset[i][:, 2], label=f'Class {i + 1}')

    # # plot distance between crucial features and new_data
    for features in crucial_features:
        ax1.plot([features[0], newData[:, 0]], [features[1], newData[:, 1]], [features[2], newData[:, 2]], '--', color='k', alpha=0.5)

    # # scatter new_data
    ax1.scatter(newData[:, 0], newData[:, 1], newData[:, 2], color='k', label='New Data')

    ax1.set_title('Prediction')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
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
        fileName = os.path.join(pathToDirectory, 'TopFeatures3D.mp4')
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

    X, y = make_blobs(n_samples=args.n_samples,
                      centers=args.n_classes,
                      n_features=3,
                      cluster_std=args.noise,
                      random_state=42)

    scaler = StandardScaler()
    x = scaler.fit_transform(X)

    knn = KNearestNeighbors(x, y)

    plotAndSaveGraphs(knn, args, scaler, X)


if __name__ == '__main__':
    main()

import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from models.k_means import KMeans

plt.style.use('seaborn')


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters to tweak K Nearest Neighbors.')

    parser.add_argument('-s', '--save', action='store_true',
                        help="Flag to save visualizations and animations")
    parser.add_argument('-m', '--n_samples', type=int, default=100,
                        help="Number of training examples. Defaults to 100")
    parser.add_argument('--noise', type=float, default=2.25,
                        help="Noise of the dataset. Defaults to 2.25")
    parser.add_argument('--n_classes', type=int, default=3,
                        help='Number of classes in the dataset. Defaults to 3')

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
    ax1 = fig.add_subplot(121)

    # scatter raw data without classifying
    sns.scatterplot(x=x[:, 0], y=x[:, 1], color='k', alpha=0.5, ax=ax1)

    # scatter initial centroids
    for i in range(km.k):
        sns.scatterplot(x=[x[i, 0]], y=[x[i, 1]], s=100, label=f'Centroid {i + 1}')

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title(f'Raw data - Given {km.k} Clusters')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    c = ['r', 'g', 'b', 'k', 'c', 'm']

    # scatter all available data color coded according to the class
    for i in range(km.k):
        sns.scatterplot(x=[km.centroids[i][0]], y=[km.centroids[i][1]], s=100, label=f'Centroid {i + 1}', ax=ax2, color=c[i])
        sns.scatterplot(dataset[i][:, 0], dataset[i][:, 1], label=f'Class {i + 1}', ax=ax2, color=c[i], alpha=0.5)

    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title(f'K Means - {km.k} Clusters Classified')
    ax2.legend()

    if args.save:
        fileName = os.path.join(pathToDirectory, 'Clusters2D.png')
        plt.savefig(fileName)
        print(f'[INFO] Plot saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print(args)

    x, _ = make_blobs(n_samples=args.n_samples,
                      centers=args.n_classes,
                      cluster_std=args.noise,
                      n_features=2,
                      random_state=1)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    km = KMeans(x, k=args.n_classes)
    km.setCentroids()

    plotAndSaveGraphs(km, args)


if __name__ == '__main__':
    main()

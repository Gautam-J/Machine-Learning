import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

    return parser.parse_args()


def plotAndSaveGraphs(knn, args, scaler, X):
    dataset = knn.dataset

    newData1 = np.random.uniform(X[:, 0].min(), X[:, 0].max(), 1)
    newData2 = np.random.uniform(X[:, 1].min(), X[:, 1].max(), 1)
    newData = np.c_[newData1, newData2]
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

    plt.figure(figsize=(10, 7))

    # scatter all available data color coded according to the class
    for i in np.unique(knn.y):
        sns.scatterplot(x=dataset[i][:, 0], y=dataset[i][:, 1], label=f'Class {i + 1}')

    # plot distance between crucial features and new_data
    for features in crucial_features:
        plt.plot([features[0], newData[:, 0]], [features[1], newData[:, 1]], '--', color='k', alpha=0.5)

    # scatter new_data
    sns.scatterplot(x=newData[:, 0], y=newData[:, 1], color='k', label='New Data')

    plt.legend()
    plt.title('Prediction')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if args.save:
        fileName = os.path.join(pathToDirectory, 'TopFeatures2D.png')
        plt.savefig(fileName)
        print(f'[INFO] Plot saved to {fileName}')
        plt.close()
    else:
        plt.show()


def main():
    args = getArguments()
    print(args)

    X, y = make_blobs(n_samples=args.n_samples,
                      centers=args.n_classes,
                      n_features=2,
                      cluster_std=args.noise,
                      random_state=42)

    scaler = StandardScaler()
    x = scaler.fit_transform(X)

    knn = KNearestNeighbors(x, y)

    plotAndSaveGraphs(knn, args, scaler, X)


if __name__ == '__main__':
    main()

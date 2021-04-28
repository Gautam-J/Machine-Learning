import numpy as np


class KMeans:

    def __init__(self, x, k, max_iters=300):
        self.x = x
        self.k = k
        self.maxIterations = max_iters
        self.centroids = {i: self.x[i] for i in range(self.k)}

    def setCentroids(self):
        for _ in range(self.maxIterations):

            # dict that holds samples of each class
            classifications = {i: [] for i in range(self.k)}

            for features in self.x:
                class_ = self.getPrediction(features)
                classifications[class_].append(features)

            for i in range(self.k):
                self.centroids[i] = np.mean(classifications[i], axis=0)

    def getPrediction(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        class_ = distances.index(min(distances))

        return class_

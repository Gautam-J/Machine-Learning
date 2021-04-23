import numpy as np


class KNearestNeighbors:

    def __init__(self, x, y, k=10):
        self.k = k
        self.x = x
        self.y = y
        print(f'[INFO] Shape of x: {self.x.shape}')
        print(f'[INFO] Shape of y: {self.y.shape}')
        print(f'[INFO] Value of k: {self.k}')

        self.initializeModel()

    def initializeModel(self):
        self.dataset = {}

        for i in np.unique(self.y):
            indices = np.where(self.y == i)[0]
            class_data = self.x[indices, :]
            self.dataset[i] = class_data

    def getPrediction(self, newData):
        self.calculateDistances(newData)
        votes = self.getVotes()
        unique = np.unique(votes)
        # np.unique returns the sorted values

        return unique[0]

    def getVotes(self):
        votes = [i[1] for i in sorted(self.distances, key=lambda x:x[0])[:self.k]]
        return votes

    def calculateDistances(self, newData):
        self.distances = []

        for i, class_ in self.dataset.items():
            for features in class_:
                # calculate euclidean distance between a single feature point and new_data to be classified
                euclideanDistance = np.sqrt(np.sum((features - newData)**2))
                self.distances.append([euclideanDistance, i, features])

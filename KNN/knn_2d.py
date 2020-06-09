import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, centers=2, n_features=2,
                  cluster_std=2.25, random_state=42)

# Separating classes for better understanding
dataset = {
    'Class 1': np.array([x[i] for i in range(len(x)) if y[i] == 0]),
    'Class 2': np.array([x[i] for i in range(len(x)) if y[i] == 1]),
}

# new data to classify / use for prediction
new_data = np.array([np.random.uniform(-5, 5, 1)[0], np.random.uniform(0, 12, 1)[0]])
# uncomment below line for manual entry
new_data = np.array([1, 6.5])

# ----------------------------------------KNN Prediction-----------------------------------------

'''
Initialize an empty list that will hold the following:
1. Euclidean Distance between a feature point and and new_data
2. Class to which the feature point belongs to
3. (Optional Step) The feature point itself, for visualizing which points contributes to voting
'''
distances = []

for class_ in dataset:
    for features in dataset[class_]:
        features = np.array(features)
        # calculate euclidean distance between a single feature point and new_data to be classified
        euclidean_distance = np.sqrt(np.sum((features - new_data)**2))
        # add to distances
        distances.append([euclidean_distance, class_, features])

# Top K votes will be considered for classifying
K = 10
# top K votes
votes = [i[1] for i in sorted(distances, key=lambda x:x[0])[:K]]
# (Optional Step) features that contributes to the top K votes
crucial_features = [i[2] for i in sorted(distances, key=lambda x:x[0])[:K]]
# majority in top K votes
vote_result = Counter(votes).most_common(1)[0][0]

print(f'\nTop {K} votes:\n{votes}\n')
print(f'Vote Count: {Counter(votes)}\n')
print(f'Vote Result: {vote_result}\n')

# -----------------------------------------Visualization----------------------------------------

fig = plt.figure(figsize=(10, 7))
# scatter all available data color coded according to the class
plt.scatter(dataset['Class 1'][:, 0], dataset['Class 1'][:, 1], color='r', label='Class 1')
plt.scatter(dataset['Class 2'][:, 0], dataset['Class 2'][:, 1], color='k', label='Class 2')

# plot distance between crucial features and new_data
for features in crucial_features:
    plt.plot([features[0], new_data[0]], [features[1], new_data[1]], '--', color='g', alpha=0.5)

# scatter new_data
plt.scatter(new_data[0], new_data[1], color='g', label='New Data')

plt.legend()
plt.title(f'Prediction: {vote_result}')
plt.show()

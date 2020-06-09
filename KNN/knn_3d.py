import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, centers=2, n_features=3,
                  cluster_std=2.25, random_state=42)

# Separating classes for better understanding
dataset = {
    'Class 1': np.array([x[i] for i in range(len(x)) if y[i] == 0]),
    'Class 2': np.array([x[i] for i in range(len(x)) if y[i] == 1]),
}

# new data to clssify / use for prediction
new_data = np.array([np.random.uniform(-4, 4, 1)[0], np.random.uniform(-5, 5, 1)[0], np.random.uniform(-10, 10, 1)[0]])
# uncomment below line for manual entry
# new_data = np.array([-4, -2.5, 0])

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
ax = fig.add_subplot(111, projection='3d')

# scatter all available data color coded according to the class
ax.scatter([i[0] for i in dataset['Class 1']], [i[1] for i in dataset['Class 1']], [i[2] for i in dataset['Class 1']], color='r', label='Class 1')
ax.scatter([i[0] for i in dataset['Class 2']], [i[1] for i in dataset['Class 2']], [i[2] for i in dataset['Class 2']], color='k', label='Class 2')

# plot distance between crucial features and new_data
for features in crucial_features:
    ax.plot([features[0], new_data[0]], [features[1], new_data[1]], [features[2], new_data[2]], '--', color='g', alpha=0.5)

# scatter new_data
ax.scatter(new_data[0], new_data[1], new_data[2], color='g', label='New Data')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title(f'Prediction: {vote_result}')
ax.legend()


def animate(i):
    ax.view_init(elev=20., azim=i)


ani = animation.FuncAnimation(fig, animate, frames=360, interval=20)

# save animation as .mp4 (takes time, comment if needed)
mywriter = animation.FFMpegWriter(fps=60)
ani.save('Animations/KNN_3D_rotating.mp4', writer=mywriter)
# show the animation
plt.show()

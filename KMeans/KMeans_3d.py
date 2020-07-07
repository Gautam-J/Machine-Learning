import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

plt.style.use('seaborn')

x, _ = make_blobs(n_samples=200, centers=4, n_features=3,
                  cluster_std=2.25, random_state=7)

K = 4  # number of clusters to find
MAX_ITERS = 300  # max number of steps to take
centroids = {i: x[i] for i in range(K)}  # initializing centroids

# -------------------------------FITTING-----------------------------
for i in range(MAX_ITERS):
    # dict that holds samples of each class
    classifications = {i: [] for i in range(K)}

    # calculate euclidean distance and classify data points
    for features in x:
        distances = [np.linalg.norm(features - centroids[centroid]) for centroid in centroids]
        class_ = distances.index(min(distances))
        classifications[class_].append(features)

    # set centroid to mean of clustered data
    for i in range(K):
        centroids[i] = np.mean(classifications[i], axis=0)


# -----------------------------VISUALIZATION--------------------------
def predict_class(data):
    distances = [np.linalg.norm(data - centroids[c]) for c in centroids]
    class_ = distances.index(min(distances))
    return class_


y = [predict_class(sample) for sample in x]

dataset = {
    'Class 1': np.array([x[i] for i in range(len(x)) if y[i] == 0]),
    'Class 2': np.array([x[i] for i in range(len(x)) if y[i] == 1]),
    'Class 3': np.array([x[i] for i in range(len(x)) if y[i] == 2]),
    'Class 4': np.array([x[i] for i in range(len(x)) if y[i] == 3]),
}

# scatter all available data color coded according to the class
fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset['Class 1'][:, 0], dataset['Class 1'][:, 1], dataset['Class 1'][:, 2], color='r', alpha=0.5, label='Class 1')
ax.scatter(centroids[0][0], centroids[0][1], centroids[0][2], c='r', s=100)
ax.scatter(dataset['Class 2'][:, 0], dataset['Class 2'][:, 1], dataset['Class 2'][:, 2], color='g', alpha=0.5, label='Class 2')
ax.scatter(centroids[1][0], centroids[1][1], centroids[0][2], c='g', s=100)
ax.scatter(dataset['Class 3'][:, 0], dataset['Class 3'][:, 1], dataset['Class 3'][:, 2], color='b', alpha=0.5, label='Class 3')
ax.scatter(centroids[2][0], centroids[2][1], centroids[0][2], c='b', s=100)
ax.scatter(dataset['Class 4'][:, 0], dataset['Class 4'][:, 1], dataset['Class 4'][:, 2], color='k', alpha=0.5, label='Class 4')
ax.scatter(centroids[3][0], centroids[3][1], centroids[0][2], c='k', s=100)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title(f'K Means - {K} Clusters Classified')
ax.legend()


def animate(i):
    ax.view_init(elev=20, azim=i)


ani = animation.FuncAnimation(fig, animate, frames=360, interval=20)
mywriter = animation.FFMpegWriter(fps=60)
ani.save('Animations/KMeans_3D_rotating.mp4', writer=mywriter)
plt.show()

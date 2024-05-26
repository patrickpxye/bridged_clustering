import numpy as np
import matplotlib.pyplot as plt

def assign_clusters(X, centroids):
    # Assign each sample to the nearest centroid
    clusters = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
    return clusters

def update_centroids(X, clusters, K):
    # Update centroids based on the current cluster assignments
    centroids = np.array([X[clusters == k].mean(axis=0) for k in range(K)])
    return centroids

def majority_vote(labels):
    # Counts the occurrences of each label and returns the label (index) with the highest count
    (values, counts) = np.unique(labels, return_counts=True)
    index = np.argmax(counts)
    return values[index]

def train_k_means(X, Y, K):
    # Initialize centroids by selecting random samples from X and Y
    np.random.seed(0)
    initial_x_centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    initial_y_centroids = Y[np.random.choice(Y.shape[0], K, replace=False)]
    
    # Perform K-Means for inputs (X)
    x_centroids = initial_x_centroids
    for _ in range(100):  # Assume convergence after 100 iterations
        x_clusters = assign_clusters(X, x_centroids)
        new_x_centroids = update_centroids(X, x_clusters, K)
        if np.allclose(x_centroids, new_x_centroids):
            break
        x_centroids = new_x_centroids
    
    # Perform K-Means for outputs (Y)
    y_centroids = initial_y_centroids
    for _ in range(100):  # Assume convergence after 100 iterations
        y_clusters = assign_clusters(Y, y_centroids)
        new_y_centroids = update_centroids(Y, y_clusters, K)
        if np.allclose(y_centroids, new_y_centroids):
            break
        y_centroids = new_y_centroids
    
    return x_clusters, y_clusters, x_centroids, y_centroids

def build_supervised_subset(X, Y, fraction):
    # Sample a fraction of the data to act as a supervised subset
    indices = np.random.choice(X.shape[0], int(X.shape[0] * fraction), replace=False)
    return X[indices], Y[indices], indices

def build_bridging_function(X, Y, x_clusters, y_clusters, supervised_indices):
    # Build a bridging function using only the supervised subset
    bridge = {}
    supervised_x_clusters = x_clusters[supervised_indices]
    supervised_y_clusters = y_clusters[supervised_indices]

    for k in np.unique(supervised_x_clusters):
        indices = np.where(supervised_x_clusters == k)
        # Use majority voting within the supervised set for bridging
        bridge[k] = majority_vote(supervised_y_clusters[indices])

    return bridge

def train_and_build_bridge(X, Y, K, supervised_fraction):
    # Training K-Means on the full dataset
    x_clusters, y_clusters, x_centroids, y_centroids = train_k_means(X, Y, K)
    
    # Create supervised subset
    supervised_X, supervised_Y, supervised_indices = build_supervised_subset(X, Y, supervised_fraction)
    
    # Build bridge using the supervised subset
    bridge = build_bridging_function(X, Y, x_clusters, y_clusters, supervised_indices)

    return x_clusters, y_clusters, x_centroids, y_centroids, bridge

def predict(x, x_centroids, bridge, y_centroids):
    # Prediction for new input x
    cluster = np.argmin(np.linalg.norm(x - x_centroids, axis=1))
    mapped_cluster = bridge[cluster]
    return y_centroids[mapped_cluster]

def plot_clusters(X, clusters, centroids, title='Clusters Visualization'):
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']
    for i in range(len(centroids)):
        # Plot data points belonging to the same cluster
        plt.scatter(X[clusters == i, 0], X[clusters == i, 1], s=30, c=colors[i], label=f'Cluster {i}')
        # Plot centroids
        plt.scatter(centroids[i][0], centroids[i][1], s=200, c='black', marker='X')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()


import numpy as np
from bkm.ss_tests.bkm_utils_2d import *

# Example usage
X = np.random.rand(1000, 5)  # 100 samples, 5 features
Y = np.random.rand(1000, 5)  # Corresponding outputs
K = 3  # Number of clusters
supervised_fraction = 0.1  # 10% as the supervised subset

x_clusters, y_clusters, x_centroids, y_centroids, bridge = train_and_build_bridge(X, Y, K, supervised_fraction)

# Predicting for a new input
new_x = np.random.rand(5)
predicted_output = predict(new_x, x_centroids, bridge, y_centroids)
print(predicted_output)
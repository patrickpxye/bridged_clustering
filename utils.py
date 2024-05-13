import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def calculate_purity(df, cluster_label, true_label):
    total_correct = 0
    for c in df[cluster_label].unique():
        cluster_subset = df[df[cluster_label] == c]
        most_frequent = cluster_subset[true_label].value_counts().idxmax()
        correct_predictions = cluster_subset[true_label].value_counts().max()
        total_correct += correct_predictions
    return total_correct / df.shape[0]


def perform_kmeans(data, I, output=False):
    if output: print(data.head())
    scaler = StandardScaler()
    print("Number of clusters: ", len(data[I].unique()))
    features_scaled = scaler.fit_transform(pd.get_dummies(data.drop(I, axis=1)))
    kmeans = KMeans(n_clusters=len(data[I].unique()), random_state=0)
    kmeans.fit(features_scaled)
    cluster_labels = kmeans.predict(features_scaled)
    subset_data_with_labels = data[[I]].copy()
    subset_data_with_labels['cluster'] = cluster_labels

    silhouette_avg = silhouette_score(features_scaled, kmeans.labels_)
    purity = calculate_purity(subset_data_with_labels, 'cluster', I)
    print("Silhouette Score: ", silhouette_avg)
    print("Purity: ", purity)

    return kmeans, silhouette_avg, purity


def plot_clusters(I, data, model, title):
    # Ensure you're applying PCA to the scaled features used in clustering
    mI = data[I]
    data = data.drop(I, axis=1)
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)
    principal_components = pca.fit_transform(features_scaled)  # Ensure this matches the clustered data
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df['Cluster'] = model.labels_
    principal_df['Individual'] = mI
    n_clusters = len(mI.unique())

    # Create a color map for the clusters
    cmap = plt.cm.get_cmap('viridis', n_clusters)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(principal_df['PC1'], principal_df['PC2'], c=principal_df['Individual'], cmap='viridis', edgecolors='black', label='Individual')

    # Overlay a color mesh to denote clusters
    x_min, x_max = principal_df['PC1'].min() - 1, principal_df['PC1'].max() + 1
    y_min, y_max = principal_df['PC2'].min() - 1, principal_df['PC2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)

    # Adding labels and title
    plt.colorbar()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.show()
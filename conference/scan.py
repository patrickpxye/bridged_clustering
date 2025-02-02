import numpy as np
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans
import warnings
warnings.filterwarnings("ignore")
# from genetic_utils import identify_spec, generate_pca_plot, generate_proportion_plot


df = pd.read_csv('data/gene_spec.csv')
df = df[df['spec'] != 'QB']
data = df.drop(['DNA_ID', 'spec', 'TreeNo'], axis=1)

scaler = StandardScaler()
data = scaler.fit_transform(data)


species_labels = df['spec'].to_numpy()

max_score = 0
max_indecies = []
max_n_clusters = 0

for n_clusters in range(2, 8):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    for i in range(500):

        n_features = np.random.randint(1, 7)
        selected_features = np.random.choice(data.shape[1], n_features, replace=False)
        data_reduced = data[:, selected_features]

        cluster_labels = kmeans.fit_predict(data_reduced)
        kmeans_score = adjusted_rand_score(species_labels, cluster_labels)

        if kmeans_score > max_score:
            max_score = kmeans_score
            max_indecies = selected_features
            max_n_clusters = n_clusters
        
        print(f'KMeans score: {kmeans_score}')

print(f'Max score: {max_score}')
print(f'Max indecies: {max_indecies}')
print(f'Max n_clusters: {max_n_clusters}')
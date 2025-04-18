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


df = pd.read_csv('data/joined_df.csv')
df = df.drop(['TreeNo'], axis=1)

scaler = StandardScaler()
data = scaler.fit_transform(df)

df_spectrum = [range(0,18),range(18,24)]

max_score = 0
max_indecies = []
max_n_clusters = 0

for n_clusters in [3]:
    kmeans_in = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_out = KMeans(n_clusters=n_clusters, random_state=0)

    for i in range(1):

        n_features = np.random.randint(1, len(df_spectrum[0])/2)
        input_features = np.random.choice(df_spectrum[0], n_features, replace=False)
        input_features = [0, 6, 15, 17, 16]
        input_data = data[:, input_features]
        input_labels = kmeans_in.fit_predict(input_data)

        n_features = np.random.randint(1, len(df_spectrum[1]))
        output_features = np.random.choice(df_spectrum[1], n_features, replace=False)
        output_features = [18, 19, 20, 21, 22]
        output_data = data[:, output_features]
        output_labels = kmeans_out.fit_predict(output_data)

        kmeans_score = adjusted_rand_score(input_labels, output_labels)

        if kmeans_score > max_score:
            max_score = kmeans_score
            max_input_features = input_features
            max_output_features = output_features
            max_n_clusters = n_clusters
        
        print(f'KMeans score: {kmeans_score}')

print(f'Max score: {max_score}')
print(f'Max n_clusters: {max_n_clusters}')
print(f'Max input features: {max_input_features}')
print(f'Max output features: {max_output_features}')
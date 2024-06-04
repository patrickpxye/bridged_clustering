import numpy as np
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings("ignore")


# Morphological traits 1 ~ 16
df = pd.read_csv('herb_data.csv')
df = df[df["Putative_spp"] != "Quercus sp."]
df = df[df["Putative_spp"] != "Quercus buckleyi"]
#df = df[df["Putative_spp"] != "Quercus shumardii"]
#df = df[df["Putative_spp"] != "Quercus shumardii var. acerifolia first, Quercus shumardii later"]
df["spp"] = df["Putative_spp"].apply(lambda x: "A" if x == "Quercus shumardii var. acerifolia" else "R" if x == "Quercus rubra" else "S" if x == "Quercus shumardii var. acerifolia first, Quercus shumardii later" or x == "Quercus shumardii" else "Other")
#df["spp"] = df["Putative_spp"].apply(lambda x: "A" if x == "Quercus shumardii var. acerifolia" else "S or R")

data = df[["Lobe.number","BL","PL","BW","TLIW","TLL","TLDW","TEL","BLL","LLL","BSR","LSR","LLDW","LLIW","MidVeinD","BL_PL"]].dropna()


best_ari = 0
best_nmi = 0

for i in range(1000):
    #randomly remove different subsets of columns
    data = df[["Lobe.number","BL","PL","BW","TLIW","TLL","TLDW","TEL","BLL","LLL","BSR","LSR","LLDW","LLIW","MidVeinD","BL_PL"]].dropna()
    biased_list = {'BLL', 'BL', 'TEL', 'TLDW', 'BW', 'TLIW', 'LLL', 'MidVeinD'}

    # probabilities = {col: 0.2 if col in biased_list else 0.8 for col in data.columns}
    # total_prob = sum(probabilities.values())
    # normalized_probabilities = [probabilities[col] / total_prob for col in data.columns]

    # num_columns_to_remove = np.random.randint(1, len(data.columns))  # random number of columns to remove
    # features_to_remove = np.random.choice(data.columns, size=num_columns_to_remove, replace=False, p=normalized_probabilities)
    # data_reduced = data.drop(columns=features_to_remove)

    num_features = len(data.columns)
    indices_to_remove = np.random.choice(num_features, np.random.randint(1, num_features), replace=False)
    features_to_remove = data.columns[indices_to_remove]
    data_reduced = data.drop(columns=features_to_remove)

    transformer = FunctionTransformer(np.log, validate=True)
    data_transformed = transformer.fit_transform(data_reduced)
    data_standardized = StandardScaler().fit_transform(data_transformed)

    sc = SpectralClustering(n_clusters=3, random_state=1, n_init=100)
    sc.fit(data_standardized)
    cluster_labels = sc.labels_
    species_labels = df['spp']

    species_counts = species_labels.value_counts().to_dict()
    species_labels = species_labels.apply(lambda x: 0 if x == "A" else 1)
    species_labels = species_labels.to_numpy()
    ari = adjusted_rand_score(species_labels, cluster_labels)
    nmi = normalized_mutual_info_score(species_labels, cluster_labels)

    if ari > best_ari:
        best_ari = ari
        best_nmi = nmi
        best_features_removed = features_to_remove.tolist()

    print(f"Features removed: {features_to_remove.tolist()}")
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")

print("Best ARI:", best_ari)
print("Best NMI:", best_nmi)
print("Best feature to remove:", best_features_removed)

    



# # Log-transform the data
# transformer = FunctionTransformer(np.log, validate=True)
# data_transformed = transformer.fit_transform(data)
# data_standardized = StandardScaler().fit_transform(data_transformed)

# # Gaussian Mixture Model
# # from sklearn.mixture import GaussianMixture
# # gmm = GaussianMixture(n_components=2, random_state=0)
# # gmm.fit(data_standardized)
# # cluster_labels = gmm.predict(data_standardized)

# # use spectral clustering
# from sklearn.cluster import SpectralClustering
# sc = SpectralClustering(n_clusters=2, random_state=1, n_init=100)
# sc.fit(data_standardized)
# cluster_labels = sc.labels_

# species_labels = df['spp']
# plot_data = pd.DataFrame({'species': species_labels, 'cluster': cluster_labels})
# cluster_counts = plot_data.groupby(['species', 'cluster']).size().unstack(fill_value=0)
# cluster_proportions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)
# cluster_proportions.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))
# plt.title('Herbarium Samples: Morphological Groups by Species')
# plt.xlabel('Species')
# plt.ylabel('Percentage of Morphological Groups')
# plt.legend(title='Morph. Groups', labels=[f'Group {x}' for x in range(1, 15)])
# plt.show()

# #calculate accuracy
# species_counts = species_labels.value_counts().to_dict()
# species_labels = species_labels.apply(lambda x: 0 if x == "A" else 1)
# species_labels = species_labels.to_numpy()
# print("Adjusted Rand Index:", adjusted_rand_score(species_labels, cluster_labels))
# print("Normalized Mutual Information:", normalized_mutual_info_score(species_labels, cluster_labels))
# print(species_counts)
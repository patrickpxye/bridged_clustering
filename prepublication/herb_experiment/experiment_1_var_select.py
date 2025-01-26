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

for i in range(5000):
    #randomly remove different subsets of columns
    #data = df[["Lobe.number","BL","PL","BW","TLIW","TLL","TLDW","TEL","BLL","LLL","BSR","LSR","LLDW","LLIW","MidVeinD","BL_PL"]].dropna()
    data = df[["Lobe.number","PL","TLIW","TLL","TLDW","TEL","BLL","BSR","LSR","LLIW","MidVeinD","BL_PL"]].dropna()

    num_features = len(data.columns)
    indices_to_remove = np.random.choice(num_features, np.random.randint(1, num_features), replace=False)
    features_to_remove = data.columns[indices_to_remove]
    data_reduced = data.drop(columns=features_to_remove)

    data_transformed = FunctionTransformer(np.log, validate=True).fit_transform(data_reduced)
    data_standardized = StandardScaler().fit_transform(data_transformed)

    sc = SpectralClustering(n_clusters=2, random_state=1, n_init=100)
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
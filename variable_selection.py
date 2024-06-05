import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings("ignore")

# Read and preprocess the data
df = pd.read_csv('herb_data.csv')
df = df[~df["Putative_spp"].isin(["Quercus sp.", "Quercus buckleyi"])]
df["spp"] = df["Putative_spp"].apply(lambda x: "A" if x == "Quercus shumardii var. acerifolia" else 
                                      "R" if x == "Quercus rubra" else 
                                      "S" if x in ["Quercus shumardii var. acerifolia first, Quercus shumardii later", "Quercus shumardii"] else 
                                      "Other")

feature_list = ["Lobe.number","BL","PL","BW","TLIW","TLL","TLDW","TEL","BLL","LLL","BSR","LSR","LLDW","LLIW","MidVeinD","BL_PL"]
data = df[feature_list].dropna()

best_ari = 0
best_features = []

selected_features = []

def calculate_ari(feature_subset):
    if not feature_subset:
        return 0  # Return a default low ARI for empty subsets to avoid further processing.

    data_subset = df[feature_subset].dropna()
    
    if data_subset.empty:
        return 0  # Return a default low ARI for empty data frames.

    # Ensure all data is positive before applying logarithmic transformation
    data_transformed = FunctionTransformer(lambda x: np.log(x + 1e-6), validate=True).fit_transform(data_subset)
    data_standardized = StandardScaler().fit_transform(data_transformed)

    # Spectral Clustering
    sc = SpectralClustering(n_clusters=2, random_state=1, n_init=100)
    sc.fit(data_standardized)
    cluster_labels = sc.labels_
    species_labels = df['spp'].apply(lambda x: 0 if x == "A" else 1).to_numpy()
    
    # Calculate ARI
    ari = adjusted_rand_score(species_labels, cluster_labels)
    return ari


def pseudo_cost_branching(selected_features, remaining_features):
    Di_plus = {}
    Di_minus = {}
    
    current_ari = calculate_ari(selected_features)
    
    for feature in remaining_features:
        # Include the feature
        ari_with_feature = calculate_ari(selected_features + [feature])
        Di_plus[feature] = ari_with_feature - current_ari
        
        # Exclude the feature (implicitly, as we don't include it in the first place)
        selected_features_without_feature = [f for f in selected_features if f != feature]
        ari_without_feature = calculate_ari(selected_features_without_feature)
        Di_minus[feature] = current_ari - ari_without_feature
    
    # Select the feature that maximizes Di+ * Di-
    best_feature = max(remaining_features, key=lambda f: Di_plus[f] * Di_minus[f])
    
    return best_feature

remaining_features = feature_list.copy()

while remaining_features:
    best_feature = pseudo_cost_branching(selected_features, remaining_features)
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)
    
    current_ari = calculate_ari(selected_features)
    if current_ari > best_ari:
        best_ari = current_ari
        best_features = selected_features.copy()
    
    print(f"Selected feature: {best_feature}")
    print(f"Current ARI: {current_ari}")

# Display the best results
print("Best ARI:", best_ari)
print("Best features:", best_features)
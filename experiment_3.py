import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('herb_data_filtered.csv')
feature_list_1 =["Lobe.number","BW","TLIW","LLL","LSR","LLDW","LLIW"]
feature_list_2 = ["Latitude", "Longitude","LLL.LLLDW","BL.BW"]

data = df[feature_list_1].dropna()

transformer = FunctionTransformer(np.log, validate=True)
data_transformed = transformer.fit_transform(data)
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_transformed)

spectral = SpectralClustering(n_clusters=2, random_state=0)
cluster_labels = spectral.fit_predict(data_standardized)
knn = KNeighborsClassifier()
knn.fit(data_standardized, cluster_labels)

df["Longitude"] = df["Longitude"].abs()
loc_data = df[feature_list_2].dropna()

transformer_2 = FunctionTransformer(np.log, validate=True)
loc_data_transformed = transformer_2.fit_transform(loc_data)
scaler_2 = StandardScaler()
loc_data_standardized = scaler_2.fit_transform(loc_data_transformed)

loc_spec = SpectralClustering(n_clusters=2, random_state=0)
loc_cluster_labels = loc_spec.fit_predict(loc_data_standardized)
loc_knn = KNeighborsClassifier()
loc_knn.fit(loc_data_standardized, loc_cluster_labels)



# Assume data_standardized is already defined and is your standardized dataset
norms = [np.linalg.norm(data) for data in data_standardized]
kde = gaussian_kde(norms)
x = np.linspace(min(norms), max(norms), 2000)
density = kde(x)

# Plot the density
plt.figure(figsize=(8, 6))  # You can adjust the size of the figure here
plt.plot(x, density, label='Random Guess')
plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization




score = 0
k = 10000
distances = []
for j in range(k):
    test_data = data.sample(1)
    test_id = test_data.index

    test_point = scaler.transform(transformer.transform(test_data))
    test_point_cluster = knn.predict(test_point)[0]

    #collect the data of group 1 as a set, and group 2 as another set
    loc_group_a = loc_data_standardized[loc_cluster_labels == 0]
    loc_group_b = loc_data_standardized[loc_cluster_labels == 1]
    loc_group_a_centroid = loc_group_a.mean(axis=0).reshape(1, -1)
    loc_group_b_centroid = loc_group_b.mean(axis=0).reshape(1, -1)
    test_data_ground_truth = loc_data_standardized[test_data.index[0]].reshape(1, -1)
    distance_from_group_a = np.linalg.norm(test_data_ground_truth - loc_group_a_centroid)
    distance_from_group_b = np.linalg.norm(test_data_ground_truth - loc_group_b_centroid)
    actual_closer_group = np.argmin([distance_from_group_a, distance_from_group_b])

    decision_vector = np.array([0,1])
    test_result = decision_vector[test_point_cluster]

    if test_result == actual_closer_group:
        score = score + 1
    if test_result == 0:
        distances.append(distance_from_group_a)
    else:
        distances.append(distance_from_group_b)

print("Accuracy: ", score/k)

kde = gaussian_kde(distances)
x = np.linspace(min(distances), max(distances), 1000)
density = kde(x)

# Plot the density
plt.plot(x, density, label='Bridged Clustering')
plt.fill_between(x, density, alpha=0.5)
plt.title('Loss in Distance Prediction')
plt.xlabel('Normalized Loss')
plt.ylabel('Density')
plt.legend()
plt.savefig('experiment_3.png')
plt.show()

norms_mean = np.mean(norms)
distances_mean = np.mean(np.array(distances))
print("Error Reduction: ", (norms_mean - distances_mean) / norms_mean)

#Accuracy:  0.9354
#Error Reduction:  0.3076861605157626
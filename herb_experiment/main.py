# This script is the executable version of the Experiment 3 Notebook.

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


def decision(matrix):
    max_indices = np.argmax(matrix, axis=1)
    if max_indices[0] == max_indices[1]:
        differences = np.abs(np.diff(matrix, axis=1)).flatten()
        row_with_smaller_difference = np.argmin(differences)
        max_indices[row_with_smaller_difference] = 1 - max_indices[row_with_smaller_difference]
    return max_indices



df = pd.read_csv('herb_data_filtered.csv')
#feature_list_1 =["Lobe.number","BW","TLIW","LLL","LSR","LLDW","LLIW"]
feature_list_1 = ["Lobe.number","PL","TLIW","TLL","TLDW","BLL","LSR","LLIW"]
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
sup = 5
distances = []
neighbour_locs = []
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

    #get 3 points closest to the test point
    random_indices = np.random.choice(len(data_standardized), sup, replace=False)
    ten_data = data_standardized[random_indices]
    ten_loc_data = loc_data_standardized[random_indices]
    n_distances = np.linalg.norm(ten_data - test_point, axis=1)
    closest_points = np.argsort(n_distances)[:3]
    average_location = np.mean(ten_loc_data[closest_points], axis=0).reshape(1, -1)
    neighbour_locs.append(np.linalg.norm(average_location - test_data_ground_truth))

    decision_matrix = np.zeros((2, 2))
    for i in random_indices:
        supervised_pt = df.loc[i].to_frame().T
        #keep only 16 columns of supervised data
        gmm_pt = supervised_pt[feature_list_1].dropna()
        kmeans_pt = supervised_pt[feature_list_2].dropna()
        pred1 = knn.predict(scaler.transform(transformer.transform(gmm_pt)))
        pred2 = loc_knn.predict(scaler_2.transform(transformer_2.transform(kmeans_pt)))
        decision_matrix[pred1[0]][pred2[0]] += 1

    decision_vector = decision(decision_matrix)

    decision_vector = np.array([0,1])
    test_result = decision_vector[test_point_cluster]

    if test_result == actual_closer_group:
        score = score + 1
    if test_result == 0:
        distances.append(distance_from_group_a)
    else:
        distances.append(distance_from_group_b)

print("Accuracy: ", score/k)


kde = gaussian_kde(neighbour_locs)
x = np.linspace(min(norms), max(norms), 2000)
density = kde(x)

plt.plot(x, density, label='KNN')
plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization



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
neighbour_mean = np.mean(np.array(neighbour_locs))
print("Error Reduction: ", (neighbour_mean - distances_mean) / neighbour_mean)

#Accuracy:  0.9354
#Error Reduction:  0.3076861605157626
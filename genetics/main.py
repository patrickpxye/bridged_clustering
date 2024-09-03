import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")


def simgleExperiment(sample_size, n_neighbors):

    gene_df = pd.read_csv('data/gene_spec.csv')
    morph_df = pd.read_csv('data/morph.csv')

    morph_df = morph_df.drop(['idx', 'Date', 'classification', 'uncertainty','Latitude', 'Longitude', 'Altitude.ft', 'Multi.Single.stem', 'General.location.Habitat', 'site', 'Putative_spp'], axis=1)
    morph_df = morph_df.dropna()
    morph_df = morph_df.replace(0, 1e-10)

    data = morph_df.drop(['TreeNo'], axis=1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data[:, [0, 11, 7, 12, 16]]
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    morph_df['morph_cluster'] = cluster_labels

    gene_df = gene_df[gene_df['spec'] != 'QB']
    gene_df = gene_df.drop(['DNA_ID', 'spec'], axis=1)
    data = gene_df.drop(['TreeNo'], axis=1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data[:, [1,2,0]]
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    gene_df['gene_cluster'] = cluster_labels

    joined_df= morph_df.merge(gene_df, on='TreeNo')

    # sample a random batch of rows
    sample = joined_df.sample(sample_size)

    # build an association matrix
    association_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            association_matrix[i,j] = np.sum((sample['morph_cluster'] == i) & (sample['gene_cluster'] == j))

    # Initialize decision array
    decision = np.zeros(3, dtype=int)

    # Create a copy of the association matrix to manipulate
    temp_matrix = association_matrix.copy()

    # Fill the decision array
    for _ in range(3):
        max_index = np.argmax(temp_matrix)
        max_location = np.unravel_index(max_index, temp_matrix.shape)
        decision[max_location[0]] = max_location[1]
        temp_matrix[max_location[0], :] = -np.inf
        temp_matrix[:, max_location[1]] = -np.inf

    morph_df["morph_predicted_gene_cluster"] = morph_df.apply(lambda x: decision[x['morph_cluster']], axis=1)

    average_gene = gene_df.groupby('gene_cluster').agg({'PC1': 'mean', 'PC2': 'mean', 'PC3': 'mean'}).reset_index()
    average_gene['gene_centroids'] = average_gene[['PC1', 'PC2', 'PC3']].values.tolist()
    average_gene = average_gene[['gene_cluster', 'gene_centroids']]

    gene_df['gene_coordinates'] = gene_df[['PC1', 'PC2', 'PC3']].values.tolist()
    gene_df = gene_df[['TreeNo', 'gene_coordinates']]
    joined_df = morph_df.merge(gene_df, on='TreeNo').drop(['morph_cluster'], axis=1)

    joined_df = joined_df.rename(columns={'morph_predicted_gene_cluster': 'gene_cluster'})
    final_df = joined_df.merge(average_gene, on='gene_cluster')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['gene_centroids']))
    final_df['bkm_distance'] = final_df.apply(euclidean_distance, axis=1)


    ############# KNN #############

    morph_df = pd.read_csv('data/morph.csv')
    morph_df = morph_df.drop(['idx', 'Date', 'classification', 'uncertainty','Latitude', 'Longitude', 'Altitude.ft', 'Multi.Single.stem', 'General.location.Habitat', 'site', 'Putative_spp'], axis=1)
    morph_df = morph_df.dropna()
    morph_df = morph_df.replace(0, 1e-10)

    data = morph_df.drop(['TreeNo'], axis=1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    morph_df[["a", "b", "c", "d", "e"]] = data[:, [0, 11, 7, 12, 16]]

    # convert to numpy matrix
    morph_df['morph_coordinates'] = morph_df[['a', 'b', 'c', 'd', 'e']].values.tolist()

    joined_df= morph_df.merge(gene_df, on='TreeNo')
    joined_df = joined_df[['TreeNo', 'morph_coordinates', 'gene_coordinates']]

    sample = joined_df.sample(sample_size)

    X = np.array(sample['morph_coordinates'].tolist())
    y = np.array(sample['gene_coordinates'].tolist())

    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X, y)

    morph_df['knn_gene_predictions'] = morph_df['morph_coordinates'].apply(lambda x: knn_regressor.predict([x])[0])
    morph_df = morph_df[['TreeNo', 'knn_gene_predictions']]

    final_df = morph_df.merge(final_df, on='TreeNo')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['knn_gene_predictions']))

    final_df['knn_distance'] = final_df.apply(euclidean_distance, axis=1)

    final_df = final_df[['TreeNo', 'bkm_distance', 'knn_distance']]

    result_bkm = list(final_df['bkm_distance'])
    result_knn = list(final_df['knn_distance'])

    return result_bkm, result_knn


def plotDensityGraph(result_bkm, result_knn):

    kde = gaussian_kde(result_bkm)
    x = np.linspace(min(result_bkm), max(result_bkm), 2000)
    density = kde(x)

    plt.plot(x, density, label='Bridged Clustering')
    plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization

    mean_bkm = np.mean(result_bkm)
    plt.axvline(mean_bkm, color='blue', linestyle='dashed', linewidth=1)
    plt.text(mean_bkm, max(density) * 0.9, f'Mean: {mean_bkm:.2f}', color='blue', ha='center')


    kde = gaussian_kde(result_knn)
    x = np.linspace(min(result_knn), max(result_knn), 2000)
    density = kde(x)

    plt.plot(x, density, label='KNN')
    plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization

    mean_knn = np.mean(result_knn)
    plt.axvline(mean_knn, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_knn, max(density) * 0.9, f'Mean: {mean_knn:.2f}', color='red', ha='center')

    plt.title('Average Deviation in Distance Prediction')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('density_graph.png')
    plt.show()


result_bkm, result_knn = [], []
for i in range(100):
    bkm, knn = simgleExperiment(10, 1)
    result_bkm.extend(bkm)
    result_knn.extend(knn)
plotDensityGraph(result_bkm, result_knn)

# score_board = {"bkm": 0, "knn": 0}
# for a in [1, 2, 3, 4, 5]:
#     for b in [5, 10, 15, 20, 25, 30]:
#         result_bkm, result_knn = [], []
#         for i in range(100):
#             bkm, knn = simgleExperiment(b, a)
#             result_bkm.extend(bkm)
#             result_knn.extend(knn)
#             #take mean of all the results
#         mean_bkm = np.mean(result_bkm)
#         mean_knn = np.mean(result_knn)
#         if mean_bkm > mean_knn:
#             score_board['knn'] += 1
#             print(a, b)
#         else:
#             score_board['bkm'] += 1

# print(score_board)


# # Assume data_standardized is already defined and is your standardized dataset
# norms = [np.linalg.norm(data) for data in data_standardized]
# kde = gaussian_kde(norms)
# x = np.linspace(min(norms), max(norms), 2000)
# density = kde(x)

# # Plot the density
# plt.figure(figsize=(8, 6))  # You can adjust the size of the figure here
# plt.plot(x, density, label='Random Guess')
# plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization


# norms_mean = np.mean(norms)
# distances_mean = np.mean(np.array(distances))
# neighbour_mean = np.mean(np.array(neighbour_locs))
# print("Error Reduction: ", (neighbour_mean - distances_mean) / neighbour_mean)

#Accuracy:  0.9354
#Error Reduction:  0.3076861605157626
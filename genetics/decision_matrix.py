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


def simgleExperiment(sample_size):

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

    return decision


truth = np.array([1, 2, 0])


accuracies = []
for n_samples in range(3, 115):
    iter = 1000
    count = 0
    for i in range(iter):
        res = simgleExperiment(n_samples)
        if np.array_equal(res, truth):
            count += 1
    accuracies.append(count/iter)
    print("Sample size: ", n_samples, "Accuracy: ", count/iter)

# Plot the results
axiss = np.arange(3, 115) / 115
print(axiss)
plt.plot(axiss, accuracies)
plt.xlabel('Percentage of Fully Supervised Samples')
plt.ylabel('Accuracy in Cluster Association')
plt.title('Association Accuracy vs Size of Supervised Set')
plt.show()
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



def preprocessMorphologicalData():
    morph_df = pd.read_csv('data/morph.csv')
    morph_df = morph_df.drop(['idx', 'Date', 'classification', 'uncertainty','Latitude', 'Longitude', 'Altitude.ft', 'Multi.Single.stem', 'General.location.Habitat', 'site', 'Putative_spp'], axis=1)
    morph_df = morph_df.dropna()
    morph_df = morph_df.replace(0, 1e-10)
    return morph_df

def fitMorphologicalClusters(morph_df):
    data = morph_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data[:, [0, 7, 11, 12, 16]]
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    morph_df['morph_coordinates'] = data.tolist()
    morph_df['morph_cluster'] = cluster_labels

    return morph_df, kmeans


def predictMorphologicalClusters(morph_df, kmeans):
    data = morph_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data[:, [0, 7, 11, 12, 16]]
    cluster_labels = kmeans.predict(data)

    morph_df['morph_coordinates'] = data.tolist()
    morph_df['morph_cluster'] = cluster_labels

    return morph_df

def preprocessGeneData():
    gene_df = pd.read_csv('data/gene_spec.csv')
    gene_df = gene_df[gene_df['spec'] != 'QB']
    gene_df = gene_df.drop(['DNA_ID', 'spec'], axis=1)
    return gene_df

def fitGeneClusters(gene_df):
    data = gene_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data[:, [0,1,2]]
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    gene_df['gene_cluster'] = cluster_labels

    return gene_df, kmeans

def predictGeneClusters(gene_df, kmeans):
    data = gene_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data[:, [0,1,2]]
    cluster_labels = kmeans.predict(data)

    gene_df['gene_cluster'] = cluster_labels

    return gene_df


def decisionVector(sample, dim=3):
    
        # build an association matrix
        association_matrix = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                association_matrix[i,j] = np.sum((sample['morph_cluster'] == i) & (sample['gene_cluster'] == j))
    
        # Initialize decision array
        decision = np.zeros(dim, dtype=int)
    
        # Create a copy of the association matrix to manipulate
        temp_matrix = association_matrix.copy()
    
        # Fill the decision array
        for _ in range(dim):
            max_index = np.argmax(temp_matrix)
            max_location = np.unravel_index(max_index, temp_matrix.shape)
            decision[max_location[0]] = max_location[1]
            temp_matrix[max_location[0], :] = -np.inf
            temp_matrix[:, max_location[1]] = -np.inf
    
        return decision


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


def bridgedClustering(super_df, gene_df, test_morph_df, test_gene_df):

    # build associative decisions
    decision = decisionVector(super_df)
    test_morph_df["gene_cluster"] = test_morph_df.apply(lambda x: decision[x['morph_cluster']], axis=1)

    average_gene = gene_df.groupby('gene_cluster').agg({'PC1': 'mean', 'PC2': 'mean', 'PC3': 'mean'}).reset_index()
    average_gene['gene_centroids'] = average_gene[['PC1', 'PC2', 'PC3']].values.tolist()
    average_gene = average_gene[['gene_cluster', 'gene_centroids']]

    test_gene_df['gene_coordinates'] = test_gene_df[['PC1', 'PC2', 'PC3']].values.tolist()
    test_joined_df = test_morph_df.merge(test_gene_df[['TreeNo', 'gene_coordinates']], on='TreeNo').drop(['morph_cluster'], axis=1)
    final_df = test_joined_df.merge(average_gene, on='gene_cluster')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['gene_centroids']))
    final_df['bkm_distance'] = final_df.apply(euclidean_distance, axis=1)

    return final_df


def baselineKNN(super_df, n_neighbors, final_df, test_morph_df):

    super_df['gene_coordinates'] = super_df[['PC1', 'PC2', 'PC3']].values.tolist()

    X = np.array(super_df['morph_coordinates'].tolist())
    y = np.array(super_df['gene_coordinates'].tolist())

    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X, y)

    test_morph_df['knn_gene_predictions'] = test_morph_df['morph_coordinates'].apply(lambda x: knn_regressor.predict([x])[0])
    test_morph_df = test_morph_df[['TreeNo', 'knn_gene_predictions']]

    final_df = test_morph_df.merge(final_df, on='TreeNo')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['knn_gene_predictions']))
    final_df['knn_distance'] = final_df.apply(euclidean_distance, axis=1)

    return final_df


def simgleExperiment(sample_size, n_neighbors):

    gene_df = preprocessGeneData()
    morph_df = preprocessMorphologicalData()

    common_treenos = set(gene_df['TreeNo']).intersection(set(morph_df['TreeNo']))
    test_treenos = list(common_treenos)[int(len(common_treenos)*0.7):]
    super_treenos = np.random.choice(list(common_treenos), size=sample_size, replace=False)
    test_morph_df = morph_df[morph_df['TreeNo'].isin(test_treenos)]
    test_gene_df = gene_df[gene_df['TreeNo'].isin(test_treenos)]
    train_morph_df = morph_df[~morph_df['TreeNo'].isin(test_treenos)]
    train_gene_df = gene_df[~gene_df['TreeNo'].isin(test_treenos)]

    morph_df, morph_kmeans = fitMorphologicalClusters(train_morph_df)
    gene_df, gene_kmeans = fitGeneClusters(train_gene_df)

    test_morph_df = predictMorphologicalClusters(test_morph_df, morph_kmeans)
    test_gene_df = predictGeneClusters(test_gene_df, gene_kmeans)

    joined_df = morph_df.merge(gene_df, on='TreeNo')
    super_df = joined_df[joined_df['TreeNo'].isin(super_treenos)]
    final_df = bridgedClustering(super_df, gene_df, test_morph_df, test_gene_df)
    final_df = baselineKNN(super_df, n_neighbors, final_df, test_morph_df)

    final_df = final_df[['TreeNo', 'bkm_distance', 'knn_distance']]

    result_bkm = list(final_df['bkm_distance'])
    result_knn = list(final_df['knn_distance'])

    return result_bkm, result_knn



result_bkm, result_knn = [], []
for i in range(100):
    bkm, knn = simgleExperiment(10, 2)
    result_bkm.extend(bkm)
    result_knn.extend(knn)
plotDensityGraph(result_bkm, result_knn)
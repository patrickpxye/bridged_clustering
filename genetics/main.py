import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")



def preprocessMorphologicalData():
    morph_df = pd.read_csv('data/morph.csv')
    morph_df = morph_df.drop(['idx', 'Date', 'classification', 'uncertainty','Latitude', 'Longitude', 'Altitude.ft', 'Multi.Single.stem', 'General.location.Habitat', 'site', 'Putative_spp'], axis=1)
    morph_df = morph_df.dropna()
    morph_df = morph_df.replace(0, 1e-10)
    return morph_df

def fitMorphologicalClusters(morph_df, full=False):
    data = morph_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if not full:
        data = data[:, [0, 7, 11, 12, 16]]
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    morph_df['morph_coordinates'] = data.tolist()
    morph_df['morph_cluster'] = cluster_labels

    return morph_df, kmeans


def predictMorphologicalClusters(morph_df, kmeans, full=False):
    data = morph_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if not full:
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


def plotDensityGraph(result_bkm, result_knn, result_lin):

    kde = gaussian_kde(result_bkm)
    x = np.linspace(min(result_bkm), max(result_bkm), 2000)
    density = kde(x)

    plt.plot(x, density, label='Bridged Clustering')
    plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization

    mean_bkm = np.mean(result_bkm)
    print("mean distance for bkm is ", mean_bkm)
    plt.axvline(mean_bkm, color='blue', linestyle='dashed', linewidth=1)
    # plt.text(mean_bkm, max(density) * 0.9, f'Mean: {mean_bkm:.2f}', color='blue', ha='center')

    kde = gaussian_kde(result_lin)
    x = np.linspace(min(result_lin), max(result_lin), 2000)
    density = kde(x)

    plt.plot(x, density, label='Linear Regression')
    plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization

    mean_lin = np.mean(result_lin)
    print("mean distance for linear regression is ", mean_lin)
    plt.axvline(mean_lin, color='green', linestyle='dashed', linewidth=1)

    # colors = ['orange', 'red', 'pink', 'white']
    # for n_neighbors, color in zip(range(1, 4), colors):
    #     kde = gaussian_kde(result_knn[n_neighbors])
    #     x = np.linspace(min(result_knn[n_neighbors]), max(result_knn[n_neighbors]), 2000)
    #     density = kde(x)

    #     plt.plot(x, density, label=f'KNN (k={n_neighbors})', color=color)
    #     plt.fill_between(x, density, alpha=0.1)

    #     mean_knn = np.mean(result_knn[n_neighbors])
    #     print("mean distance for k = ", n_neighbors, " is ", mean_knn)
    #     plt.axvline(mean_knn, color=color, linestyle='dashed', linewidth=1)


    # kde = gaussian_kde(result_knn)
    # x = np.linspace(min(result_knn), max(result_knn), 2000)
    # density = kde(x)

    # plt.plot(x, density, label='KNN')
    # plt.fill_between(x, density, alpha=0.5)  # Fill under the curve for better visualization

    # mean_knn = np.mean(result_knn)
    # plt.axvline(mean_knn, color='red', linestyle='dashed', linewidth=1)
    # # plt.text(mean_knn, max(density) * 0.9, f'Mean: {mean_knn:.2f}', color='red', ha='center')

    # plt.title('Euclidean Distance between Predicted and True Gene Coordinates', fontsize=16)
    plt.xlabel('Euclidean Distance', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig('density_graph-33.pdf')
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
    test_joined_df = test_joined_df.merge(average_gene, on='gene_cluster')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['gene_centroids']))
    test_joined_df['bkm_distance'] = test_joined_df.apply(euclidean_distance, axis=1)
    result_bkm = list(test_joined_df['bkm_distance'])

    return result_bkm


def baselineKNN(super_df, n_neighbors, test_morph_df, test_gene_df):

    super_df['gene_coordinates'] = super_df[['PC1', 'PC2', 'PC3']].values.tolist()

    X = np.array(super_df['morph_coordinates'].tolist())
    y = np.array(super_df['gene_coordinates'].tolist())

    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X, y)

    test_morph_df['knn_gene_predictions'] = test_morph_df['morph_coordinates'].apply(lambda x: knn_regressor.predict([x])[0])
    test_gene_df['gene_coordinates'] = test_gene_df[['PC1', 'PC2', 'PC3']].values.tolist()
    test_morph_df = test_morph_df[['TreeNo', 'knn_gene_predictions']]

    test_joined_df = test_morph_df.merge(test_gene_df[['TreeNo', 'gene_coordinates']], on='TreeNo')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['knn_gene_predictions']))
    test_joined_df['knn_distance'] = test_joined_df.apply(euclidean_distance, axis=1)
    result_knn = list(test_joined_df['knn_distance'])

    return result_knn

def baselineLinear(super_df, test_morph_df, test_gene_df):
    # Combine PC1, PC2, PC3 into a single 'gene_coordinates' column
    super_df['gene_coordinates'] = super_df[['PC1', 'PC2', 'PC3']].values.tolist()

    # Prepare the feature matrix (X) and target matrix (y)
    X = np.array(super_df['morph_coordinates'].tolist())
    y = np.array(super_df['gene_coordinates'].tolist())

    # Train a Linear Regression model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)

    # Predict on the test morph dataset
    test_morph_df['lin_gene_predictions'] = test_morph_df['morph_coordinates'].apply(
        lambda x: linear_regressor.predict([x])[0]
    )

    # Prepare the true gene coordinates in the test set
    test_gene_df['gene_coordinates'] = test_gene_df[['PC1', 'PC2', 'PC3']].values.tolist()

    # Keep only necessary columns
    test_morph_df = test_morph_df[['TreeNo', 'lin_gene_predictions']]

    # Join predicted values with the true gene coordinates
    test_joined_df = test_morph_df.merge(
        test_gene_df[['TreeNo', 'gene_coordinates']], 
        on='TreeNo'
    )

    # Compute Euclidean distance between true and predicted gene coordinates
    def euclidean_distance(row):
        return np.linalg.norm(
            np.array(row['gene_coordinates']) - np.array(row['lin_gene_predictions'])
        )

    test_joined_df['lin_distance'] = test_joined_df.apply(euclidean_distance, axis=1)
    result_lin = list(test_joined_df['lin_distance'])

    return result_lin


def simgleExperiment(super_size, n_neighbors, test_size=20):

    gene_df = preprocessGeneData()
    morph_df = preprocessMorphologicalData()

    common_treenos = set(gene_df['TreeNo']).intersection(set(morph_df['TreeNo']))
    test_treenos = np.random.choice(list(common_treenos), size=test_size, replace=False)
    super_treenos = np.random.choice(list(common_treenos - set(test_treenos)), size=super_size, replace=False)
    test_morph_df = morph_df[morph_df['TreeNo'].isin(test_treenos)]
    test_gene_df = gene_df[gene_df['TreeNo'].isin(test_treenos)]
    train_morph_df = morph_df[~morph_df['TreeNo'].isin(test_treenos)]
    train_gene_df = gene_df[~gene_df['TreeNo'].isin(test_treenos)]

    morph_df, morph_kmeans = fitMorphologicalClusters(train_morph_df, full=True)
    gene_df, gene_kmeans = fitGeneClusters(train_gene_df)

    test_morph_df = predictMorphologicalClusters(test_morph_df, morph_kmeans, full=True)
    test_gene_df = predictGeneClusters(test_gene_df, gene_kmeans)

    joined_df = morph_df.merge(gene_df, on='TreeNo')
    super_df = joined_df[joined_df['TreeNo'].isin(super_treenos)]
    result_bkm = bridgedClustering(super_df, gene_df, test_morph_df, test_gene_df)
    # result_knn = baselineKNN(super_df, n_neighbors, test_morph_df, test_gene_df)
    result_knn = {}
    # for n_neighbors in range(1, 4):
    #     result = baselineKNN(super_df, n_neighbors, test_morph_df, test_gene_df)
    #     result_knn[n_neighbors] = result
    
    result_lin = baselineLinear(super_df, test_morph_df, test_gene_df)

    return result_bkm, result_knn, result_lin



result_bkm = []
result_lin = []
result_knn = {n: [] for n in range(1, 4)}
lsls= [6, 11, 17, 22, 28, 33]
for i in range(1000):
    for super_size in [lsls[5]]:
        bkm, knn, lin = simgleExperiment(super_size,1)
        result_bkm.extend(bkm)
        result_lin.extend(lin)
        # for n_neighbors in range(1, 4):
        #     result_knn[n_neighbors].extend(knn[n_neighbors])

plotDensityGraph(result_bkm, result_knn, result_lin)
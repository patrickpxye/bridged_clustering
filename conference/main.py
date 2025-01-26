import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from utils import decisionVector, plotDensityGraph
from bio.input import preprocessMorphData, fitMorphClusters, predictMorphClusters
from bio.output import preprocessGeneData, fitGeneClusters, predictGeneClusters

import warnings
warnings.filterwarnings("ignore")



def bridgedClustering(super_df, gene_df, test_morph_df, test_gene_df):

    # build associative decisions
    decision = decisionVector(super_df, n_families)
    test_morph_df["gene_cluster"] = test_morph_df.apply(lambda x: decision[x['morph_cluster']], axis=1)

    average_gene = gene_df.groupby('gene_cluster')[outputfeaturelist].mean().reset_index()
    average_gene['gene_centroids'] = average_gene[outputfeaturelist].values.tolist()
    average_gene = average_gene[['gene_cluster', 'gene_centroids']]

    test_gene_df['gene_coordinates'] = test_gene_df[outputfeaturelist].values.tolist()
    test_joined_df = test_morph_df.merge(test_gene_df[['TreeNo', 'gene_coordinates']], on='TreeNo').drop(['morph_cluster'], axis=1)
    test_joined_df = test_joined_df.merge(average_gene, on='gene_cluster')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['gene_centroids']))
    test_joined_df['bkm_distance'] = test_joined_df.apply(euclidean_distance, axis=1)
    result_bkm = list(test_joined_df['bkm_distance'])

    return result_bkm


def baselineKNN(super_df, n_neighbors, test_morph_df, test_gene_df):

    super_df['gene_coordinates'] = super_df[outputfeaturelist].values.tolist()

    X = np.array(super_df['morph_coordinates'].tolist())
    y = np.array(super_df['gene_coordinates'].tolist())

    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X, y)

    test_morph_df['knn_gene_predictions'] = test_morph_df['morph_coordinates'].apply(lambda x: knn_regressor.predict([x])[0])
    test_gene_df['gene_coordinates'] = test_gene_df[outputfeaturelist].values.tolist()
    test_morph_df = test_morph_df[['TreeNo', 'knn_gene_predictions']]

    test_joined_df = test_morph_df.merge(test_gene_df[['TreeNo', 'gene_coordinates']], on='TreeNo')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['knn_gene_predictions']))
    test_joined_df['knn_distance'] = test_joined_df.apply(euclidean_distance, axis=1)
    result_knn = list(test_joined_df['knn_distance'])

    return result_knn


def baselineLinear(super_df, test_morph_df, test_gene_df):

    super_df['gene_coordinates'] = super_df[outputfeaturelist].values.tolist()

    X = np.array(super_df['morph_coordinates'].tolist())
    y = np.array(super_df['gene_coordinates'].tolist())

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)

    test_morph_df['lin_gene_predictions'] = test_morph_df['morph_coordinates'].apply(lambda x: linear_regressor.predict([x])[0])
    test_gene_df['gene_coordinates'] = test_gene_df[outputfeaturelist].values.tolist()
    test_morph_df = test_morph_df[['TreeNo', 'lin_gene_predictions']]

    test_joined_df = test_morph_df.merge(test_gene_df[['TreeNo', 'gene_coordinates']], on='TreeNo')

    def euclidean_distance(row):
        return np.linalg.norm(np.array(row['gene_coordinates']) - np.array(row['lin_gene_predictions']))
    test_joined_df['lin_distance'] = test_joined_df.apply(euclidean_distance, axis=1)
    result_lin = list(test_joined_df['lin_distance'])

    return result_lin


def simgleExperiment(super_size, n_families, n_neighbors, test_size=20):

    ### Dataset Preparation

    # read and preprocess data
    morph_df = preprocessMorphData()
    gene_df = preprocessGeneData()

    # isolate fully labeled dataset: label a fraction as test set and the rest as fully supervised set
    common_nos = set(gene_df['TreeNo']).intersection(set(morph_df['TreeNo']))
    test_nos = np.random.choice(list(common_nos), size=test_size, replace=False)
    super_nos = np.random.choice(list(common_nos - set(test_nos)), size=super_size, replace=False)

    # create test set, then use the rest of the data for training
    test_morph_df = morph_df[morph_df['TreeNo'].isin(test_nos)]
    test_gene_df = gene_df[gene_df['TreeNo'].isin(test_nos)]
    train_morph_df = morph_df[~morph_df['TreeNo'].isin(test_nos)]
    train_gene_df = gene_df[~gene_df['TreeNo'].isin(test_nos)]

    ## Bridged Clustering

    # train clustering models; formalize data as coordinates; predict clusters for train set, only useful for the fully supervised portion
    morph_df, morph_kmeans = fitMorphClusters(train_morph_df, n_families, full_feat=False)
    gene_df, gene_kmeans = fitGeneClusters(train_gene_df, n_families, full_feat=False)

    # predict cluster assignment for test set, only useful for the bridged clustering method
    test_morph_df = predictMorphClusters(test_morph_df, morph_kmeans, full_feat=False)
    test_gene_df = predictGeneClusters(test_gene_df, gene_kmeans, full_feat=False)

    # build cluster associations through supervised set, and run bridged clustering on test set
    joined_df = morph_df.merge(gene_df, on='TreeNo')
    super_df = joined_df[joined_df['TreeNo'].isin(super_nos)]
    result_bkm = bridgedClustering(super_df, gene_df, test_morph_df, test_gene_df)

    ## Baseline Models

    # instantiate baseline models, train with fully supervised set, and test with test set
    result_knn = baselineKNN(super_df, n_neighbors, test_morph_df, test_gene_df)
    result_lin = baselineLinear(super_df, test_morph_df, test_gene_df)

    return result_bkm, result_knn, result_lin


result_bkm, result_knn, result_lin = [], [], []
samplesize = [6, 11, 17, 22, 28, 33]
n_families = 3
outputfeaturelist = ['PC1', 'PC2', 'PC3']
for i in range(10):
    for super_size in [samplesize[2]]:
        bkm, knn, lin = simgleExperiment(super_size,n_families,1)
        result_bkm.extend(bkm)
        result_lin.extend(lin)
        result_knn.extend(knn)

plotDensityGraph(result_bkm, result_knn, result_lin)
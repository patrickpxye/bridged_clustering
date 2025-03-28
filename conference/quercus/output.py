import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# feature_list = [0,4]
feature_list = [0]

# Reads csv file and return preprocessed dataframe with only numerical columns
def preprocessGeneData():
    gene_df = pd.read_csv('data/gene_spec.csv')
    gene_df = gene_df[gene_df['spec'] != 'QB']
    gene_df = gene_df.drop(['DNA_ID', 'spec'], axis=1)
    return gene_df

# train clustering model; formalize data as coordinates; predict clusters for train set, only useful for the fully supervised portion
def fitGeneClusters(gene_df, n_families, full_feat=False):

    data = gene_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if not full_feat:
        # data = data[:, [0,1,2]]
        data = data[:, feature_list]

    kmeans = KMeans(n_clusters=n_families, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    gene_df['gene_coordinates'] = data.tolist()
    gene_df['gene_cluster'] = cluster_labels

    return gene_df, kmeans

def predictGeneClusters(gene_df, kmeans, full_feat=False):
    data = gene_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if not full_feat:
        # data = data[:, [0,1,2]]
        data = data[:, feature_list]
    cluster_labels = kmeans.predict(data)

    gene_df['gene_coordinates'] = data.tolist()
    gene_df['gene_cluster'] = cluster_labels

    return gene_df
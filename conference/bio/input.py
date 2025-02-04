import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# feature_list = [0,16,5]
feature_list = [0, 7, 11, 12, 16]

# Reads csv file and return preprocessed dataframe with only numerical columns
def preprocessMorphData():
    morph_df = pd.read_csv('data/morph.csv')
    morph_df = morph_df.drop(['idx', 'Date', 'classification', 'uncertainty','Latitude', 'Longitude', 'Altitude.ft', 'Multi.Single.stem', 'General.location.Habitat', 'site', 'Putative_spp'], axis=1)
    morph_df = morph_df.dropna()
    morph_df = morph_df.replace(0, 1e-10)
    return morph_df

# train clustering model; formalize data as coordinates; predict clusters for train set, only useful for the fully supervised portion
def fitMorphClusters(morph_df, n_families, full_feat=False):

    data = morph_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if not full_feat:
        # data = data[:, [0, 7, 11, 12, 16]]
        data = data[:, feature_list]

    kmeans = KMeans(n_clusters=n_families, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    morph_df['morph_coordinates'] = data.tolist()
    morph_df['morph_cluster'] = cluster_labels

    return morph_df, kmeans

# predict cluster assignment for test set, only useful for the bridged clustering method
def predictMorphClusters(morph_df, kmeans, full_feat=False):

    data = morph_df.drop(['TreeNo'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if not full_feat:
        # data = data[:, [0, 7, 11, 12, 16]]
        data = data[:, feature_list]

    cluster_labels = kmeans.predict(data)

    morph_df['morph_coordinates'] = data.tolist()
    morph_df['morph_cluster'] = cluster_labels

    return morph_df
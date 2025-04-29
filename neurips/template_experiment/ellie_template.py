# Bridged Clustering for WikiArt: Predicting Year from Paintings using Style as Latent Space

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from torchvision import models, transforms
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#############################
# Parameters
#############################
# TODO: ELLIE: 
# CHANGE TO 3 CLUSTERS 
# RUN WITH NEW BASELINES 
N_CLUSTERS = 2              # Latent T space (3: Abstract Exp, Realism, Cubism)
SUPERVISED_VALUES = [0.0015] #, 0.01, 0.05, 0.1]  # Percent of labeled (image, year) pairs
N_TRIALS = 3
IMAGE_SIZE = 224

# TODO ELLIE: pick first 50 (or 150) paintings that fit the genre description 
# beats knn on algorithm / mean teachers by a large margin -- ie a percentage on average 
# if doesn't work; tweak supervised_values, number of clusters (both go down) 
# if it doesn't work, manually make sure cluster selection is good (choose smallest and largest year) 
# try artists as latent if gneres doesn't work 

# TODO ELLIE: ASK FOR STYLE EXTRACTOR MODEL... PRODUCES ACCURATE CLASSIFICATION LOGITS 
# IF ORIGINAL THING DOESN'T WORK: take the logits (ie the last layer of the embedding) as the embedding for the painting 
# this INSTEAD of resnet-50 encoder 
# YAYAYAYAY 
#############################
# Image Preprocessing
#############################
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

def encode_images(df, image_folder):
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(image_folder, row['style'].replace(" ", "_"), row['filename'])
        # print("PATH: ", path)
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        try:
            image = Image.open(path).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                feat = resnet(image).squeeze().numpy()
            features.append(feat)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    return np.array(features)

#############################
# Clustering + Mapping
#############################
def cluster_features(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X), kmeans

def cluster_years(y, n_clusters):
    y = np.array(y).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(y), kmeans

def learn_bridge(x_clusters, y_clusters, supervised_indices, n_clusters):
    mapping = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        counts = np.zeros(n_clusters)
        for idx in supervised_indices:
            if x_clusters[idx] == i:
                counts[y_clusters[idx]] += 1
        mapping[i] = np.argmax(counts)
    return mapping

def compute_centroids(y_values, y_clusters, n_clusters):
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        centroids[i] = np.mean([y for y, c in zip(y_values, y_clusters) if c == i])
    return centroids

def predict_years(x_clusters, mapping, centroids):
    predicted_years = [centroids[mapping[c]] for c in x_clusters]
    return np.array(predicted_years)

#############################
# Baseline: KNN
#############################
def knn_baseline(train_X, train_y, test_X):
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(train_X, train_y)
    return knn.predict(test_X)

#############################
# Experiment Loop
#############################
def run_experiment(df, image_folder, supervised_fraction, trial_num=0, supervised_value=0.01):
    df = df.sample(frac=1).reset_index(drop=True)
    image_features = encode_images(df, image_folder)

    if image_features.shape[0] == 0:
        raise ValueError("No valid images were loaded. Check your paths and formats.")

    # Cluster X and Y
    x_clusters, _ = cluster_features(image_features, N_CLUSTERS)
    y_clusters, _ = cluster_years(df['year'][:len(image_features)], N_CLUSTERS)

    # Supervised set
   # Uniform per movement supervised sampling
    supervised_indices = []

    for style in df['style'].unique():
        style_indices = df.index[df['style'] == style].tolist()
        n_supervised_style = max(1, int(supervised_fraction * len(style_indices)))
        sampled = np.random.choice(style_indices, n_supervised_style, replace=False)
        supervised_indices.extend(sampled)

    supervised_indices = np.array(supervised_indices)

    # ✨ Add sanity print
    print(f"Sanity Check (Trial {trial_num+1}): Supervised samples per style:")
    for style in df['style'].unique():
        count = np.sum(df.iloc[supervised_indices]['style'] == style)
        print(f"  {style}: {count} supervised points")

    mapping = learn_bridge(x_clusters, y_clusters, supervised_indices, N_CLUSTERS)
    centroids = compute_centroids(df['year'][:len(image_features)].values, y_clusters, N_CLUSTERS)
    bridged_preds = predict_years(x_clusters, mapping, centroids)

    bridged_mae = mean_absolute_error(df['year'][:len(image_features)], bridged_preds)

    X_train = image_features[supervised_indices]
    y_train = df.iloc[supervised_indices]['year'].values
    X_test = image_features
    knn_preds = knn_baseline(X_train, y_train, X_test)
    knn_mae = mean_absolute_error(df['year'][:len(image_features)], knn_preds)

    #############################################
    # ✨ ADD VISUALIZATION BELOW ✨
    #############################################
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(image_features)

    plt.figure(figsize=(16, 4))

    # Plot 1: Color by true movement (style)
    plt.subplot(1, 3, 1)
    for style in df['style'].unique():
        idx = (df['style'][:len(image_features)] == style).values
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=style, alpha=0.7)
    plt.title(f'Trial {trial_num+1} — Movement')
    plt.legend()

    # Plot 2: Color by year
    plt.subplot(1, 3, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['year'][:len(image_features)], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Year')
    plt.title(f'Trial {trial_num+1} — Year')

    # Plot 3: Color by cluster assignment
    plt.subplot(1, 3, 3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=x_clusters, cmap='tab10', alpha=0.7)
    plt.title(f'Trial {trial_num+1} — X Clusters')

    plt.suptitle(f'Supervised {supervised_value:.0%} — Trial {trial_num+1}', fontsize=16)
    plt.tight_layout()
    plt.show()

    return bridged_mae, knn_mae

#############################
# Multi-Trial Evaluation
#############################
def run_all_trials(df, image_folder):
    results = {s: {'BKM': [], 'KNN': []} for s in SUPERVISED_VALUES}

    for s in SUPERVISED_VALUES:
        print(f"Supervised fraction: {s}")
        for trial in range(N_TRIALS):
            print(f"Trial {trial + 1}...")
            bkm_mae, knn_mae = run_experiment(df.copy(), image_folder, s, trial_num=trial, supervised_value=s)
            results[s]['BKM'].append(bkm_mae)
            results[s]['KNN'].append(knn_mae)
            bkm_mean = results[s]['BKM'][trial]
            knn_mean = results[s]['KNN'][trial]
            print(f"Supervised {s:.2%} — BKM MAE: {bkm_mean:.2f}, KNN MAE: {knn_mean:.2f}")


    return results

#############################
# Main
#############################
if __name__ == '__main__':
    metadata_csv = "filtered_styles.csv"
    image_folder = "wikiart"

    df = pd.read_csv(metadata_csv)
    df = df[df['style'].isin(['Early_Renaissance', 'Naive_Art_Primitivism'])]
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)


    df["full_path"] = df.apply(lambda row: os.path.join(image_folder, row["style"].replace(" ", "_"), row["filename"]), axis=1)
    df = df[df["full_path"].apply(os.path.exists)]
    df = df.drop(columns=["full_path"])

    results = run_all_trials(df, image_folder)

    for s in SUPERVISED_VALUES:
        bkm_mean = np.mean(results[s]['BKM'])
        knn_mean = np.mean(results[s]['KNN'])
        print(f"Supervised {s:.2%} — BKM MAE: {bkm_mean:.2f}, KNN MAE: {knn_mean:.2f}")

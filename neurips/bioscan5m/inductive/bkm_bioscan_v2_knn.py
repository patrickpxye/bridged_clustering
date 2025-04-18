import pandas as pd
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error
from scipy.stats import gaussian_kde
import torch.nn as nn
import torch.optim as optim

#########################################
# Data Loading and Splitting Functions  #
#########################################

def load_dataset(csv_path, image_folder, n_families=5, n_samples=50):
    """
    Load the dataset from a CSV file while prioritizing homogeneous sampling within families 
    and selecting families from distinct classes. If there are not enough families from distinct 
    classes, then families from different orders are selected.
    
    For each family, the function groups samples by 'species', then 'genus', then 'subfamily' (in that order).
    If a grouping produces a group with at least n_samples, that group is chosen.
    If no grouping produces a valid set for a family, that family is skipped and a different family is chosen.
    
    Parameters:
      csv_path (str): Path to the CSV file.
      image_folder (str): Directory containing the image files.
      n_families (int): Number of families to select.
      n_samples (int): Number of samples to select per family.
      
    Returns:
      final_df (DataFrame): The concatenated DataFrame containing the selected samples.
      images (dict): A mapping from processid to image path.
      
    Raises:
      ValueError: If fewer than n_families with a valid homogeneous group can be found.
    """
    df = pd.read_csv(csv_path)
    
    # Filter families that have at least n_samples overall.
    family_counts = df['family'].value_counts()
    eligible_families = family_counts[family_counts >= n_samples].index.tolist()
    if len(eligible_families) < n_families:
        raise ValueError(f"Not enough families with at least {n_samples} samples.")
    
    # Build a mapping from family to its class and order (assumes one unique class and order per family)
    family_info = df[['family', 'class', 'order']].drop_duplicates().set_index('family')
    
    # Group eligible families by "class"
    class_to_families = {}
    for fam in eligible_families:
        cls = family_info.loc[fam, 'class']
        class_to_families.setdefault(cls, []).append(fam)
    
    # Prioritize selecting one family per class.
    selected_families = []
    classes = list(class_to_families.keys())
    random.shuffle(classes)
    for cls in classes:
        fam_list = class_to_families[cls]
        random.shuffle(fam_list)
        selected_families.append(fam_list[0])
        if len(selected_families) == n_families:
            break

    # If not enough families from distinct classes, try selecting families from distinct orders.
    if len(selected_families) < n_families:
        order_to_families = {}
        for fam in eligible_families:
            order_val = family_info.loc[fam, 'order']
            order_to_families.setdefault(order_val, []).append(fam)
        orders = list(order_to_families.keys())
        random.shuffle(orders)
        for order_val in orders:
            candidates = [fam for fam in order_to_families[order_val] if fam not in selected_families]
            if candidates:
                random.shuffle(candidates)
                selected_families.append(candidates[0])
                if len(selected_families) == n_families:
                    break

    # If still not enough, fill the rest randomly from eligible families not yet chosen.
    if len(selected_families) < n_families:
        remaining = [fam for fam in eligible_families if fam not in selected_families]
        random.shuffle(remaining)
        selected_families.extend(remaining[: (n_families - len(selected_families))])
    
    # Now try to build valid homogeneous groups from the selected families.
    valid_family_samples = []
    failed_families = set()
    for family in selected_families:
        family_data = df[df['family'] == family]
        group_found = False
        for group_col in ['species', 'genus', 'subfamily']:
            groups = family_data.groupby(group_col)
            valid_groups = [(name, group_df) for name, group_df in groups if len(group_df) >= n_samples]
            if valid_groups:
                valid_groups.sort(key=lambda x: len(x[1]), reverse=True)
                chosen_group = valid_groups[0][1]
                group_found = True
                break
        if group_found:
            sample = chosen_group.sample(n=n_samples, random_state=42)
            valid_family_samples.append(sample)
        else:
            print(f"Family {family} does not have a homogeneous group with at least {n_samples} samples. Skipping.")
            failed_families.add(family)
        if len(valid_family_samples) == n_families:
            break

    # Try additional families if needed.
    if len(valid_family_samples) < n_families:
        remaining_families = [fam for fam in eligible_families if fam not in set(selected_families).union(failed_families)]
        random.shuffle(remaining_families)
        for family in remaining_families:
            family_data = df[df['family'] == family]
            group_found = False
            for group_col in ['species', 'genus', 'subfamily']:
                groups = family_data.groupby(group_col)
                valid_groups = [(name, group_df) for name, group_df in groups if len(group_df) >= n_samples]
                if valid_groups:
                    valid_groups.sort(key=lambda x: len(x[1]), reverse=True)
                    chosen_group = valid_groups[0][1]
                    group_found = True
                    break
            if group_found:
                sample = chosen_group.sample(n=n_samples, random_state=42)
                valid_family_samples.append(sample)
            if len(valid_family_samples) == n_families:
                break

    if len(valid_family_samples) < n_families:
        raise ValueError(f"Could not find {n_families} families with a valid homogeneous group of at least {n_samples} samples.")
    
    final_df = pd.concat(valid_family_samples)
    
    # Print the families that were eventually selected.
    selected_family_names = list(final_df['family'].unique())
    print("Selected families:", selected_family_names)
    
    # Build a dictionary mapping processid to image file path.
    images = {row['processid']: os.path.join(image_folder, f"{row['processid']}.jpg")
              for _, row in final_df.iterrows()}
    
    return final_df, images


def split_family_samples(family_data, supervised=0.05, random_state=42):
    """
    Randomly permute and split a family's data into four non-overlapping sets based on given proportions.
    
    Parameters:
      family_data (DataFrame): The data for one family.
      proportions (dict, optional): A dictionary with keys 'image', 'gene', 'supervised', and 'inference'
         specifying the split proportions. They must sum to 1. Default is 
         {'image': 0.3, 'gene': 0.3, 'supervised': 0.2, 'inference': 0.2}.
      random_state (int, optional): Random seed for shuffling.
      
    Returns:
      tuple: Four DataFrames corresponding to image_samples, gene_samples, supervised_samples, inference_samples.
    """
    # Set default proportions if none provided
    proportions = {'image': 0.4, 'gene': 0.4, 'supervised': supervised, 'inference': 1.0 - 0.8 - supervised}
    
    # Verify the proportions sum to 1
    total = sum(proportions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proportions must sum to 1. Provided sum: {total}")
    
    # Shuffle the data
    family_data = family_data.sample(frac=1, random_state=random_state)
    n = len(family_data)
    
    # Calculate the number of samples per split
    n_image = int(proportions['image'] * n)
    n_gene = int(proportions['gene'] * n)
    n_supervised = int(proportions['supervised'] * n)
    # Use remaining samples for inference to ensure full coverage
    n_inference = n - (n_image + n_gene + n_supervised)
    
    image_samples = family_data.iloc[:n_image]
    gene_samples = family_data.iloc[n_image:n_image + n_gene]
    supervised_samples = family_data.iloc[n_image + n_gene:n_image + n_gene + n_supervised]
    inference_samples = family_data.iloc[n_image + n_gene + n_supervised:]
    
    return image_samples, gene_samples, supervised_samples, inference_samples


def get_data_splits(df, supervised):
    """
    Loop over families in the DataFrame and concatenate splits from each family.
    Returns four DataFrames: image_samples, gene_samples, supervised_samples, inference_samples.
    """
    image_list, gene_list, sup_list, inf_list = [], [], [], []
    for family in df['family'].unique():
        family_data = df[df['family'] == family]
        img, gene, sup, inf = split_family_samples(family_data, supervised=supervised)
        image_list.append(img)
        gene_list.append(gene)
        sup_list.append(sup)
        inf_list.append(inf)
    return pd.concat(image_list), pd.concat(gene_list), pd.concat(sup_list), pd.concat(inf_list)

#############################
# Model and Preprocessing   #
#############################
def load_pretrained_models():

    """
    Load and return pre-trained models and associated preprocessors.
    """
    # Load BarcodeBERT for genetic barcode encoding
    barcode_model_name = "bioscan-ml/BarcodeBERT"
    barcode_tokenizer = AutoTokenizer.from_pretrained(barcode_model_name, trust_remote_code=True)
    barcode_model = AutoModel.from_pretrained(barcode_model_name, trust_remote_code=True)

    # Load ResNet50 model for image encoding
    image_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    image_model.eval()

    # Define image preprocessing (resizing, cropping, normalization for ResNet50)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return barcode_tokenizer, barcode_model, image_model, image_transform

##################################
# Encoding Functions             #
##################################
def encode_images(image_ids, image_folder, model, transform):
    """
    Encode images using the provided model and transform.
    Returns a NumPy array of image features.
    """
    features = []
    for processid in image_ids:
        image_path = image_folder.get(processid, None)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(image)
            features.append(output.squeeze().numpy())
        else:
            print(f"Warning: Image {processid} not found or invalid!")
            features.append(np.zeros(model.fc.in_features))  # Placeholder if image missing
    return np.array(features)

def encode_genes(dna_barcodes, tokenizer, model):
    """
    Encode DNA barcode sequences using the tokenizer and model.
    Returns a NumPy array of gene features.
    """
    if isinstance(dna_barcodes, np.ndarray):
        dna_barcodes = [str(barcode) for barcode in dna_barcodes]
    
    embeddings = []
    for barcode in dna_barcodes:
        encodings = tokenizer(barcode, return_tensors="pt", padding=True, truncation=True)
        # Add batch dimension
        encodings = {key: value.unsqueeze(0) for key, value in encodings.items()}
        with torch.no_grad():
            embedding = model(**encodings).last_hidden_state.mean(dim=1).numpy()
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    if len(embeddings.shape) == 3:
        embeddings = embeddings.squeeze(1)
    return embeddings

def encode_images_for_samples(df, image_folder, image_model, image_transform):
    features = encode_images(df['processid'].values, image_folder, image_model, image_transform)
    df['morph_coordinates'] = features.tolist()
    return df

def encode_genes_for_samples(df, barcode_tokenizer, barcode_model):
    gene_features = encode_genes(df['dna_barcode'].values, barcode_tokenizer, barcode_model)
    df['gene_coordinates'] = gene_features.tolist()
    return df

##################################
# Clustering and Decision Making #
##################################
def perform_clustering(image_samples, gene_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families):
    """
    Perform KMeans clustering on image and gene samples.
    Returns the trained KMeans objects and raw features.
    """
    image_features = encode_images(image_samples['processid'].values, images, image_model, image_transform)
    image_kmeans = KMeans(n_clusters=n_families, random_state=42).fit(image_features)
    image_clusters = image_kmeans.predict(image_features)
    
    gene_features = encode_genes(gene_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    gene_kmeans = KMeans(n_clusters=n_families, random_state=42).fit(gene_features)
    gene_clusters = gene_kmeans.predict(gene_features)
    
    return image_kmeans, gene_kmeans, image_features, gene_features, image_clusters, gene_clusters

def decisionVector(sample, morph_column='morph_cluster', gene_column='gene_cluster', dim=5):

    # Check if the specified columns exist in the DataFrame
    if morph_column not in sample.columns:
        raise KeyError(f"Column '{morph_column}' not found in the DataFrame.")
    if gene_column not in sample.columns:
        raise KeyError(f"Column '{gene_column}' not found in the DataFrame.")

    # Create association matrix
    association_matrix = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            association_matrix[i, j] = np.sum((sample[morph_column] == i) & (sample[gene_column] == j))
    
    # Initialize decision array (this could be improved based on specific logic for decision making)
    decision = np.zeros(dim, dtype=int)
    
    # Logic to compute the decision vector based on association_matrix (you can modify this logic)
    # For now, just assigning maximum values
    for i in range(dim):
        decision[i] = np.argmax(association_matrix[i, :])  # You can customize this

    return decision

def build_decision_matrix(supervised_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, image_kmeans, gene_kmeans, n_families):
    """
    Build the decision matrix (association vector) using the supervised samples.
    """
    supervised_samples = supervised_samples.copy()
    sup_image_features = encode_images(supervised_samples['processid'].values, images, image_model, image_transform)
    supervised_samples['image_cluster'] = image_kmeans.predict(sup_image_features)
    
    sup_gene_features = encode_genes(supervised_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    supervised_samples['gene_cluster'] = gene_kmeans.predict(sup_gene_features)
    
    decision_matrix = decisionVector(supervised_samples, morph_column='image_cluster', gene_column='gene_cluster', dim=n_families)
    return decision_matrix

def compute_gene_centroids(gene_samples, gene_features, gene_kmeans, n_families):
    """
    Compute centroids for gene clusters based on gene_samples.
    """
    gene_samples = gene_samples.copy()
    gene_samples['gene_cluster'] = gene_kmeans.labels_
    gene_samples['gene_coordinates'] = gene_features.tolist()
    
    centroids = []
    for cluster in range(n_families):
        cluster_data = gene_samples[gene_samples['gene_cluster'] == cluster]
        if len(cluster_data) > 0:
            centroid = np.mean(np.stack(cluster_data['gene_coordinates'].values), axis=0)
        else:
            centroid = np.zeros(gene_features.shape[1])
        centroids.append(centroid)
    return np.array(centroids)

def perform_inference(inference_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, image_kmeans, decision_matrix, centroids):
    """
    Assign clusters to inference samples and predict gene coordinates.
    """
    inference_samples = inference_samples.copy()
    inf_image_features = encode_images(inference_samples['processid'].values, images, image_model, image_transform)
    inference_samples['image_cluster'] = image_kmeans.predict(inf_image_features)
    
    inference_samples['predicted_gene_cluster'] = inference_samples['image_cluster'].apply(lambda x: decision_matrix[x])
    inference_samples['predicted_gene_coordinates'] = inference_samples['predicted_gene_cluster'].apply(lambda x: centroids[x])
    
    inf_gene_features = encode_genes(inference_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    inference_samples['gene_coordinates'] = inf_gene_features.tolist()
    return inference_samples

def compute_error_on_inference(df):
    """
    Compute the error between predicted and actual gene coordinates.
    """
    predicted_gene_coords = np.array(df['predicted_gene_coordinates'].tolist())
    actual_gene_coords = np.array(df['gene_coordinates'].tolist())

    # Ensure that both predicted and actual have the same shape
    if predicted_gene_coords.shape != actual_gene_coords.shape:
        print(f"Shape mismatch detected: Predicted shape = {predicted_gene_coords.shape}, Actual shape = {actual_gene_coords.shape}")
        raise ValueError("Shape mismatch between predicted and actual coordinates.")

    # Calculate Euclidean distance for each sample (row-wise distance)
    distances = np.linalg.norm(predicted_gene_coords - actual_gene_coords, axis=1)  # Euclidean distance for each sample

    # Return the mean and standard deviation of the distances
    return distances
    

##################################
# KNN Regression       #
##################################
def knn_regression(supervised_df, test_df, n_neighbors=1, image_model=None, image_transform=None,
                   barcode_tokenizer=None, barcode_model=None, image_folder=None):
    """
    Train a KNN regressor using the supervised samples and evaluate on the test samples.
    Returns the Euclidean distances for each test sample.
    """
    supervised_df = encode_images_for_samples(supervised_df, image_folder, image_model, image_transform)
    supervised_df = encode_genes_for_samples(supervised_df, barcode_tokenizer, barcode_model)
    
    test_df = encode_images_for_samples(test_df, image_folder, image_model, image_transform)
    test_df = encode_genes_for_samples(test_df, barcode_tokenizer, barcode_model)
    
    X_train = np.array(supervised_df['morph_coordinates'].tolist())
    y_train = np.array(supervised_df['gene_coordinates'].tolist())
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    X_test = np.array(test_df['morph_coordinates'].tolist())
    predictions = knn.predict(X_test)
    y_test = np.array(test_df['gene_coordinates'].tolist())
    distances = np.linalg.norm(predictions - y_test, axis=1)
    return distances



###########################
# Plots and Visualization   #
###########################

def plot_pca_clustering(features, cluster_labels, group_labels=None, title="PCA Clustering", save_path=None):

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    
    if group_labels is None:
        sc = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                         c=cluster_labels, cmap='viridis', s=50)
        plt.colorbar(sc, label="Cluster")
    else:
        # Define a list of marker styles.
        markers = ['o', 's', '^', 'P', 'D', 'X', 'v', '<', '>', '*', '+']
        unique_groups = np.unique(group_labels)
        # For color mapping, we use the cluster labels.
        # Plot each group separately with its own marker style.
        scatter_handles = []
        for i, group in enumerate(unique_groups):
            idx = np.where(np.array(group_labels) == group)[0]
            sc = plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1],
                             marker=markers[i % len(markers)],
                             c=np.array(cluster_labels)[idx],
                             cmap='viridis',
                             s=50,
                             label=str(group))
            # Store the handle for legend (markers will indicate family)
            scatter_handles.append(sc)
        # Add colorbar from the last scatter handle (they all share the same colormap)
        plt.colorbar(sc, label="Cluster")
        plt.legend(title="Family")
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    #measure the NMI of the clustering
    nmi = normalized_mutual_info_score(cluster_labels, group_labels)
    print(f"Normalized Mutual Information (NMI) of the clustering: {nmi}")
    #text at bottom right corner
    plt.text(0.99, 0.01, f"NMI: {nmi:.3f}", ha='right', va='bottom', transform=plt.gca().transAxes)
    if save_path:
        plt.savefig(save_path)
    #clear the plot!!
    plt.clf()


def plotDensityGraph(result_bkm, result_knn, save_path=None):

    result_bkm = np.ravel(result_bkm)  # Ensure 1D array
    result_knn = np.ravel(result_knn)  # Ensure 1D array

    kde = gaussian_kde(result_bkm)
    x = np.linspace(min(result_bkm), max(result_bkm), 2000)
    density = kde(x)
    plt.plot(x, density, label='Bridged Clustering')
    plt.fill_between(x, density, alpha=0.5)
    mean_bkm = np.mean(result_bkm)
    print("mean distance for bkm is ", mean_bkm)
    plt.axvline(mean_bkm, color='blue', linestyle='dashed', linewidth=1)

    kde = gaussian_kde(result_knn)
    x = np.linspace(min(result_knn), max(result_knn), 2000)
    density = kde(x)
    plt.plot(x, density, label='KNN Regression')
    plt.fill_between(x, density, alpha=0.5)
    mean_knn = np.mean(result_knn)
    print("mean distance for KNN Regression is ", mean_knn)
    plt.axvline(mean_knn, color='red', linestyle='dashed', linewidth=1)

    plt.xlabel('Euclidean Distance', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


###########################
# Main Experiment Routine #
###########################
def run_experiment(csv_path, image_folder, n_families, n_samples=50, supervised=0.05):
    
    # Load the full dataset and images
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    
    # Split the dataset into non-overlapping sets per family
    image_samples, gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised)
    
    # Load pre-trained models and preprocessors
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()
    
    # Perform clustering on image and gene samples
    image_kmeans, gene_kmeans, image_features, gene_features, image_clusters, gene_clusters = perform_clustering(
        image_samples, gene_samples, images, image_model, image_transform,
        barcode_tokenizer, barcode_model, n_families
    )
    # Generate PCA plots for both image and gene clustering
    plot_pca_clustering(image_features, image_clusters,
                        group_labels=image_samples['family'].values,
                        title="PCA of Image Clusters", save_path="pca_images.png")
    # For the gene PCA, we pass the family column from gene_samples.
    plot_pca_clustering(gene_features, gene_clusters,
                        group_labels=gene_samples['family'].values,
                        title="PCA of Gene Clusters", save_path="pca_genes.png")
    
    
    # Build the decision matrix using the supervised split only
    decision_matrix = build_decision_matrix(supervised_samples, images, image_model, image_transform,
                                              barcode_tokenizer, barcode_model, image_kmeans, gene_kmeans, n_families)
    
    # Compute centroids for gene clusters using the gene samples
    centroids = compute_gene_centroids(gene_samples, gene_features, gene_kmeans, n_families)
    
    # Perform inference on the inference samples only
    inference_samples = perform_inference(inference_samples, images, image_model, image_transform,
                                          barcode_tokenizer, barcode_model, image_kmeans, decision_matrix, centroids)
    
    # Compute Bridged Clustering error on inference samples
    bkm_error = compute_error_on_inference(inference_samples)
    
    # Baseline KNN regression using supervised and inference splits
    knn_error = knn_regression(supervised_samples, inference_samples, n_neighbors=int(n_samples*supervised),
                                      image_model=image_model, image_transform=image_transform,
                                      barcode_tokenizer=barcode_tokenizer, barcode_model=barcode_model,
                                      image_folder=images)
    
    
    print(f"Bridged Clustering Error: Mean={np.mean(bkm_error)}, Std={np.std(bkm_error)}")
    print(f"KNN Error: Mean={np.mean(knn_error)}, Std={np.std(knn_error)}")

    return bkm_error, knn_error






if __name__ == '__main__':
    csv_path = '../test_data.csv'
    image_folder = '../test_images'
    bkm_total, knn_total = [], []
    n_families = 5
    n_samples= 100
    supervised= 0.03
    for i in range(1):
        print(f"Experiment {i+1}")
        bkm, knn = run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=supervised)
        bkm_total.append(bkm)
        knn_total.append(knn)
    print("Bridged Clustering Errors Average:", np.mean(np.array(bkm_total)))
    print("KNN Regression Errors Average:", np.mean(np.array(knn_total)))
    plotDensityGraph(np.array(bkm_total), np.array(knn_total), save_path=f'plot_{n_families}_{supervised}_{n_samples}.pdf')
    print("Experiment Completed!")
    print(f"With {n_families} families, {n_samples} samples per family, and {supervised} percent supervised samples per family:")
    print("BKM to KNN Ratio:", np.mean(np.array(bkm_total)) / np.mean(np.array(knn_total)))
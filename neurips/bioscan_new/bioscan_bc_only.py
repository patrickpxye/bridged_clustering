import pandas as pd
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error, mean_absolute_error, adjusted_mutual_info_score
from scipy.stats import gaussian_kde
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.optim import Adam
import torch_geometric
import torch_geometric.utils
from sklearn.metrics import r2_score
from k_means_constrained import KMeansConstrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    df = df[df['class'] == 'Insecta']
    
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
            sample = chosen_group.sample(n=n_samples)
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
                sample = chosen_group.sample(n=n_samples)
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

def split_family_samples(family_data, supervised=0.05, out_only=0.5):
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
    proportions = {'gene': out_only, 'supervised': supervised, 'inference': 1.0 - out_only - supervised}
    
    # Verify the proportions sum to 1
    total = sum(proportions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proportions must sum to 1. Provided sum: {total}")
    
    # Shuffle the data
    family_data = family_data.sample(frac=1)
    n = len(family_data)
    
    # Calculate the number of samples per split
    n_gene = int(proportions['gene'] * n)
    n_supervised = int(proportions['supervised'] * n)
    n_supervised = max(n_supervised, 1)

    # Use remaining samples for inference to ensure full coverage
    n_inference = n - (n_gene + n_supervised)
    
    gene_samples = family_data.iloc[: n_gene]
    supervised_samples = family_data.iloc[n_gene:n_gene + n_supervised]
    inference_samples = family_data.iloc[n_gene + n_supervised:]
    
    return gene_samples, supervised_samples, inference_samples

def get_data_splits(df, supervised, out_only):
    """
    Loop over families in the DataFrame and concatenate splits from each family.
    Returns four DataFrames: image_samples, gene_samples, supervised_samples, inference_samples.
    """
    image_list, gene_list, sup_list, inf_list = [], [], [], []
    for family in df['family'].unique():
        family_data = df[df['family'] == family]
        gene, sup, inf = split_family_samples(family_data, supervised=supervised, out_only=out_only)
        gene_list.append(gene)
        sup_list.append(sup)
        inf_list.append(inf)
    return pd.concat(gene_list), pd.concat(sup_list), pd.concat(inf_list)

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
    barcode_model = AutoModel.from_pretrained(barcode_model_name, trust_remote_code=True).to(device)

    # Load ResNet50 model for image encoding
    image_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
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
            image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                output = model(image)
            features.append(output.squeeze().cpu().numpy())
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
        encodings = {key: value.unsqueeze(0).to(device) for key, value in encodings.items()}
        with torch.no_grad():
            embedding = model(**encodings).last_hidden_state.mean(dim=1).cpu().numpy()
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
# Bridged Clustering Functions #
##################################

def perform_clustering(image_samples, gene_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families):
    """
    Perform KMeans clustering on image and gene samples.
    Returns the trained KMeans objects and raw features.
    """

    N = len(image_samples)
    base_size = N // n_families
    size_min = base_size
    size_max = base_size + (1 if N % n_families else 0)

    M = len(gene_samples)
    base_size_g = M // n_families
    size_min_g = base_size_g
    size_max_g = base_size_g + (1 if M % n_families else 0)

    image_features = encode_images(image_samples['processid'].values, images, image_model, image_transform)
    image_kmeans = KMeansConstrained(n_clusters=n_families, size_min=size_min, size_max=size_max, random_state=42).fit(image_features)
    image_clusters = image_kmeans.predict(image_features)

    
    gene_features = encode_genes(gene_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    gene_kmeans = KMeansConstrained(n_clusters=n_families, size_min=size_min_g, size_max=size_max_g, random_state=42).fit(gene_features)
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


def build_true_decision_vector(img_df: pd.DataFrame,
                               gene_df: pd.DataFrame,
                               dim: int):
    """
    img_df : DataFrame with columns ['image_cluster','family']
    gene_df: DataFrame with columns ['gene_cluster','family']
    dim    : number of clusters (n_families)

    Returns:
      decision: np.array of length dim, where
                decision[i] = gene_cluster whose majority-family
                               matches image_cluster i’s majority-family
                (or -1 if no match)
      image_to_family: dict {image_cluster -> majority family}
      gene_to_family:  dict {gene_cluster  -> majority family}
    """
    # 1) majority-family for each image-cluster
    image_to_family = (
        img_df
        .groupby('image_cluster')['family']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    print(f"Image to family mapping: {image_to_family}")

    # 2) majority-family for each gene-cluster
    gene_to_family = (
        gene_df
        .groupby('gene_cluster')['family']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    print(f"Gene to family mapping: {gene_to_family}")

    # 3) invert 1–1 mapping directly
    family_to_gene = {fam: gc for gc, fam in gene_to_family.items()}

    # 4) build decision vector with default -1
    decision = np.full(dim, -1, dtype=int)
    for i in range(dim):
        fam = image_to_family.get(i)           # might be None if i missing
        decision[i] = family_to_gene.get(fam, -1)

    return decision


def build_decision_matrix(supervised_samples, image_clusters, gene_clusters, n_families):
    """
    Build the decision matrix (association vector) using the supervised samples.
    """
    N_sup = len(supervised_samples)
    supervised_samples = supervised_samples.copy()
    supervised_samples['image_cluster'] = image_clusters[-N_sup:]
    supervised_samples['gene_cluster'] = gene_clusters[-N_sup:]
    
    decision_matrix = decisionVector(supervised_samples, morph_column='image_cluster', gene_column='gene_cluster', dim=n_families)

    return decision_matrix



def compute_gene_centroids(gene_samples, gene_features, gene_clusters, n_families):
    """
    Compute centroids for gene clusters based on gene_samples.
    """
    gene_samples = gene_samples.copy()
    N_gene = len(gene_samples)
    gene_samples['gene_cluster'] = gene_clusters[:N_gene]
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

def perform_inference(inference_samples, image_clusters, barcode_tokenizer, barcode_model, image_kmeans, decision_matrix, centroids):
    """
    Assign clusters to inference samples and predict gene coordinates.
    """
    N_inf = len(inference_samples)
    inference_samples = inference_samples.copy()
    inference_samples['image_cluster'] = image_clusters[:N_inf]
    
    inference_samples['predicted_gene_cluster'] = inference_samples['image_cluster'].apply(lambda x: decision_matrix[x])
    inference_samples['predicted_gene_coordinates'] = inference_samples['predicted_gene_cluster'].apply(lambda x: centroids[x])
    
    inf_gene_features = encode_genes(inference_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    inference_samples['gene_coordinates'] = inf_gene_features.tolist()
    return inference_samples

def bkm_regression(df):
    """
    Compute the error between predicted and actual gene coordinates.
    """
    predicted_gene_coords = np.array(df['predicted_gene_coordinates'].tolist())
    actual_gene_coords = np.array(df['gene_coordinates'].tolist())

    return predicted_gene_coords, actual_gene_coords


###################################
# Different Models #
###################################

# Model B: KNN Regression
#################################

def knn_regression(supervised_df, test_df, n_neighbors=1):
    """
    Train a KNN regressor using the supervised samples and evaluate on the test samples.
    Returns the Euclidean distances for each test sample.
    """
    
    X_train = np.array(supervised_df['morph_coordinates'].tolist())
    y_train = np.array(supervised_df['gene_coordinates'].tolist())
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    X_test = np.array(test_df['morph_coordinates'].tolist())
    predictions = knn.predict(X_test)
    y_test = np.array(test_df['gene_coordinates'].tolist())

    return predictions, y_test


###################################
# Metrics #
###################################

def evaluate_loss(predictions, actuals):
    """
    Evaluate the loss between predictions and actual values.
    For now, just return MAE.
    """
    # return np.mean(np.abs(predictions - actuals))

    # Mean Absolute Error
    
    mae = mean_absolute_error(predictions, actuals)
    
    # Mean Squared Error
    mse = mean_squared_error(predictions, actuals)
    
    return mae, mse

    # Another Option: Calculate Euclidean distance for each sample (row-wise distance)
    # distances = np.linalg.norm(predictions - actuals, axis=1)  # Euclidean distance for each sample
    # mae = np.mean(np.abs(predictions - actuals))


###########################
# Main Experiment Routine #
###########################
def run_experiment(csv_path, image_folder, n_families, n_samples=50, supervised=0.05, out_only=0.5):

    # Load dataset and images
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised, out_only=out_only)

    # Load pre-trained models
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    supervised_samples = encode_genes_for_samples(supervised_samples, barcode_tokenizer, barcode_model)
    gene_samples = encode_genes_for_samples(gene_samples, barcode_tokenizer, barcode_model)
    gene_plus_supervised = pd.concat([gene_samples, supervised_samples], axis=0)

    supervised_samples = encode_images_for_samples(supervised_samples, images, image_model, image_transform)
    inference_samples = encode_genes_for_samples(inference_samples, barcode_tokenizer, barcode_model)
    inference_samples = encode_images_for_samples(inference_samples, images, image_model, image_transform)
    inference_plus_supervised = pd.concat([inference_samples, supervised_samples], axis=0)

    # Perform Bridged Clustering
    image_kmeans, gene_kmeans, _, gene_features, image_clusters, gene_clusters = perform_clustering(
        inference_plus_supervised, gene_plus_supervised, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families
    )

    ami_image = adjusted_mutual_info_score(image_clusters, inference_plus_supervised['family'].values)
    ami_gene = adjusted_mutual_info_score(gene_clusters, gene_plus_supervised['family'].values)
    print(f"Adjusted Mutual Information (Image): {ami_image}")
    print(f"Adjusted Mutual Information (Gene): {ami_gene}")
    inference_plus_supervised['image_cluster'] = image_clusters
    gene_plus_supervised['gene_cluster'] = gene_clusters

    true_decision_vector = build_true_decision_vector(inference_plus_supervised, gene_plus_supervised, n_families)
    decision_matrix = build_decision_matrix(supervised_samples, image_clusters, gene_clusters, n_families)
    decision_accuracy = np.mean(true_decision_vector == decision_matrix)
    print(f"Decision: {decision_matrix}")
    print(f"Oracle Decision: {true_decision_vector}")
    print(f"Decision Accuracy: {decision_accuracy}")


    # Compute centroids for gene clusters using the gene samples
    centroids = compute_gene_centroids(gene_plus_supervised, gene_features, gene_clusters, n_families)

    # Perform inference using Bridged Clustering
    inference_samples_bc = perform_inference(inference_samples, image_clusters, barcode_tokenizer, barcode_model, image_kmeans, decision_matrix, centroids)


    bkm_predictions, bkm_actuals = bkm_regression(inference_samples_bc)

    ### Unlike BKM, we don't call a series of functions for the basline models, instead we put them in the helper "regression" functions
    knn_predictions, knn_actuals = knn_regression(supervised_samples, inference_samples, n_neighbors=max(1, int(n_samples * supervised)))

    # Compute errors
    bkm_error, bkm_r2 = evaluate_loss(bkm_predictions, bkm_actuals)
    knn_error, knn_r2 = evaluate_loss(knn_predictions, knn_actuals)

    # Print results
    print(f"Bridged Clustering Error: {bkm_error}")
    print(f"KNN Error: {knn_error}")
    # Store results in a dictionary

    errors = {
        'BKM': bkm_error,
        'KNN': knn_error,
    }

    rs = {
        'BKM': bkm_r2,
        'KNN': knn_r2,
    }
    

    return errors, rs, ami_image, ami_gene, decision_accuracy



if __name__ == '__main__':
    csv_path = 'test_data.csv'
    image_folder = 'test_images'

    experiment_key = "036"

    n_families_values = [3,5,7]
    n_samples_values = [100]
    supervised_values = [0.01, 0.02, 0.04, 0.08]
    out_only_values = [0.2]
    models = ['BKM', 'KNN']

    n_trials = 20
    

    # Initialize a 5D matrix to store results for each experiment
    # Dimensions: [n_families, n_samples, supervised, models, trials]
    results_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(out_only_values), len(models), n_trials))
    rs_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(out_only_values), len(models), n_trials))
    ami_image_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(out_only_values), n_trials))
    ami_gene_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(out_only_values), n_trials))
    decesion_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(out_only_values), n_trials))

    # Initialize a dictionary to store average results for each experiment setting
    average_results = {}
    average_rs_results = {}

    # Run experiments
    for n_families_idx, n_families in enumerate(n_families_values):
        for n_samples_idx, n_samples in enumerate(n_samples_values):
            for supervised_idx, supervised in enumerate(supervised_values):
                for out_only_idx, out_only in enumerate(out_only_values):
                    # Initialize a dictionary to store cumulative errors for each model
                    cumulative_errors = {model: 0 for model in models}
                    cumulative_rs = {model: 0 for model in models}
                    
                    for trial in range(n_trials):
                        print(f"Running trial {trial + 1} for n_families={n_families}, n_samples={n_samples}, supervised={supervised}, out_only={out_only}")
                        errors,rs,ami_image,ami_gene,decision_accuracy = \
                            run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=supervised, out_only=out_only)

                        ami_image_matrix[n_families_idx, n_samples_idx, supervised_idx, out_only_idx, trial] = ami_image
                        ami_gene_matrix[n_families_idx, n_samples_idx, supervised_idx, out_only_idx, trial] = ami_gene
                        decesion_matrix[n_families_idx, n_samples_idx, supervised_idx, out_only_idx, trial] = decision_accuracy
                        
                        # Accumulate errors for each model
                        for model_name in models:
                            cumulative_errors[model_name] += errors[model_name]
                            cumulative_rs[model_name] += rs[model_name]
                            
                        # Store results in the matrix
                        for model_idx, model_name in enumerate(models):
                            results_matrix[n_families_idx, n_samples_idx, supervised_idx, out_only_idx, model_idx, trial] = errors[model_name]
                            rs_matrix[n_families_idx, n_samples_idx, supervised_idx, out_only_idx, model_idx, trial] = rs[model_name]

                        # Save the results matrix to a file
                        np.save(f"results/mae_matrix_{experiment_key}.npy", results_matrix)
                        np.save(f"results/mse_matrix_{experiment_key}.npy", rs_matrix)
                        np.save(f"results/ami_image_matrix_{experiment_key}.npy", ami_image_matrix)
                        np.save(f"results/ami_gene_matrix_{experiment_key}.npy", ami_gene_matrix)
                        np.save(f"results/decision_matrix_{experiment_key}.npy", decesion_matrix)
                    

    # Save the results matrix to a file
    np.save(f"results/mae_matrix_{experiment_key}.npy", results_matrix)
    np.save(f"results/mse_matrix_{experiment_key}.npy", rs_matrix)
    np.save(f"results/ami_image_matrix_{experiment_key}.npy", ami_image_matrix)
    np.save(f"results/ami_gene_matrix_{experiment_key}.npy", ami_gene_matrix)
    np.save(f"results/decision_matrix_{experiment_key}.npy", decesion_matrix)
    print("Experiment completed.")
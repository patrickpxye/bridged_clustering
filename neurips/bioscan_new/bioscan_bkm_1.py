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
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error
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
import warnings
warnings.filterwarnings("ignore")


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
    proportions = {'gene': 0.4, 'supervised': supervised, 'inference': 1.0 - 0.4 - supervised}
    
    # Verify the proportions sum to 1
    total = sum(proportions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proportions must sum to 1. Provided sum: {total}")
    
    # Shuffle the data
    family_data = family_data.sample(frac=1, random_state=random_state)
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

def get_data_splits(df, supervised):
    """
    Loop over families in the DataFrame and concatenate splits from each family.
    Returns four DataFrames: image_samples, gene_samples, supervised_samples, inference_samples.
    """
    image_list, gene_list, sup_list, inf_list = [], [], [], []
    for family in df['family'].unique():
        family_data = df[df['family'] == family]
        gene, sup, inf = split_family_samples(family_data, supervised=supervised)
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
    barcode_model = AutoModel.from_pretrained(barcode_model_name, trust_remote_code=True)

    # Load ResNet50 model for image encoding
    image_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
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
# Bridged Clustering Functions #
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

################################################################################
# Batch A · Classical supervised regressors (OLS → MLP)                        #
################################################################################
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor


def _split_arrays(sup_df, inf_df):
    """utility: convert morph_coordinates / gene_coordinates to numpy arrays"""
    Xtr = np.vstack(sup_df['morph_coordinates'])
    ytr = np.vstack(sup_df['gene_coordinates'])
    Xte = np.vstack(inf_df['morph_coordinates'])
    yte = np.vstack(inf_df['gene_coordinates'])
    return Xtr, ytr, Xte, yte


# 1) Ordinary Least Squares -----------------------------------------------------
def ols_regression(sup_df, inf_df):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    model = LinearRegression(n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 2) Ridge Regression -----------------------------------------------------------
def ridge_regression(sup_df, inf_df, alpha=1.0):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 3) Lasso Regression -----------------------------------------------------------
def lasso_regression(sup_df, inf_df, alpha=1e-3):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    model = MultiOutputRegressor(Lasso(alpha=alpha, max_iter=5000, random_state=42))
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 4) Elastic‑Net ---------------------------------------------------------------
def elasticnet_regression(sup_df, inf_df, alpha=1e-2, l1_ratio=0.5):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    base = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 5) Partial Least Squares (PLS) -----------------------------------------------
def pls_regression(sup_df, inf_df, n_components=10):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    n_components = min(n_components, min(Xtr.shape[1], ytr.shape[1]))
    model = PLSRegression(n_components=n_components)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 6) Support‑Vector Regression (multi‑output) -----------------------------------
def svr_regression(sup_df, inf_df, C=10.0, epsilon=0.1, kernel='rbf'):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    base = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma='scale')
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 7) Decision‑Tree Regressor ----------------------------------------------------
def dt_regression(sup_df, inf_df, max_depth=None):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    model = MultiOutputRegressor(
        DecisionTreeRegressor(max_depth=max_depth, random_state=42))
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 8) Random‑Forest Regressor ----------------------------------------------------
def rf_regression(sup_df, inf_df, n_estimators=200, max_depth=None):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    base = RandomForestRegressor(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 n_jobs=-1,
                                 random_state=42)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# 9) Gaussian‑Process Regression ------------------------------------------------
def gpr_regression(sup_df, inf_df, alpha=1e-6):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    # wrap independent GPR per output dimension
    gprs = [GaussianProcessRegressor(alpha=alpha, random_state=42).fit(Xtr, ytr[:, i])
            for i in range(ytr.shape[1])]
    preds = np.column_stack([gpr.predict(Xte) for gpr in gprs])
    return preds, yte


# 10) Multi‑Layer Perceptron (MLP) ---------------------------------------------
def mlp_regression(sup_df, inf_df, hidden=(256,128), lr=1e-3, max_iter=500):
    Xtr, ytr, Xte, yte = _split_arrays(sup_df, inf_df)
    model = MLPRegressor(hidden_layer_sizes=hidden,
                         activation='relu',
                         solver='adam',
                         learning_rate_init=lr,
                         max_iter=max_iter,
                         random_state=42)
    model = MultiOutputRegressor(model, n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


################################################################################
# Batch B · Classical graph‑based semi‑supervised regressors                    #
################################################################################
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics.pairwise import rbf_kernel

# -----------------------------------------------------------------------------#
# shared utilities                                                             #
# -----------------------------------------------------------------------------#
def _split_arrays(sup_df, inf_df):
    Xl = np.vstack(sup_df['morph_coordinates'])
    yl = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    yu = np.vstack(inf_df['gene_coordinates'])
    return Xl, yl, Xu, yu


def _affinity_matrix(X, k=10, gamma=None):
    """k‑NN symmetrised RBF affinity."""
    if gamma is None:
        # heuristic: γ = 1 / median²
        dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        gamma = 1.0 / (np.median(dists) ** 2 + 1e-9)

    W_dense = rbf_kernel(X, X, gamma=gamma)
    # keep only k nearest neighbours per row
    idx = np.argsort(-W_dense, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = idx.ravel()
    data = W_dense[rows, cols]
    W = sp.coo_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    W = (W + W.T).maximum(W.T)          # symmetrise
    return W.tocsr(), gamma


def _harmonic_solution(W, y_l, idx_l):
    """Solve the classical harmonic function regression (Gaussian fields / LGC)."""
    n = W.shape[0]
    D = sp.diags(W.sum(1).A1)
    L = D - W

    idx_u = np.setdiff1d(np.arange(n), idx_l)
    L_uu = L[idx_u][:, idx_u]
    L_ul = L[idx_u][:, idx_l]

    # Solve  −L_uu f_u = L_ul y_l  ⇒  f_u = −L_uu⁻¹ L_ul y_l
    f_u = sp.linalg.spsolve(-L_uu, L_ul @ y_l)
    return f_u, idx_u


# -----------------------------------------------------------------------------#
# 11) Label‑Propagation / Gaussian‑Field regression                            #
# -----------------------------------------------------------------------------#
def label_propagation_regression(sup_df, inf_df, k=10):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    X = np.vstack([Xl, Xu])
    W, _ = _affinity_matrix(X, k=k)

    idx_l = np.arange(Xl.shape[0])
    f_u, idx_u = _harmonic_solution(W, yl, idx_l)

    # assemble predictions (only need those for unlabeled set)
    preds = f_u
    return preds, yu


# -----------------------------------------------------------------------------#
# 12) Local‑Global Consistency (LGC)                                           #
#     identical to harmonic but with α‑scaling; α≈0.99 in the original paper  #
# -----------------------------------------------------------------------------#
def lgc_regression(sup_df, inf_df, alpha=0.99, k=10):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    X = np.vstack([Xl, Xu])
    W, _ = _affinity_matrix(X, k=k)
    D = sp.diags(W.sum(1).A1)
    D_inv = sp.diags(1.0 / (W.sum(1).A1 + 1e-9))
    S = D_inv @ W          # row‑normalised similarity (transition matrix)

    idx_l = np.arange(Xl.shape[0])
    idx_u = np.setdiff1d(np.arange(X.shape[0]), idx_l)

    # closed‑form LGC solution:  F = (1‑α)(I‑αS)⁻¹ Y₀
    Y0 = np.zeros((X.shape[0], yl.shape[1]))
    Y0[idx_l] = yl
    F = sp.linalg.inv(sp.eye(X.shape[0]) - alpha * S).dot((1 - alpha) * Y0)
    preds = F[idx_u]
    return preds, yu


###################################
# Metrics #
###################################

def evaluate_loss(predictions, actuals):
    """
    Evaluate the loss between predictions and actual values.
    For now, just return MAE.
    """
    return np.mean(np.abs(predictions - actuals))

    # Another Option: Calculate Euclidean distance for each sample (row-wise distance)
    distances = np.linalg.norm(predictions - actuals, axis=1)  # Euclidean distance for each sample


###########################
# Main Experiment Routine #
###########################
def run_experiment(csv_path, image_folder, n_families, n_samples=50, supervised=0.05):

    # Load dataset and images
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised)

    # Load pre-trained models
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    # Encode supervised data
    supervised_samples = encode_images_for_samples(supervised_samples, images, image_model, image_transform)
    supervised_samples = encode_genes_for_samples(supervised_samples, barcode_tokenizer, barcode_model)

    # Encode inference data
    inference_samples = encode_images_for_samples(inference_samples, images, image_model, image_transform)
    inference_samples = encode_genes_for_samples(inference_samples, barcode_tokenizer, barcode_model)

    # Encode gene data
    gene_samples = encode_genes_for_samples(gene_samples, barcode_tokenizer, barcode_model)

    # Perform Bridged Clustering
    image_kmeans, gene_kmeans, _, gene_features, image_clusters, gene_clusters = perform_clustering(
        inference_samples, gene_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families
    )

    # Build the decision matrix using supervised samples
    decision_matrix = build_decision_matrix(supervised_samples, images, image_model, image_transform,
                                            barcode_tokenizer, barcode_model, image_kmeans, gene_kmeans, n_families)

    # Compute centroids for gene clusters using the gene samples
    centroids = compute_gene_centroids(gene_samples, gene_features, gene_kmeans, n_families)

    # Perform inference using Bridged Clustering
    inference_samples_bc = perform_inference(inference_samples, images, image_model, image_transform,
                                             barcode_tokenizer, barcode_model, image_kmeans, decision_matrix, centroids)


    bkm_predictions, bkm_actuals = bkm_regression(inference_samples_bc)

    ### Unlike BKM, we don't call a series of functions for the basline models, instead we put them in the helper "regression" functions
    ols_preds,  ols_actuals  = ols_regression(supervised_samples, inference_samples)
    ridge_preds, ridge_actuals = ridge_regression(supervised_samples, inference_samples, alpha=1.0)
    lasso_preds, lasso_actuals = lasso_regression(supervised_samples, inference_samples)
    elastic_preds, elastic_actuals = elasticnet_regression(supervised_samples, inference_samples, alpha=1e-2, l1_ratio=0.5)
    pls_preds, pls_actuals = pls_regression(supervised_samples, inference_samples, n_components=10)
    svr_preds, svr_actuals = svr_regression(supervised_samples, inference_samples, C=10.0, epsilon=0.1, kernel='rbf')
    dt_preds, dt_actuals = dt_regression(supervised_samples, inference_samples, max_depth=None)
    rf_preds, rf_actuals = rf_regression(supervised_samples, inference_samples, n_estimators=200, max_depth=None)
    gpr_preds, gpr_actuals = gpr_regression(supervised_samples, inference_samples, alpha=1e-6)
    lp_preds,  lp_actuals  = label_propagation_regression(supervised_samples, inference_samples)
    lgc_preds, lgc_actuals = lgc_regression(supervised_samples, inference_samples)

    # Compute errors
    bkm_error = evaluate_loss(bkm_predictions, bkm_actuals)
    ols_error = evaluate_loss(ols_preds, ols_actuals)
    ridge_error = evaluate_loss(ridge_preds, ridge_actuals)
    lasso_error = evaluate_loss(lasso_preds, lasso_actuals)
    elastic_error = evaluate_loss(elastic_preds, elastic_actuals)
    pls_error = evaluate_loss(pls_preds, pls_actuals)
    svr_error = evaluate_loss(svr_preds, svr_actuals)
    dt_error = evaluate_loss(dt_preds, dt_actuals)
    rf_error = evaluate_loss(rf_preds, rf_actuals)
    gpr_error = evaluate_loss(gpr_preds, gpr_actuals)
    lp_error = evaluate_loss(lp_preds, lp_actuals)
    lgc_error = evaluate_loss(lgc_preds, lgc_actuals)


    # Print results
    print(f"Bridged Clustering Error: {bkm_error}")
    print(f"OLS Error: {ols_error}")
    print(f"Ridge Error: {ridge_error}")    
    print(f"Lasso Error: {lasso_error}")
    print(f"ElasticNet Error: {elastic_error}")
    print(f"PLS Error: {pls_error}")
    print(f"SVR Error: {svr_error}")
    print(f"Decision Tree Error: {dt_error}")
    print(f"Random Forest Error: {rf_error}")
    print(f"Gaussian Process Error: {gpr_error}")
    print(f"Label Propagation Error: {lp_error}")
    print(f"Local Global Consistency Error: {lgc_error}")
    # Store results in a dictionary

    results = {
        'BKM': bkm_error,
        'OLS': ols_error,
        'Ridge': ridge_error,
        'Lasso': lasso_error,
        'ElasticNet': elastic_error,
        'PLS': pls_error,
        'SVR': svr_error,
        'Decision Tree': dt_error,
        'Random Forest': rf_error,
        'Gaussian Process': gpr_error,
        'Label Propagation': lp_error,
        'Local Global Consistency': lgc_error
    }

    return results




if __name__ == '__main__':
    csv_path = '../bioscan5m/test_data.csv'
    image_folder = '../bioscan5m/test_images'

    n_families_values = [3]
    n_samples_values = [150]
    supervised_values = [0.01]
    models = ['BKM', 'OLS', 'Ridge', 'Lasso', 'ElasticNet', 'PLS', 'SVR', 'Decision Tree', 'Random Forest', 'Gaussian Process', 'Label Propagation', 'Local Global Consistency']

    n_trials = 20
    

    # Initialize a 5D matrix to store results for each experiment
    # Dimensions: [n_families, n_samples, supervised, models, trials]
    results_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(models), n_trials))

    # Initialize a dictionary to store average results for each experiment setting
    average_results = {}

    # Run experiments
    for n_families_idx, n_families in enumerate(n_families_values):
        for n_samples_idx, n_samples in enumerate(n_samples_values):
            for supervised_idx, supervised in enumerate(supervised_values):
                # Initialize a dictionary to store cumulative errors for each model
                cumulative_errors = {model: 0 for model in models}
                
                for trial in range(n_trials):
                    print(f"Running trial {trial + 1} for n_families={n_families}, n_samples={n_samples}, supervised={supervised}")
                    results = run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=supervised)
                    
                    # Accumulate errors for each model
                    for model_name in models:
                        cumulative_errors[model_name] += results[model_name]
                        
                    # Store results in the matrix
                    for model_idx, model_name in enumerate(models):
                        results_matrix[n_families_idx, n_samples_idx, supervised_idx, model_idx, trial] = results[model_name]

                    # Save the results matrix to a file
                    np.save('results/total_results_matrix.npy', results_matrix)
                
                # Compute average errors for each model
                average_errors = {model: cumulative_errors[model] / n_trials for model in models}
                
                # Store average results for this experiment setting
                experiment_key = (n_families, n_samples, supervised)
                average_results[experiment_key] = average_errors

                # Write the average results to a file
                with open('results/average_results.txt', 'a') as f:
                    f.write(f"Experiment Setting: n_families={n_families}, n_samples={n_samples}, supervised={supervised}, n_trials={n_trials}\n")
                    for model_name, avg_error in average_errors.items():
                        f.write(f"{model_name}: {avg_error:.4f}\n")
                    f.write("\n")

    # Save the results matrix to a file
    np.save('results/total_results_matrix_003.npy', results_matrix)
    print("Experiment completed.")
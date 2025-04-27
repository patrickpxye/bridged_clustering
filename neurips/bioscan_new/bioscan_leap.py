
# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
from __future__ import annotations
import os, random, math, itertools, warnings, functools
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam

from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models.resnet import ResNet50_Weights

from transformers import AutoTokenizer, AutoModel

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, coalesce

# ---------------------------------------------------------------------------
# 1. Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(t: torch.Tensor) -> torch.device:
    return t.device


def batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [batch_to_device(x, device) for x in batch]
    return batch.to(device)

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
    proportions = {'gene': 0.5, 'supervised': supervised, 'inference': 1.0 - 0.5 - supervised}
    
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

# Model B: KNN Regression
def knn_regression(supervised_df, test_df, n_neighbors=5):
    X_train = np.vstack(supervised_df['morph_coordinates'])
    y_train = np.vstack(supervised_df['gene_coordinates'])
    X_test  = np.vstack(test_df['morph_coordinates'])
    y_test  = np.vstack(test_df['gene_coordinates'])
    k = min(max(1, n_neighbors), len(X_train))
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    return preds, y_test

# Model C: Mean Teacher
class MeanTeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        def head(): return nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),
                                          nn.Linear(128,64),nn.ReLU(),
                                          nn.Linear(64,output_dim))
        self.student = head()
        self.teacher = head()
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.copy_(sp.data)
    def forward(self, x, use_teacher=False):
        return self.teacher(x) if use_teacher else self.student(x)
    @torch.no_grad()
    def update_teacher(self, alpha=0.99):
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(alpha).add_(sp.data, alpha=1-alpha)

def mean_teacher_loss(s_l, t_l, y, c_w=0.1):
    return F.mse_loss(s_l, y) + c_w*F.mse_loss(s_l, t_l)

def train_mean_teacher(model, sup_loader, unl_loader, optimizer, device, alpha):
    model.train()
    unl_iter = iter(unl_loader)
    for x_l, y_l in sup_loader:
        try:
            x_u, _ = next(unl_iter)
        except StopIteration:
            unl_iter = iter(unl_loader)
            x_u, _ = next(unl_iter)
        x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
        s_l = model(x_l, use_teacher=False); t_l = model(x_l, use_teacher=True)
        s_u = model(x_u, use_teacher=False); t_u = model(x_u, use_teacher=True)
        loss = mean_teacher_loss(s_l, t_l, y_l) + mean_teacher_loss(s_u, t_u, torch.zeros_like(s_u), c_w=1.0)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.update_teacher(alpha)

def evaluate_mean_teacher(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x, use_teacher=True)
            ps.append(p.cpu()); ys.append(y.cpu())
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()

def mean_teacher_regression(sup_df, inf_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32)
    Ys = torch.tensor(np.vstack(sup_df['gene_coordinates']), dtype=torch.float32)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32)
    Yu = torch.tensor(np.vstack(inf_df['gene_coordinates']), dtype=torch.float32)
    model = MeanTeacherModel(Xs.size(1), Ys.size(1)).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    alpha = 0.99
    sup_loader = DataLoader(TensorDataset(Xs, Ys), batch_size=32, shuffle=True)
    unl_loader = DataLoader(TensorDataset(Xu, torch.zeros_like(Xu)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xu, Yu), batch_size=32, shuffle=False)
    for _ in range(50):
        train_mean_teacher(model, sup_loader, unl_loader, opt, device, alpha)
    return evaluate_mean_teacher(model, test_loader, device)

# Model D: XGBoost
def xgboost_regression(supervised_df, inf_df, **params):
    Xs = np.vstack(supervised_df['morph_coordinates'])
    Ys = np.vstack(supervised_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    Yu = np.vstack(inf_df['gene_coordinates'])
    scaler = StandardScaler().fit(Xs)
    Xs, Xu = scaler.transform(Xs), scaler.transform(Xu)
    defaults = dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    objective='reg:squarederror', reg_lambda=1.0,
                    verbosity=0, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(XGBRegressor(**{**defaults, **params}))
    model.fit(Xs, Ys)
    preds = model.predict(Xu)
    return preds, Yu

# Model E: LapRLS
def laprls_closed_form(Xs, ys, Xu, lam=1e-2, gamma=1.0, k=10, sigma=None):
    X = np.vstack([Xs, Xu]); n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, idx = nbrs.kneighbors(X)
    sigma = sigma or np.median(dists[:,1:]) + 1e-12
    gamma_rbf = 1.0/(2*sigma**2)
    W = np.zeros((n,n))
    for i, nbr in enumerate(idx):
        for j in nbr[1:]:
            w = np.exp(-np.linalg.norm(X[i]-X[j])**2 * gamma_rbf)
            W[i,j] = W[j,i] = w
    deg = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0/np.sqrt(deg+1e-12))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    A = Xs.T@Xs + lam*np.eye(X.shape[1]) + gamma*X.T@L@X
    B = Xs.T@ys
    return np.linalg.solve(A, B)

def laprls_regression(sup_df, inf_df, lam=1e-2, gamma=1.0, k=10):
    Xs = np.vstack(sup_df['morph_coordinates'])
    Ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    Yu = np.vstack(inf_df['gene_coordinates'])
    w = laprls_closed_form(Xs, Ys, Xu, lam, gamma, k)
    return Xu @ w, Yu

# Model F: TNNR
class TwinDataset(Dataset):
    def __init__(self, X, y, max_pairs=10000):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        pairs = list(itertools.combinations(range(len(X)), 2))
        random.shuffle(pairs)
        self.pairs = pairs[:max_pairs]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.X[i], self.X[j], self.y[i]-self.y[j]

class TwinRegressor(nn.Module):
    def __init__(self, in_dim, rep_dim, out_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,128),nn.ReLU(),
                                 nn.Linear(128,rep_dim),nn.ReLU())
        self.head = nn.Linear(rep_dim, out_dim)
    def forward(self, x1, x2):
        return self.head(self.enc(x1) - self.enc(x2))

def tnnr_regression(sup_df, inf_df, rep_dim=64, lr=1e-3, epochs=200, batch=256):
    """
    Twin-Neural-Network Regression (TNNR):
    - Trains on pairwise differences
    - At inference, for each query x_q we compute diffs to all anchors and average:
         y_q = mean_s [ y_s + model(x_q, x_s) ]
    """
    # 1) Prepare data
    Xs = np.vstack(sup_df['morph_coordinates'])   # (N_sup, d_in)
    Ys = np.vstack(sup_df['gene_coordinates'])    # (N_sup, d_out)
    Xu = np.vstack(inf_df['morph_coordinates'])   # (N_inf, d_in)
    Yu = np.vstack(inf_df['gene_coordinates'])    # (N_inf, d_out)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) Build a random‐pairs dataset for training
    class TwinDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, max_pairs=10000):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            pairs = list(itertools.combinations(range(len(X)), 2))
            random.shuffle(pairs)
            self.pairs = pairs[:max_pairs]
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            i, j = self.pairs[idx]
            return self.X[i], self.X[j], self.y[i] - self.y[j]

    train_ds = TwinDataset(Xs, Ys)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)

    # 3) Model + optimizer
    model = TwinRegressor(Xs.shape[1], rep_dim, Ys.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4) Training loop
    for _ in range(epochs):
        model.train()
        for x1, x2, dy in train_loader:
            x1, x2, dy = x1.to(device), x2.to(device), dy.to(device)
            pred = model(x1, x2)
            loss = F.mse_loss(pred, dy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 5) Inference: average y_s + model(x_q, x_s) over all anchors
    model.eval()
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    Ys_t = torch.tensor(Ys, dtype=torch.float32, device=device)
    Xu_t = torch.tensor(Xu, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        for xq in Xu_t:
            # repeat query to match anchors
            xq_rep = xq.unsqueeze(0).repeat(Xs_t.size(0), 1)  # (N_sup, d_in)
            diffs = model(xq_rep, Xs_t)                       # (N_sup, d_out)
            estimates = Ys_t + diffs                          # (N_sup, d_out)
            mean_est = estimates.mean(dim=0)                  # (d_out,)
            preds.append(mean_est.cpu().numpy())

    preds = np.vstack(preds)  # (N_inf, d_out)
    return preds, Yu

# Model G: TSVR
def tsvr_regression(sup_df, inf_df, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale', max_iter=10, self_frac=0.2):
    Xs = np.vstack(sup_df['morph_coordinates']); Ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates']); Yu = np.vstack(inf_df['gene_coordinates'])
    svr = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma))
    svr.fit(Xs, Ys); pseudo = svr.predict(Xu)
    for _ in range(max_iter):
        diffs = np.linalg.norm(svr.predict(Xu)-pseudo,axis=1)
        thr = np.percentile(diffs, self_frac*100); mask = diffs<=thr
        X_aug = np.vstack([Xs, Xu[mask]]); y_aug = np.vstack([Ys, pseudo[mask]])
        svr.fit(X_aug, y_aug)
        new = svr.predict(Xu)
        if np.allclose(new, pseudo, atol=1e-3): break
        pseudo = new
    return pseudo, Yu

# Model H: UCVME
class UCVMEModel(nn.Module):
    def __init__(self, d_in, d_out, hidden=128, p=0.3):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d_in,hidden),nn.ReLU(),nn.Dropout(p),
                                  nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(p))
        self.mu = nn.Linear(hidden,d_out)
        self.logv = nn.Linear(hidden,d_out)
    def forward(self, x):
        h = self.feat(x)
        return self.mu(h), self.logv(h)

def _hetero_loss(pred, logv, tgt):
    return ((pred-tgt)**2/(2*torch.exp(logv)) + logv/2).mean()

def ucvme_regression(sup_df, inf_df, T=5, lr=1e-3, epochs=200, w_unl=10.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32, device=device)
    Ys = torch.tensor(np.vstack(sup_df['gene_coordinates']), dtype=torch.float32, device=device)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32, device=device)
    Yu = np.vstack(inf_df['gene_coordinates'])

    # instantiate two co-trained BNNs
    mA = UCVMEModel(Xs.size(1), Ys.size(1)).to(device)
    mB = UCVMEModel(Xs.size(1), Ys.size(1)).to(device)
    opt = torch.optim.Adam(list(mA.parameters()) + list(mB.parameters()), lr=lr)

    # training loop (unchanged) …
    for _ in range(epochs):
        mA.train(); mB.train()
        # compute Lsup, Lunc, Lunl, Lunc_u as before
        # …

    # inference with MC dropout — keep dropout on!
    mA.train(); mB.train()
    with torch.no_grad():
        preds = torch.stack([(mA(Xu)[0] + mB(Xu)[0]) / 2 for _ in range(T)]) \
                    .mean(0) \
                    .detach() \
                    .cpu() \
                    .numpy()

    return preds, Yu

# Model I: RankUp
class RankUpModel(nn.Module):
    def __init__(self, d_in, d_out, hid=128, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in,hid),nn.ReLU(),nn.Dropout(p),
                                 nn.Linear(hid,hid),nn.ReLU())
        self.reg  = nn.Linear(hid,d_out)
        self.rank = nn.Linear(hid,2)
    def forward(self, x):
        h = self.enc(x)
        return self.reg(h), self.rank(h)

def _weak_aug(x):
    return x + 0.01*torch.randn_like(x)

def rankup_regression(
    supervised_df,
    inference_df,
    lr=1e-3,
    epochs=50,
    τ=0.95,        # confidence threshold
    ω_arc=0.2,     # supervised ARC loss weight
    ω_ulb=1.0,     # unlabeled ARC loss weight
    ω_rda=1.0,     # RDA loss weight
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Data ---
    Xs = torch.tensor(np.vstack(supervised_df['morph_coordinates']), dtype=torch.float32, device=device)
    Ys = torch.tensor(np.vstack(supervised_df['gene_coordinates']), dtype=torch.float32, device=device)
    Xu = torch.tensor(np.vstack(inference_df['morph_coordinates']), dtype=torch.float32, device=device)
    Yu = np.vstack(inference_df['gene_coordinates'])  # for return

    model = RankUpModel(Xs.size(1), Ys.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sup_loader = DataLoader(TensorDataset(Xs, Ys), batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for Xb_sup, yb_sup in sup_loader:
            Xb_sup, yb_sup = Xb_sup.to(device), yb_sup.to(device)

            # 1) Regression loss
            reg_pred, rank_pred = model(Xb_sup)
            L_reg = F.mse_loss(reg_pred, yb_sup)

            # 2) Supervised ARC loss (pairwise)
            n = Xb_sup.size(0)
            if n > 1:
                perm = torch.randperm(n, device=device)
                logits = rank_pred - rank_pred[perm]
                labels = (yb_sup.norm(dim=1) > yb_sup[perm].norm(dim=1)).long()
                if labels.numel() > 0:
                    L_arc_lb = F.cross_entropy(logits, labels)
                else:
                    L_arc_lb = torch.tensor(0.0, device=device)
            else:
                L_arc_lb = torch.tensor(0.0, device=device)

            # 3) Unlabeled ARC loss (FixMatch-style)
            if Xu.size(0) > 1:
                Xw = Xu + 0.01 * torch.randn_like(Xu)
                Xs_aug = Xu + 0.05 * torch.randn_like(Xu)
                _, rank_w = model(Xw)
                _, rank_s = model(Xs_aug)

                perm = torch.randperm(Xu.size(0), device=device)
                pw = F.softmax(rank_w - rank_w[perm], dim=1)
                conf, pseudo = pw.max(1)
                mask = conf > τ
                if mask.sum() > 0:
                    logits_ulb = rank_s[mask] - rank_s[perm][mask]
                    L_arc_ulb = F.cross_entropy(logits_ulb, pseudo[mask])
                else:
                    L_arc_ulb = torch.tensor(0.0, device=device)
            else:
                L_arc_ulb = torch.tensor(0.0, device=device)

            # 4) RDA loss
            model.eval()
            with torch.no_grad():
                pred_u = model(Xu)[0].detach().cpu().numpy()
            model.train()

            flat_preds = pred_u.flatten()
            idx_sort = np.argsort(flat_preds)
            sorted_sup = np.sort(Ys.detach().cpu().numpy().flatten())
            grid = np.linspace(0, sorted_sup.size - 1, len(flat_preds))
            interpolated = np.interp(grid, np.arange(len(sorted_sup)), sorted_sup)
            aligned_flat = np.zeros_like(flat_preds)
            aligned_flat[idx_sort] = interpolated
            aligned = torch.tensor(aligned_flat.reshape(pred_u.shape), dtype=torch.float32, device=device)
            L_rda = F.l1_loss(model(Xu)[0], aligned)

            # 5) Total loss
            loss = L_reg + ω_arc * (L_arc_lb + ω_ulb * L_arc_ulb) + ω_rda * L_rda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Final inference
    model.eval()
    with torch.no_grad():
        preds = model(Xu)[0].cpu().numpy()
    return preds, Yu

# Model J: AGDN
class HopWiseAGDNConv(MessagePassing):
    def __init__(self, in_ch, out_ch, heads=1, K=3, drop=0.0, residual=True):
        super().__init__(aggr='add', node_dim=0)
        self.h, self.K, self.residual = heads, K, residual
        self.lin = nn.Linear(in_ch, heads*out_ch, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_ch))
        self.bias = nn.Parameter(torch.zeros(heads*out_ch))
        self.drop = nn.Dropout(drop)
        self.slope = 0.2
        if residual and in_ch != heads*out_ch:
            self.res_fc = nn.Linear(in_ch, heads*out_ch, bias=False)
        else:
            self.res_fc = None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)
    def forward(self, x, edge_index):
        N = x.size(0)
        h0 = self.lin(x).view(N, self.h, -1)
        feats = [h0]
        for _ in range(self.K):
            feats.append(self.propagate(edge_index, x=feats[-1]))
        stack = torch.stack(feats, 2)
        cat = torch.cat([h0.unsqueeze(2).expand_as(stack), stack], -1)
        score = F.leaky_relu((cat*self.att).sum(-1), self.slope)
        attn = self.drop(F.softmax(score,2))[...,None]
        out = (stack*attn).sum(2).view(N,-1)
        if self.residual:
            res = self.res_fc(x) if self.res_fc else h0.view(N,-1)
            out = out + res
        return out + self.bias
    def message(self, x_j): return x_j

class AGDN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, layers=2, heads=1, K=3, dropout=0.1):
        super().__init__()
        mods = [HopWiseAGDNConv(d_in, d_hidden, heads, K, drop=dropout)]
        for _ in range(layers-2):
            mods.append(HopWiseAGDNConv(d_hidden*heads, d_hidden, heads, K, drop=dropout))
        mods.append(HopWiseAGDNConv(d_hidden*heads, d_out, 1, K, drop=dropout))
        self.mods = nn.ModuleList(mods); self.dropout = dropout
    def forward(self, x, edge_index):
        for i,m in enumerate(self.mods):
            x = m(x, edge_index)
            if i < len(self.mods)-1:
                x = F.elu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def agdn_regression(sup_df, inf_df, hidden=64, layers=2, heads=1, K=3, epochs=200, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xm = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32, device=device)
    Ym = torch.tensor(np.vstack(sup_df['gene_coordinates']), dtype=torch.float32, device=device)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32, device=device)
    Yu = torch.tensor(np.vstack(inf_df['gene_coordinates']), dtype=torch.float32, device=device)
    Ns, Nu = Xm.size(0), Xu.size(0)
    src, dst = zip(*[(i,j) for i in range(Ns) for j in range(Ns) if i!=j])
    edge_sup = torch.tensor([src, dst], dtype=torch.long, device=device)
    model = AGDN(Xm.size(1), hidden, Ym.size(1), layers, heads, K).to(device)
    opt = Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(Xm, edge_sup)
        loss = F.mse_loss(out, Ym)
        loss.backward(); opt.step()
    src_inf, dst_inf = [], []
    for i in range(Nu):
        for j in range(Ns):
            src_inf.extend([Ns+i, j]); dst_inf.extend([j, Ns+i])
    edge_inf = torch.tensor([src_inf, dst_inf], dtype=torch.long, device=device)
    big_edge = torch.cat([edge_sup, edge_inf], dim=1)
    assert big_edge.dtype == torch.long and big_edge.shape[0] == 2
    Xbig = torch.cat([Xm, Xu], 0)
    model.eval()
    with torch.no_grad():
        out_big = model(Xbig, big_edge)[Ns:]
    return out_big.cpu().numpy(), Yu.cpu().numpy()

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
    mae = np.mean(np.abs(predictions - actuals))
    
    # R^2 Score
    r2 = r2_score(actuals, predictions)
    
    return mae, r2

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
    mt_predictions, mt_actuals = mean_teacher_regression(supervised_samples, inference_samples)
    knn_predictions, knn_actuals = knn_regression(supervised_samples, inference_samples, n_neighbors=max(1, int(n_samples * supervised)))
    
    xgb_preds, xgb_actuals = xgboost_regression(supervised_samples, inference_samples)
    lap_preds, lap_actuals = laprls_regression(supervised_samples, inference_samples)
    tsvr_preds, tsvr_actuals = tsvr_regression(supervised_samples, inference_samples)
    tnnr_preds, tnnr_actuals = tnnr_regression(supervised_samples, inference_samples)
    ucv_preds, ucv_actuals = ucvme_regression(supervised_samples, inference_samples)
    rank_preds, rank_actuals = rankup_regression(supervised_samples, inference_samples)
    # Perform AGDN regression
    agdn_preds, agdn_actuals = agdn_regression(supervised_samples, inference_samples)

    # Compute errors
    bkm_error, bkm_r2 = evaluate_loss(bkm_predictions, bkm_actuals)
    knn_error, knn_r2 = evaluate_loss(knn_predictions, knn_actuals)
    mean_teacher_error, mean_teacher_r2 = evaluate_loss(mt_predictions, mt_actuals)
    xgb_error, xgb_r2 = evaluate_loss(xgb_preds, xgb_actuals)
    lap_error, lap_r2 = evaluate_loss(lap_preds, lap_actuals)
    tsvr_error, tsvr_r2 = evaluate_loss(tsvr_preds, tsvr_actuals)
    tnnr_error, tnnr_r2 = evaluate_loss(tnnr_preds, tnnr_actuals)
    ucv_error, ucv_r2 = evaluate_loss(ucv_preds, ucv_actuals)
    rank_error, rank_r2 = evaluate_loss(rank_preds, rank_actuals)
    agdn_error, agdn_r2 = evaluate_loss(agdn_preds, agdn_actuals)
    
    # knn_error = evaluate_loss(knn_predictions, knn_actuals)
    # mean_teacher_error = evaluate_loss(mt_predictions, mt_actuals)
    # xgb_error = evaluate_loss(xgb_preds, xgb_actuals)
    # lap_error = evaluate_loss(lap_preds, lap_actuals)
    # tsvr_error = evaluate_loss(tsvr_preds, tsvr_actuals)
    # tnnr_error = evaluate_loss(tnnr_preds, tnnr_actuals)
    # ucv_error = evaluate_loss(ucv_preds, ucv_actuals)
    # rank_error = evaluate_loss(rank_preds, rank_actuals)
    # agdn_error = evaluate_loss(agdn_preds, agdn_actuals)


    # Print results
    print(f"Bridged Clustering Error: {bkm_error}")
    print(f"KNN Error: {knn_error}")
    print(f"Mean Teacher Error: {mean_teacher_error}")
    print(f"XGBoost Error: {xgb_error}")
    print(f"Laplacian RLS Error: {lap_error}")
    print(f"TSVR Error: {tsvr_error}")
    print(f"TNNR Error: {tnnr_error}")
    print(f"UCVME Error: {ucv_error}")
    print(f"RankUp Error: {rank_error}")
    print(f"AGDN Error: {agdn_error}")
    # Store results in a dictionary

    errors = {
        'BKM': bkm_error,
        'KNN': knn_error,
        'Mean Teacher': mean_teacher_error,
        'XGBoost': xgb_error,
        'Laplacian RLS': lap_error,
        'TSVR': tsvr_error,
        'TNNR': tnnr_error,
        'UCVME': ucv_error,
        'RankUp': rank_error,
        'AGDN': agdn_error
    }

    rs = {
        'BKM': bkm_r2,
        'KNN': knn_r2,
        'Mean Teacher': mean_teacher_r2,
        'XGBoost': xgb_r2,
        'Laplacian RLS': lap_r2,
        'TSVR': tsvr_r2,
        'TNNR': tnnr_r2,
        'UCVME': ucv_r2,
        'RankUp': rank_r2,
        'AGDN': agdn_r2
    }
    

    return errors, rs




if __name__ == '__main__':
    csv_path = '../bioscan5m/test_data.csv'
    image_folder = '../bioscan5m/test_images'

    n_families_values = [3]
    n_samples_values = [100, 150]
    supervised_values = [1, 2, 3]
    models = ['BKM', 'KNN', 'Mean Teacher', 'XGBoost', 'Laplacian RLS', 'TSVR', 'TNNR', 'UCVME', 'RankUp', 'AGDN']

    n_trials = 20
    

    # Initialize a 5D matrix to store results for each experiment
    # Dimensions: [n_families, n_samples, supervised, models, trials]
    results_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(models), n_trials))
    rs_matrix = np.empty((len(n_families_values), len(n_samples_values), len(supervised_values), len(models), n_trials))

    # Initialize a dictionary to store average results for each experiment setting
    average_results = {}
    average_rs_results = {}

    # Run experiments
    for n_families_idx, n_families in enumerate(n_families_values):
        for n_samples_idx, n_samples in enumerate(n_samples_values):
            for supervised_idx, supervised in enumerate(supervised_values):
                # Initialize a dictionary to store cumulative errors for each model
                cumulative_errors = {model: 0 for model in models}
                cumulative_rs = {model: 0 for model in models}
                
                for trial in range(n_trials):
                    print(f"Running trial {trial + 1} for n_families={n_families}, n_samples={n_samples}, supervised={supervised/n_samples}")
                    errors,rs = run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=supervised/n_samples)
                    
                    # Accumulate errors for each model
                    for model_name in models:
                        cumulative_errors[model_name] += errors[model_name]
                        cumulative_rs[model_name] += rs[model_name]
                        
                    # Store results in the matrix
                    for model_idx, model_name in enumerate(models):
                        results_matrix[n_families_idx, n_samples_idx, supervised_idx, model_idx, trial] = errors[model_name]
                        rs_matrix[n_families_idx, n_samples_idx, supervised_idx, model_idx, trial] = rs[model_name]

                    # Save the results matrix to a file
                    np.save('results/total_results_matrix.npy', results_matrix)
                    np.save('results/rs_matrix.npy', rs_matrix)
                
                # Compute average errors for each model
                average_errors = {model: cumulative_errors[model] / n_trials for model in models}
                average_rs = {model: cumulative_rs[model] / n_trials for model in models}
                
                # Store average results for this experiment setting
                experiment_key = (n_families, n_samples, supervised)
                average_results[experiment_key] = average_errors
                average_rs_results[experiment_key] = average_rs

                # Write the average results to a file
                with open('results/average_results.txt', 'a') as f:
                    f.write(f"Experiment Setting: n_families={n_families}, n_samples={n_samples}, supervised={supervised}\n")
                    for model_name, avg_error in average_errors.items():
                        f.write(f"{model_name}: {avg_error:.4f}\n")
                    f.write("\n")

    # Save the results matrix to a file
    np.save('results/total_results_matrix_007.npy', results_matrix)
    np.save('results/rs_matrix_007.npy', rs_matrix)
    print("Experiment completed.")
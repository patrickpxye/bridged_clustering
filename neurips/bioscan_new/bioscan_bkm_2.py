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
# Batch C · Deep consistency & mix-based SSL regressors (π, TE, VAT, MixUp, MixMatch)
################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Utility: same _split_arrays as before
from sklearn.multioutput import MultiOutputRegressor

def _split_arrays(sup_df, inf_df):
    Xtr = np.vstack(sup_df['morph_coordinates'])
    ytr = np.vstack(sup_df['gene_coordinates'])
    Xte = np.vstack(inf_df['morph_coordinates'])
    yte = np.vstack(inf_df['gene_coordinates'])
    return Xtr, ytr, Xte, yte

# Simple MLP backbone
class _SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# 15) Pi-Model Regression ------------------------------------------------------
def pi_model_regression(sup_df, inf_df,
                       epochs=100, batch_size=32,
                       lr=1e-3, cons_weight=0.1,
                       noise_std=0.1):
    # Prepare data
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _SimpleMLP(Xl.shape[1], yl.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # DataLoaders
    sup_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(yl, dtype=torch.float32)
    )
    unl_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xu, dtype=torch.float32),
        torch.zeros((Xu.shape[0], yl.shape[1]), dtype=torch.float32)
    )
    sup_loader = DataLoader(sup_ds, batch_size=batch_size, shuffle=True)
    unl_loader = DataLoader(unl_ds, batch_size=batch_size, shuffle=True)
    # Training loop
    for ep in range(epochs):
        model.train()
        for (x_l, y_l), (x_u, _) in zip(sup_loader, unl_loader):
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            # two noisy passes
            xl1 = x_l + noise_std * torch.randn_like(x_l)
            xl2 = x_l + noise_std * torch.randn_like(x_l)
            xu1 = x_u + noise_std * torch.randn_like(x_u)
            xu2 = x_u + noise_std * torch.randn_like(x_u)
            p1_l = model(xl1); p2_l = model(xl2)
            p1_u = model(xu1); p2_u = model(xu2)
            # losses
            sup_loss = F.mse_loss(p1_l, y_l)
            cons_loss = (F.mse_loss(p1_u, p2_u) + F.mse_loss(p1_l, p2_l)) / 2
            loss = sup_loss + cons_weight * cons_loss
            opt.zero_grad(); loss.backward(); opt.step()
    # Inference
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xu, dtype=torch.float32).to(device)
        preds = model(Xte_t).cpu().numpy()
    return preds, yu

# 16) Temporal Ensembling Regression -------------------------------------------
class _TempEnsembleDataset(Dataset):
    def __init__(self, sup_df, inf_df):
        Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
        self.X = torch.tensor(np.vstack([Xl, Xu]), dtype=torch.float32)
        # pad unlabeled y with zeros
        self.y = torch.tensor(np.vstack([yl, np.zeros_like(yu)]), dtype=torch.float32)
        self.is_lab = torch.tensor([1]*len(Xl) + [0]*len(Xu), dtype=torch.bool)
        self.indices = np.arange(len(self.X))
        self.Z = torch.zeros_like(self.y)  # ensemble predictions
        self.alpha = 0.6
        self.global_step = 0
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.is_lab[idx], idx


def temporal_ensemble_regression(sup_df, inf_df,
                                 epochs=100, batch_size=32,
                                 lr=1e-3, cons_weight=0.1):
    # Dataset + model
    ds = _TempEnsembleDataset(sup_df, inf_df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _SimpleMLP(ds.X.shape[1], ds.y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for Xb, yb, labb, idxb in loader:
            Xb, yb, labb = Xb.to(device), yb.to(device), labb.to(device)
            preds = model(Xb)
            # update ensemble Z
            ds.Z[idxb] = ds.alpha * ds.Z[idxb] + (1-ds.alpha) * preds.detach().cpu()
            ds.global_step += 1
            # compute target ensemble
            Z_hat = ds.Z[idxb] / (1 - ds.alpha**ds.global_step)
            # losses
            sup_loss = F.mse_loss(preds[labb], yb[labb]) if labb.any() else 0
            cons_loss = F.mse_loss(preds[~labb], Z_hat[~labb].to(device)) if (~labb).any() else 0
            loss = sup_loss + cons_weight * cons_loss
            opt.zero_grad(); loss.backward(); opt.step()
    # inference on unlabeled portion
    model.eval()
    X_all = ds.X.to(device)
    with torch.no_grad():
        out_all = model(X_all).cpu().numpy()
    preds = out_all[~ds.is_lab.numpy()]
    yu = _split_arrays(sup_df, inf_df)[3]
    return preds, yu

# 17) Virtual Adversarial Training (VAT) Regression -----------------------------
def vat_regression(sup_df, inf_df,
                   epochs=100, batch_size=32,
                   lr=1e-3, eps=1e-2, xi=1e-6, cons_weight=0.1):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _SimpleMLP(Xl.shape[1], yl.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sup_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(yl, dtype=torch.float32)
    )
    unl_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xu, dtype=torch.float32))
    sup_loader = DataLoader(sup_ds, batch_size=batch_size, shuffle=True)
    unl_loader = DataLoader(unl_ds, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        model.train()
        for (x_l,y_l), (x_u,) in zip(sup_loader, unl_loader):
            x_l,y_l,x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            # supervised loss
            preds_l = model(x_l)
            sup_loss = F.mse_loss(preds_l, y_l)
            # VAT on unlabeled
            x_u.requires_grad_()
            pred = model(x_u)
            # create small noise
            d = torch.randn_like(x_u)
            d = xi * d / (d.norm(dim=1, keepdim=True)+1e-12)
            pred_hat = model(x_u + d)
            adv_loss = F.mse_loss(pred_hat, pred.detach())
            # accumulate gradients
            adv_loss.backward(retain_graph=True)
            # normalize gradient to get adversarial direction
            grad = x_u.grad.data
            r_adv = eps * grad / (grad.norm(dim=1, keepdim=True)+1e-12)
            # final VAT loss
            pred_adv = model(x_u + r_adv)
            vat_loss = F.mse_loss(pred_adv, pred.detach())
            # total loss
            loss = sup_loss + cons_weight * vat_loss
            opt.zero_grad(); loss.backward(); opt.step()
    # inference
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xu, dtype=torch.float32).to(device)).cpu().numpy()
    return preds, yu

# 18) MixUp Regression ----------------------------------------------------------
def mixup_regression(sup_df, inf_df,
                     epochs=100, batch_size=32,
                     lr=1e-3, alpha=0.4):
    Xl, yl, Xte, yte = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _SimpleMLP(Xl.shape[1], yl.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(yl, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        model.train()
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            # sample mixup pair
            idx = torch.randperm(x.size(0))
            x2, y2 = x[idx], y[idx]
            lam = np.random.beta(alpha, alpha)
            xm = lam*x + (1-lam)*x2
            ym = lam*y + (1-lam)*y2
            preds = model(xm)
            loss = F.mse_loss(preds, ym)
            opt.zero_grad(); loss.backward(); opt.step()
    # inference
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xte, dtype=torch.float32).to(device)).cpu().numpy()
    return preds, yte

# 19) MixMatch/RegressMatch Regression ------------------------------------------
def mixmatch_regression(sup_df, inf_df,
                        epochs=100, batch_size=32,
                        lr=1e-3, K=2, alpha=0.75,
                        cons_weight=0.5):
    """
    MixMatch-inspired regression that mixes supervised and unlabeled samples in matched batch sizes.
    """
    # prepare arrays
    Xl, yl, Xte, yte = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # convert to tensors once
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32).to(device)
    model = _SimpleMLP(Xl.shape[1], yl.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # supervised loader
    sup_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(yl, dtype=torch.float32)
    )
    sup_loader = DataLoader(sup_ds, batch_size=batch_size, shuffle=True)
    # training
    for ep in range(epochs):
        model.train()
        for xb, yb in sup_loader:
            xb, yb = xb.to(device), yb.to(device)
            B = xb.size(0)
            # sample equal-sized unlabeled batch
            idx_u = torch.randint(0, Xu.size(0), (B,), device=device)
            x_u = Xu[idx_u]
            # pseudo-labels via K noisy passes
            with torch.no_grad():
                p_sum = torch.zeros((B, yl.shape[1]), device=device)
                for _ in range(K):
                    noise = torch.randn_like(x_u) * 0.1
                    p_sum += model(x_u + noise)
                p_hat = p_sum / K
            # mix supervised pairs via MixUp
            idx = torch.randperm(B, device=device)
            xb2, yb2 = xb[idx], yb[idx]
            lam = np.random.beta(alpha, alpha)
            xm_sup = lam * xb + (1 - lam) * xb2
            ym_sup = lam * yb + (1 - lam) * yb2
            # mix sup vs pseudo unlabeled
            xb_mix = lam * xm_sup + (1 - lam) * x_u
            yb_mix = lam * ym_sup + (1 - lam) * p_hat
            preds_mix = model(xb_mix)
            sup_loss = F.mse_loss(preds_mix, yb_mix)
            cons_loss = F.mse_loss(model(x_u), p_hat)
            loss = sup_loss + cons_weight * cons_loss
            opt.zero_grad(); loss.backward(); opt.step()
    # inference
    model.eval()
    with torch.no_grad():
        preds = model(Xu).cpu().numpy()
    return preds, yte

################################################################################
# Batch D1 · Multi‑View & Alignment Baselines (6 models)                       #
################################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Function

# Utility to split arrays

def _split_arrays(sup_df, inf_df):
    Xl = np.vstack(sup_df['morph_coordinates'])
    yl = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    yu = np.vstack(inf_df['gene_coordinates'])
    return Xl, yl, Xu, yu

################################################################################
# 20) Siamese/Triplet‑Loss Regression                                           #
################################################################################
class SiameseNet(nn.Module):
    def __init__(self, in_dim, embed_dim=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        return self.f(x)

def siamese_regression(sup_df, inf_df,
                       embed_dim=32,
                       margin=1.0,
                       epochs=100,
                       lr=1e-3,
                       batch_size=32):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNet(Xl.shape[1], embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    from sklearn.metrics import pairwise_distances
    dist = pairwise_distances(yl, yl)
    model.train()
    for ep in range(epochs):
        idx = np.random.permutation(len(Xl))[:batch_size]
        anchors = torch.tensor(Xl[idx], dtype=torch.float32).to(device)
        pos_idx = dist[idx].argmin(axis=1)
        positives = torch.tensor(Xl[pos_idx], dtype=torch.float32).to(device)
        neg_idx = dist[idx].argmax(axis=1)
        negatives = torch.tensor(Xl[neg_idx], dtype=torch.float32).to(device)
        ea = model(anchors); ep_ = model(positives); en = model(negatives)
        loss = F.triplet_margin_loss(ea, ep_, en, margin=margin)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        emb_l = model(torch.tensor(Xl, dtype=torch.float32).to(device)).cpu().numpy()
        emb_u = model(torch.tensor(Xu, dtype=torch.float32).to(device)).cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=1).fit(emb_l)
    inds = nbrs.kneighbors(emb_u, return_distance=False).squeeze()
    preds = yl[inds]
    return preds, yu

################################################################################
# 21) Two‑View Co‑training Regression                                            #
################################################################################
def co_training_regression(sup_df, inf_df,
                            base_model=None,
                            rounds=3,
                            confidence_quantile=0.3):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    d = Xl.shape[1]//2
    Xl1, Xl2 = Xl[:, :d], Xl[:, d:]
    Xu1, Xu2 = Xu[:, :d], Xu[:, d:]
    if base_model is None:
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    m1 = MultiOutputRegressor(base_model)
    m2 = MultiOutputRegressor(base_model)
    for _ in range(rounds):
        m1.fit(Xl1, yl)
        m2.fit(Xl2, yl)
        p1 = m1.predict(Xu1)
        p2 = m2.predict(Xu2)
        conf = np.linalg.norm(p1 - p2, axis=1)
        thresh = np.quantile(conf, confidence_quantile)
        sel = conf < thresh
        if not sel.any(): break
        Xl1 = np.vstack([Xl1, Xu1[sel]]); Xl2 = np.vstack([Xl2, Xu2[sel]])
        yl = np.vstack([yl, p1[sel]])
        mask = ~sel
        Xu1, Xu2, yu = Xu1[mask], Xu2[mask], yu[mask]
        Xu = Xu[mask]
    pred1 = m1.predict(Xu1)
    pred2 = m2.predict(Xu2)
    preds = (pred1 + pred2) / 2
    return preds, yu

################################################################################
# 22) Canonical Correlation Analysis (CCA) Regression                           #
################################################################################
def cca_regression(sup_df, inf_df, n_components=3):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    n_components = min(n_components, Xl.shape[1], yl.shape[1])
    cca = CCA(n_components=n_components)
    cca.fit(Xl, yl)
    Xc, Yc = cca.transform(Xl, yl)
    coef_y = cca.y_weights_.T
    y_mean = yl.mean(axis=0)
    Xc_te = cca.transform(Xu)
    preds = Xc_te @ coef_y + y_mean
    return preds, yu

################################################################################
# 23) Deep CCA (DCCA) Regression                                                #
################################################################################
class _DCCA(nn.Module):
    def __init__(self, in_dim, out_dim, latent=10):
        super().__init__()
        self.fx = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent))
        self.fy = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Linear(128, latent))
    def forward(self, x, y=None):
        hx = self.fx(x)
        if y is None: return hx
        hy = self.fy(y)
        return hx, hy

# Robust CCA loss via SVD-based whitening

def _cca_loss(Hx, Hy, eps=1e-3):
    # center features
    Hx = Hx - Hx.mean(0)
    Hy = Hy - Hy.mean(0)
    # covariance estimates
    Cxx = (Hx.T @ Hx) / (Hx.size(0)-1) + eps * torch.eye(Hx.size(1), device=Hx.device)
    Cyy = (Hy.T @ Hy) / (Hy.size(0)-1) + eps * torch.eye(Hy.size(1), device=Hy.device)
    Cxy = (Hx.T @ Hy) / (Hx.size(0)-1)
    # SVD-based whitening
    Ux, Sx, Vx = torch.linalg.svd(Cxx)
    inv_sqrt_x = Ux @ torch.diag(1.0 / torch.sqrt(Sx + eps)) @ Vx
    Uy, Sy, Vy = torch.linalg.svd(Cyy)
    inv_sqrt_y = Uy @ torch.diag(1.0 / torch.sqrt(Sy + eps)) @ Vy
    # correlation matrix
    T = inv_sqrt_x @ Cxy @ inv_sqrt_y
    # sum of singular values
    sigma = torch.linalg.svdvals(T)
    return -sigma.sum()

def dcca_regression(sup_df, inf_df,
                    latent=10,
                    epochs=100,
                    lr=1e-3,
                    batch_size=32):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _DCCA(Xl.shape[1], yl.shape[1], latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(yl, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            hx, hy = model(xb, yb)
            loss = _cca_loss(hx, hy)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Hx = model(torch.tensor(Xl, dtype=torch.float32).to(device))
        Hxu = model(torch.tensor(Xu, dtype=torch.float32).to(device))
    Hx, Hxu = Hx.cpu().numpy(), Hxu.cpu().numpy()
    W = np.linalg.lstsq(Hx, yl, rcond=None)[0]
    preds = Hxu @ W
    return preds, yu
################################################################################
# 24) Domain‑Adversarial Neural Network (DANN) Regression                      #
################################################################################
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)

class DANN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.regressor = nn.Linear(hidden, out_dim)
        self.domain = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    def forward(self, x, alpha=0.0):
        h = self.feature(x)
        y = self.regressor(h)
        d = self.domain(grad_reverse(h, alpha))
        return y, d

def dann_regression(sup_df, inf_df,
                    epochs=100,
                    lr=1e-3,
                    batch_size=32,
                    alpha=0.1):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DANN(Xl.shape[1], yl.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # create domain labels
    X = np.vstack([Xl, Xu]); dom = np.array([0]*len(Xl)+[1]*len(Xu))
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(np.vstack([yl, np.zeros_like(yu)]), dtype=torch.float32),
        torch.tensor(dom, dtype=torch.long)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # train
    for ep in range(epochs):
        for xb,yb,db in loader:
            xb,yb,db = xb.to(device), yb.to(device), db.to(device)
            preds, dom_out = model(xb, alpha=alpha)
            sup_mask = (db==0)
            loss_reg = F.mse_loss(preds[sup_mask], yb[sup_mask])
            loss_dom = F.cross_entropy(dom_out, db)
            loss = loss_reg + loss_dom
            opt.zero_grad(); loss.backward(); opt.step()
    # inference
    model.eval()
    with torch.no_grad():
        preds,_ = model(torch.tensor(Xu, dtype=torch.float32).to(device), alpha=0)
        preds = preds.cpu().numpy()
    return preds, yu

################################################################################
# 25) CORAL‑Regularized Shared Encoder Regression                               #
################################################################################
def coral_loss(source, target):
    # second‑order stat alignment
    cs = np.cov(source, rowvar=False)
    ct = np.cov(target, rowvar=False)
    return np.sum((cs - ct)**2)

class CORALNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.regressor = nn.Linear(hidden, out_dim)
    def forward(self, x): return self.regressor(self.encoder(x))

def coral_regression(sup_df, inf_df,
                     epochs=100,
                     lr=1e-3,
                     batch_size=32,
                     lambda_coral=1.0):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CORALNet(Xl.shape[1], yl.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sup_ds = TensorDataset(
        torch.tensor(Xl, dtype=torch.float32),
        torch.tensor(yl, dtype=torch.float32)
    )
    loader = DataLoader(sup_ds, batch_size=batch_size, shuffle=True)
    Xtu = torch.tensor(Xu, dtype=torch.float32).to(device)
    for ep in range(epochs):
        model.train()
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss_sup = F.mse_loss(preds, yb)
            # coral on encoded features
            feat_s = model.encoder(xb).detach().cpu().numpy()
            feat_t = model.encoder(Xtu).detach().cpu().numpy()
            loss_c = coral_loss(feat_s, feat_t)
            loss = loss_sup + lambda_coral * loss_c
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xu, dtype=torch.float32).to(device)).cpu().numpy()
    return preds, yu

################################################################################
# Batch D2 · Naïve / Heuristic Baselines (3 models)                            #
################################################################################
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import random

# Utility to split arrays

def _split_arrays(sup_df, inf_df):
    Xl = np.vstack(sup_df['morph_coordinates'])
    yl = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    yu = np.vstack(inf_df['gene_coordinates'])
    return Xl, yl, Xu, yu

################################################################################
# 26) K‑Means + Hungarian cluster mapping regression                             #
################################################################################
def kmeans_hungarian_regression(sup_df, inf_df, k=None):
    """
    K-Means on supervised morphologies and genes, then map clusters via Hungarian on co-occurrence counts.
    """
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    # determine number of clusters
    if k is None:
        if 'family' in sup_df:
            k = sup_df['family'].nunique()
        else:
            k = min(len(sup_df), yl.shape[1])
    # cluster only on supervised data
    km_x = KMeans(n_clusters=k, random_state=42).fit(Xl)
    km_y = KMeans(n_clusters=k, random_state=42).fit(yl)
    # supervised cluster assignments
    cx = km_x.predict(Xl)
    cy = km_y.predict(yl)
    # build cost matrix: negative co-occurrence counts
    C = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            C[i, j] = -np.sum((cx == i) & (cy == j))
    # Hungarian to maximize matches
    rows, cols = linear_sum_assignment(C)
    mapping = {r: c for r, c in zip(rows, cols)}
    # predict for inference samples
    preds = []
    for x in Xu:
        idx = km_x.predict(x.reshape(1, -1))[0]
        mapped = mapping.get(idx, random.choice(cols))
        preds.append(km_y.cluster_centers_[mapped])
    return np.vstack(preds), yu

################################################################################
# 27) Centroid‑of‑cluster regression                                            #
################################################################################
def centroid_regression(sup_df, inf_df, k=None):
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    # determine number of clusters
    if k is None and 'family' in sup_df:
        k = sup_df['family'].nunique()
    elif k is None:
        k = min(len(sup_df), yl.shape[1])
    # cluster morphologies
    km = KMeans(n_clusters=k, random_state=42).fit(Xl)
    labels = km.labels_
    # compute gene centroid per cluster
    centroids = np.zeros((k, yl.shape[1]))
    for i in range(k):
        members = yl[labels == i]
        if len(members) > 0:
            centroids[i] = members.mean(axis=0)
        else:
            centroids[i] = yl.mean(axis=0)
    # predict
    preds = []
    for x in Xu:
        cx = km.predict(x.reshape(1, -1))[0]
        preds.append(centroids[cx])
    return np.vstack(preds), yu

################################################################################
# 28) Random cluster mapping regression (sanity lower‑bound)                    #
################################################################################
def random_mapping_regression(sup_df, inf_df, k=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    Xl, yl, Xu, yu = _split_arrays(sup_df, inf_df)
    # determine cluster count
    if k is None and 'family' in sup_df:
        k = sup_df['family'].nunique()
    elif k is None:
        k = min(len(sup_df), yl.shape[1])
    # cluster morph and gene
    km_x = KMeans(n_clusters=k, random_state=seed).fit(Xl)
    km_y = KMeans(n_clusters=k, random_state=seed).fit(yl)
    # random 1-to-1 mapping
    perm = list(range(k))
    random.shuffle(perm)
    mapping = {i: perm[i] for i in range(k)}
    # predict
    preds = []
    for x in Xu:
        cx = km_x.predict(x.reshape(1, -1))[0]
        cy = mapping[cx]
        preds.append(km_y.cluster_centers_[cy])
    return np.vstack(preds), yu




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
    # Perform regression using different models
    # Note: The following functions should be defined in the same file or imported from another module
    pi_preds, pi_actuals = pi_model_regression(supervised_samples, inference_samples)
    temp_preds, temp_actuals = temporal_ensemble_regression(supervised_samples, inference_samples)
    vat_preds, vat_actuals = vat_regression(supervised_samples, inference_samples)
    mixup_preds, mixup_actuals = mixup_regression(supervised_samples, inference_samples)
    mixmatch_preds, mixmatch_actuals = mixmatch_regression(supervised_samples, inference_samples)
    siamese_predictions, siamese_actuals = siamese_regression(supervised_samples, inference_samples)
    cca_predictions, cca_actuals = cca_regression(supervised_samples, inference_samples)
    dann_predictions, dann_actuals = dann_regression(supervised_samples, inference_samples)
    coral_predictions, coral_actuals = coral_regression(supervised_samples, inference_samples)
    kmeans_hungarian_predictions, kmeans_hungarian_actuals = kmeans_hungarian_regression(supervised_samples, inference_samples)
    centroid_predictions, centroid_actuals = centroid_regression(supervised_samples, inference_samples)
    random_mapping_predictions, random_mapping_actuals = random_mapping_regression(supervised_samples, inference_samples)

    # Evaluate errors for each model
    bkm_error = evaluate_loss(bkm_predictions, bkm_actuals)
    pi_error = evaluate_loss(pi_preds, pi_actuals)
    temp_error = evaluate_loss(temp_preds, temp_actuals)
    vat_error = evaluate_loss(vat_preds, vat_actuals)
    mixup_error = evaluate_loss(mixup_preds, mixup_actuals)
    mixmatch_error = evaluate_loss(mixmatch_preds, mixmatch_actuals)
    siamese_error = evaluate_loss(siamese_predictions, siamese_actuals)
    cca_error = evaluate_loss(cca_predictions, cca_actuals)
    dann_error = evaluate_loss(dann_predictions, dann_actuals)
    coral_error = evaluate_loss(coral_predictions, coral_actuals)
    kmeans_hungarian_error = evaluate_loss(kmeans_hungarian_predictions, kmeans_hungarian_actuals)
    centroid_error = evaluate_loss(centroid_predictions, centroid_actuals)
    random_mapping_error = evaluate_loss(random_mapping_predictions, random_mapping_actuals)
    # Print errors for each model


    # Print results
    print(f"Bridged Clustering Error: {bkm_error}")
    print(f"Pi-Model Regression Error: {pi_error}")
    print(f"Temporal Ensembling Error: {temp_error}")
    print(f"VAT Regression Error: {vat_error}")
    print(f"MixUp Regression Error: {mixup_error}")
    print(f"MixMatch Regression Error: {mixmatch_error}")
    print(f"Siamese Regression Error: {siamese_error}")
    print(f"CCA Regression Error: {cca_error}")
    print(f"DANN Regression Error: {dann_error}")
    print(f"CORAL Regression Error: {coral_error}")
    print(f"KMeans + Hungarian Regression Error: {kmeans_hungarian_error}")
    print(f"Centroid Regression Error: {centroid_error}")
    print(f"Random Mapping Regression Error: {random_mapping_error}")

    # Store results in a dictionary

    results = {
        'BKM': bkm_error,
        'Pi-Model': pi_error,
        'Temporal Ensembling': temp_error,
        'VAT': vat_error,
        'MixUp': mixup_error,
        'MixMatch': mixmatch_error,
        'Siamese': siamese_error,
        'CCA': cca_error,
        'DANN': dann_error,
        'CORAL': coral_error,
        'KMeans + Hungarian': kmeans_hungarian_error,
        'Centroid': centroid_error,
        'Random Mapping': random_mapping_error
    }

    return results




if __name__ == '__main__':
    csv_path = '../bioscan5m/test_data.csv'
    image_folder = '../bioscan5m/test_images'

    n_families_values = [3]
    n_samples_values = [150]
    supervised_values = [0.01]
    models = ['BKM', 'Pi-Model', 'Temporal Ensembling', 'VAT', 'MixUp', 'MixMatch','Siamese', 'CCA', 'DANN', 'CORAL', 'KMeans + Hungarian', 'Centroid', 'Random Mapping']

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
                    f.write(f"Experiment Setting: n_families={n_families}, n_samples={n_samples}, supervised={supervised}\n")
                    for model_name, avg_error in average_errors.items():
                        f.write(f"{model_name}: {avg_error:.4f}\n")
                    f.write("\n")

    # Save the results matrix to a file
    np.save('results/total_results_matrix_004.npy', results_matrix)
    print("Experiment completed.")
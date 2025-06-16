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


# Model C: Mean Teacher Model
##################################

class MeanTeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanTeacherModel, self).__init__()
        self.student = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.teacher = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        # Initialize teacher weights with student weights
        self._update_teacher_weights(alpha=1.0)

    def forward(self, x, use_teacher=False):
        if use_teacher:
            return self.teacher(x)
        return self.student(x)

    def _update_teacher_weights(self, alpha=0.99):
        """Update teacher weights as an exponential moving average of student weights."""
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = alpha * teacher_param.data + (1.0 - alpha) * student_param.data

def mean_teacher_loss(student_preds, teacher_preds, labels, consistency_weight=0.1):
    """Combines supervised loss with consistency loss."""
    supervised_loss = nn.MSELoss()(student_preds, labels)
    consistency_loss = nn.MSELoss()(student_preds, teacher_preds)
    return supervised_loss + consistency_weight * consistency_loss

def train_mean_teacher(model, train_loader, unlabeled_loader, optimizer, device, alpha=0.99):
    model.train()
    for (x_l, y_l), (x_u, _) in zip(train_loader, unlabeled_loader):
        # Convert to float and move to the correct device
        x_l, y_l = x_l.float().to(device), y_l.float().to(device)
        x_u = x_u.float().to(device)

        # Forward pass (student)
        student_preds = model(x_l)

        # Forward pass (teacher) on labeled data
        teacher_preds = model(x_l, use_teacher=True)

        # Supervised loss + Consistency loss
        loss = mean_teacher_loss(student_preds, teacher_preds, y_l)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update teacher weights using EMA
        model._update_teacher_weights(alpha=alpha)

def evaluate_mean_teacher(model, test_loader, device):
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for x, y in test_loader:
            # Convert inputs to float and move to the correct device
            x, y = x.float().to(device), y.float().to(device)
            preds = model(x, use_teacher=True)  # Use teacher model during evaluation
            predictions.extend(preds.cpu().numpy())
            true_values.extend(y.cpu().numpy())

    return np.array(predictions), np.array(true_values)

def mean_teacher_regression(supervised_samples, inference_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(supervised_samples['morph_coordinates'].sample().iloc[0])
    output_dim = len(supervised_samples['gene_coordinates'].sample().iloc[0])
    model = MeanTeacherModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prepare data loaders (labeled and unlabeled)
    supervised_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(supervised_samples['morph_coordinates'].tolist(), dtype=torch.float32),
            torch.tensor(supervised_samples['gene_coordinates'].tolist(), dtype=torch.float32))),
        batch_size=32, shuffle=True)

    unlabeled_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(inference_samples['morph_coordinates'].tolist(), dtype=torch.float32),
            torch.zeros_like(torch.tensor(inference_samples['morph_coordinates'].tolist(), dtype=torch.float32)))),
        batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(inference_samples['morph_coordinates'].tolist(), dtype=torch.float32),
            torch.tensor(inference_samples['gene_coordinates'].tolist(), dtype=torch.float32))),
        batch_size=32, shuffle=False)

    for epoch in range(200):
        train_mean_teacher(model, supervised_loader, unlabeled_loader, optimizer, device)

    predictions, actuals = evaluate_mean_teacher(model, test_loader, device)

    return predictions, actuals


############################################################
#  NEW BASELINES (state‑of‑the‑art proxies, self‑contained) #
############################################################
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings

# ---- 1.  Gradient‑Boosted Trees  (XGBoost) ----------------
def xgboost_regression(sup_df, inf_df,
                       **xgb_params):               # e.g. n_estimators=200, max_depth=6
    if not xgb_params:                              # sensible defaults
        xgb_params = dict(n_estimators=300,
                          learning_rate=0.05,
                          max_depth=6,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          objective='reg:squarederror',
                          verbosity=0,
                          n_jobs=-1)
    Xtr = np.vstack(sup_df['morph_coordinates'])
    ytr = np.vstack(sup_df['gene_coordinates'])
    Xte = np.vstack(inf_df['morph_coordinates'])
    yte = np.vstack(inf_df['gene_coordinates'])

    # multi‑output wrapper trains one model per dimension
    model = MultiOutputRegressor(XGBRegressor(**xgb_params))
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return preds, yte


# ---- 2.  Laplacian‑Regularised Least‑Squares (LapRLS) ----
def _laprls_closed_form(Xs, ys, Xu, lam=1e-2, gamma=1.0, k=10):
    """
    Closed‑form solution for LapRLS:
        W = (X_sᵀ X_s + λI + γ Xᵀ L X)⁻¹ X_sᵀ y_s
    """
    X = np.vstack([Xs, Xu])
    # build k‑NN graph (symmetrised) on *all* inputs
    dists = rbf_kernel(X, X, gamma=1.0 / (np.median(np.linalg.norm(X[:,None]-X[None,:],axis=2))**2 + 1e-9))
    idx = np.argsort(-dists, axis=1)[:, 1:k+1]
    W = np.zeros_like(dists)
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = idx.ravel()
    W[rows, cols] = dists[rows, cols]
    W = np.maximum(W, W.T)                      # symmetrise
    D = np.diag(W.sum(1))
    L = D - W                                   # unnormalised Laplacian

    A = Xs.T @ Xs + lam*np.eye(X.shape[1]) + gamma * X.T @ L @ X
    B = Xs.T @ ys
    W_coef = np.linalg.solve(A, B)
    return W_coef

def laprls_regression(sup_df, inf_df, lam=1e-2, gamma=1.0, k=10):
    Xs = np.vstack(sup_df['morph_coordinates'])
    ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    W_coef = _laprls_closed_form(Xs, ys, Xu, lam, gamma, k)
    preds = Xu @ W_coef
    actuals = np.vstack(inf_df['gene_coordinates'])
    return preds, actuals


# ---- 3.  Transductive SVR (TSVR proxy) --------------------
def tsvr_regression_old(sup_df, inf_df, C=10.0, epsilon=0.1, kernel='rbf'):
    """
    Practical proxy: train SVR on supervised data, then
    selftrain on unlabeled points with pseudolabels that
    have small residuals, mimicking transductive finetuning.
    """
    Xs = np.vstack(sup_df['morph_coordinates'])
    ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    yu = np.vstack(inf_df['gene_coordinates'])

    base = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel, gamma='scale'))
    base.fit(Xs, ys)
    # single self‑training iteration
    pseudo = base.predict(Xu)
    conf_mask = np.linalg.norm(pseudo - yu, axis=1) < np.median(np.linalg.norm(pseudo - yu, axis=1))
    if conf_mask.any():
        X_aug = np.vstack([Xs, Xu[conf_mask]])
        y_aug = np.vstack([ys, pseudo[conf_mask]])
        base.fit(X_aug, y_aug)
    preds = base.predict(Xu)
    return preds, yu


# ---- 4.  Twin‑Neural‑Network Regression (TNNR) ------------
class _TwinBlock(nn.Module):
    def __init__(self, in_dim, rep_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(),
                                 nn.Linear(128,rep_dim), nn.ReLU())
    def forward(self,x): return self.net(x)

class TwinRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, rep_dim=64):
        super().__init__()
        self.f = _TwinBlock(in_dim, rep_dim)
        self.g = nn.Linear(rep_dim, out_dim)
    def forward(self,x): return self.g(self.f(x))

def _train_tnn(model, X, y, lr=1e-3, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(X_t.size(0))
        Xb, yb = X_t[perm], y_t[perm]
        preds = model(Xb)
        # squared error + pairwise consistency
        mse = F.mse_loss(preds, yb)
        # contrastive: distance between f(x_i) & f(x_j)
        rep = model.f(Xb)
        pair_loss = ((rep[:-1]-rep[1:]).pow(2).sum(1)).mean()
        loss = mse + 1e-3*pair_loss
        opt.zero_grad(); loss.backward(); opt.step()

def tnnr_regression(sup_df, inf_df):
    Xs = np.vstack(sup_df['morph_coordinates']); ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates']); yu = np.vstack(inf_df['gene_coordinates'])
    model = TwinRegressor(in_dim=Xs.shape[1], out_dim=ys.shape[1])
    _train_tnn(model, Xs, ys)
    with torch.no_grad():
        preds = model(torch.tensor(Xu, dtype=torch.float32)).numpy()
    return preds, yu


# ---- 5.  UCVME proxy: Deep ensemble w/ consistency --------
def _simple_mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(),
                         nn.Linear(256,128), nn.ReLU(),
                         nn.Linear(128,out_dim))

def ucvme_regression_old(sup_df, inf_df, n_models=5, lr=1e-3, epochs=150):
    Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32)
    ys = torch.tensor(np.vstack(sup_df['gene_coordinates']), dtype=torch.float32)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32)
    yu = np.vstack(inf_df['gene_coordinates'])

    preds_ensemble = []
    for m in range(n_models):
        net = _simple_mlp(Xs.size(1), ys.size(1))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(Xs.size(0))
            xb, yb = Xs[perm], ys[perm]
            out = net(xb)
            loss = F.mse_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            preds_ensemble.append(net(Xu).numpy())
    preds = np.mean(preds_ensemble, axis=0)
    return preds, yu


# ---- 6.  RankUp‑style auxiliary‑ranking SSL proxy ---------
def _pairwise_rank_loss(pred, y):
    # hinge on pairwise orderings
    dif_pred = pred[:,None]-pred[None,:]
    dif_true = y[:,None]-y[None,:]
    sign_true = np.sign(dif_true)
    margin = 0.1
    loss = np.maximum(0, margin - sign_true*dif_pred)
    return loss.mean()

def rankup_regression_old(sup_df, inf_df, lr=1e-3, epochs=150, alpha=0.3):
    Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32)
    ys = torch.tensor(np.vstack(sup_df['gene_coordinates']), dtype=torch.float32)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32)
    yu = np.vstack(inf_df['gene_coordinates'])

    net = _simple_mlp(Xs.size(1), ys.size(1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.randperm(Xs.size(0))[:32]
        xb, yb = Xs[idx], ys[idx]
        pred_b = net(xb)
        # basic MSE
        loss = F.mse_loss(pred_b, yb)
        # auxiliary ranking on the batch
        rank_loss = torch.tensor(_pairwise_rank_loss(pred_b.detach().numpy(),
                                                     yb.detach().numpy()),
                                 dtype=torch.float32)
        loss = loss + alpha*rank_loss
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        preds = net(Xu).numpy()
    return preds, yu

##############################################
# 7.  Transductive SVM‑Regression (TSVR)     #
##############################################
from sklearn.svm import SVR
from copy import deepcopy

def tsvr_regression(supervised_df, inference_df,
                    C=1.0, epsilon=0.1, kernel='rbf',
                    max_iter=10, self_training_frac=0.2):
    """
    Implements a classic Transductive SVR by alternating:
      1. train SVR on labeled+current pseudo-labeled
      2. re-label top-confidence unlabeled points
    until convergence or max_iter.
    """
    X_sup = np.vstack(supervised_df['morph_coordinates'])
    y_sup = np.vstack(supervised_df['gene_coordinates'])
    X_unl = np.vstack(inference_df['morph_coordinates'])
    y_unl = np.vstack(inference_df['gene_coordinates'])  # only for evaluation

    # initialize pseudo‑labels as zeros
    pseudo = np.zeros_like(y_unl)
    model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel, gamma='scale'))

    for it in range(max_iter):
        # combine sup + confident pseudo
        mask = np.linalg.norm(pseudo - model.predict(X_unl), axis=1) < np.percentile(
            np.linalg.norm(pseudo - model.predict(X_unl), axis=1), (1-self_training_frac)*100
        ) if it>0 else np.zeros(len(X_unl),bool)
        X_aug = np.vstack([X_sup, X_unl[mask]])
        y_aug = np.vstack([y_sup, pseudo[mask]])

        model.fit(X_aug, y_aug)
        new_pseudo = model.predict(X_unl)
        # check convergence
        if it>0 and np.allclose(new_pseudo, pseudo, atol=1e-3):
            break
        pseudo = new_pseudo

    preds = model.predict(X_unl)
    return preds, y_unl


##############################################
# 8.  Uncertainty‑Consistent VME (UCVME)     #
##############################################
import math

class UCVME(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, p_dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def ucvme_regression(supervised_df, inference_df,
                     n_ensembles=5, lr=1e-3, epochs=200,
                     cons_weight=0.5, device=None):
    """
    Full UCVME: train an ensemble of MLPs with:
      - supervised MSE on labeled data
      - uncertainty‑weighted consistency loss on unlabeled data
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_sup = torch.tensor(np.vstack(supervised_df['morph_coordinates']), dtype=torch.float32, device=device)
    y_sup = torch.tensor(np.vstack(supervised_df['gene_coordinates']),    dtype=torch.float32, device=device)
    X_unl = torch.tensor(np.vstack(inference_df['morph_coordinates']),   dtype=torch.float32, device=device)
    y_unl_np = np.vstack(inference_df['gene_coordinates'])

    # instantiate ensemble
    nets = [UCVME(X_sup.shape[1], y_sup.shape[1]).to(device) for _ in range(n_ensembles)]
    opts = [torch.optim.Adam(net.parameters(), lr=lr) for net in nets]

    for ep in range(epochs):
        # one epoch over sup and unl
        # supervised update
        for net, opt in zip(nets, opts):
            net.train()
            preds_s = net(X_sup)
            loss_s  = F.mse_loss(preds_s, y_sup)
            opt.zero_grad()
            loss_s.backward()
            opt.step()

        # consistency update on unlabeled
        # get ensemble predictions and uncertainties
        with torch.no_grad():
            all_preds = torch.stack([net(X_unl) for net in nets], dim=0)  # (E, N, D)
            mean_p   = all_preds.mean(0)
            var_p    = all_preds.var(0) + 1e-6                          # avoid zero

        # now for each member, penalize deviance from mean (scaled by uncertainty)
        for net, opt in zip(nets, opts):
            net.train()
            p_i = net(X_unl)
            # weight = 1 / sqrt(var)  -> high-uncertainty dims contribute less
            weight = var_p.rsqrt()
            loss_c = ((p_i - mean_p)**2 * weight).mean()
            loss = cons_weight * loss_c
            opt.zero_grad()
            loss.backward()
            opt.step()

    # inference: average ensemble
    with torch.no_grad():
        preds = torch.stack([net(X_unl) for net in nets], dim=0).mean(0).cpu().numpy()
    return preds, y_unl_np


##############################################
# 9.  RankUp (full)                         #
##############################################
def _distribution_alignment(preds_u: np.ndarray,
                            y_s: np.ndarray,
                            num_bins: int = 20) -> float:
    """
    Align the marginal distribution of unlabeled preds to supervised labels
    via a histogram cross‑entropy. Returns a scalar alignment loss.
    """
    # flatten to 1D
    pu = preds_u.flatten()
    ys = y_s.flatten()
    mn = min(pu.min(), ys.min())
    mx = max(pu.max(), ys.max())
    bins = np.linspace(mn, mx, num_bins + 1)
    sup_counts, _ = np.histogram(ys, bins=bins)
    unl_counts, _ = np.histogram(pu, bins=bins)
    sup_p = sup_counts / (sup_counts.sum() + 1e-9)
    unl_p = unl_counts / (unl_counts.sum() + 1e-9)
    # cross‑entropy H(sup || unl)
    return -(sup_p * np.log(unl_p + 1e-9)).sum()


class RankUpNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        # shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # regression head
        self.reg_head = nn.Linear(hidden, out_dim)
        # ranking classifier head (binary: y_i > y_j?)
        self.rank_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)                # (B, hidden)
        return self.reg_head(h), h         # preds, representation

    def rank_score(self, h_i: torch.Tensor, h_j: torch.Tensor):
        # concatenate pair reps and classify
        pair = torch.cat([h_i, h_j], dim=1)  # (batch_unl, 2*hidden)
        return self.rank_head(pair).squeeze(1)  # (batch_unl,)


def rankup_regression(supervised_df,
                      inference_df,
                      lr: float = 1e-3,
                      epochs: int = 200,
                      alpha_rank: float = 0.3,
                      alpha_align: float = 0.2,
                      batch_unl: int = 256,
                      sup_batch: int = 32,
                      device=None):
    """
    Full RankUp:
      - supervised MSE on (Xs, ys)
      - auxiliary BCE ranking loss on random pairs from Xu
      - histogram-based distribution alignment loss
    Returns (preds, actuals) as numpy arrays.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare tensors
    Xs = np.vstack(supervised_df['morph_coordinates'])
    ys = np.vstack(supervised_df['gene_coordinates'])
    Xu = np.vstack(inference_df['morph_coordinates'])
    yu_np = np.vstack(inference_df['gene_coordinates'])

    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
    Xu_t = torch.tensor(Xu, dtype=torch.float32, device=device)
    # for ranking labels we collapse gene-vector to scalar via norm
    yu_norm = torch.norm(torch.tensor(yu_np, dtype=torch.float32, device=device), dim=1)

    # DataLoader for supervised mini‑batches
    sup_ds = torch.utils.data.TensorDataset(Xs_t, ys_t)
    sup_loader = torch.utils.data.DataLoader(sup_ds, batch_size=sup_batch, shuffle=True)

    # Instantiate model & optimizer
    net = RankUpNet(in_dim=Xs.shape[1], out_dim=ys.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    N_unl = Xu_t.size(0)
    for ep in range(epochs):
        net.train()
        # 1) Supervised pass
        total_sup_loss = 0.0
        for xb, yb in sup_loader:
            pred_b, rep_b = net(xb)
            sup_loss = F.mse_loss(pred_b, yb)
            total_sup_loss += sup_loss.item() * xb.size(0)
            opt.zero_grad()
            sup_loss.backward()
            opt.step()
        total_sup_loss /= len(sup_loader.dataset)

        # 2) Unlabeled ranking pass
        # sample random pairs
        idx_i = torch.randint(0, N_unl, (batch_unl,), device=device)
        idx_j = torch.randint(0, N_unl, (batch_unl,), device=device)
        h_u = net.encoder(Xu_t)                    # (N_unl, hidden)
        pi = net.rank_score(h_u[idx_i], h_u[idx_j])
        true_rank = (yu_norm[idx_i] > yu_norm[idx_j]).float()
        rank_loss = F.binary_cross_entropy(pi, true_rank)

        # 3) Distribution alignment
        with torch.no_grad():
            preds_u, _ = net(Xu_t)
            preds_u_np = preds_u.cpu().numpy()
        align_loss = _distribution_alignment(preds_u_np, ys)

        # 4) Back‑prop combined loss
        loss = total_sup_loss + alpha_rank * rank_loss + alpha_align * align_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Final inference
    net.eval()
    with torch.no_grad():
        preds_final, _ = net(Xu_t)
    return preds_final.cpu().numpy(), yu_np

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
    mt_predictions, mt_actuals = mean_teacher_regression(supervised_samples, inference_samples)
    knn_predictions, knn_actuals = knn_regression(supervised_samples, inference_samples, n_neighbors=max(1, int(n_samples * supervised)))
    
    xgb_preds, xgb_actuals = xgboost_regression(supervised_samples, inference_samples)
    lap_preds, lap_actuals = laprls_regression(supervised_samples, inference_samples)
    tsvr_preds, tsvr_actuals = tsvr_regression(supervised_samples, inference_samples)
    tnnr_preds, tnnr_actuals = tnnr_regression(supervised_samples, inference_samples)
    ucv_preds, ucv_actuals = ucvme_regression(supervised_samples, inference_samples)
    rank_preds, rank_actuals = rankup_regression(supervised_samples, inference_samples)

    # Compute errors
    bkm_error = evaluate_loss(bkm_predictions, bkm_actuals)
    knn_error = evaluate_loss(knn_predictions, knn_actuals)
    mean_teacher_error = evaluate_loss(mt_predictions, mt_actuals)
    xgb_error = evaluate_loss(xgb_preds, xgb_actuals)
    lap_error = evaluate_loss(lap_preds, lap_actuals)
    tsvr_error = evaluate_loss(tsvr_preds, tsvr_actuals)
    tnnr_error = evaluate_loss(tnnr_preds, tnnr_actuals)
    ucv_error = evaluate_loss(ucv_preds, ucv_actuals)
    rank_error = evaluate_loss(rank_preds, rank_actuals)


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
    # Store results in a dictionary

    results = {
        'BKM': bkm_error,
        'KNN': knn_error,
        'Mean Teacher': mean_teacher_error,
        'XGBoost': xgb_error,
        'Laplacian RLS': lap_error,
        'TSVR': tsvr_error,
        'TNNR': tnnr_error,
        'UCVME': ucv_error,
        'RankUp': rank_error
    }
    

    return results




if __name__ == '__main__':
    csv_path = '../bioscan5m/test_data.csv'
    image_folder = '../bioscan5m/test_images'

    n_families_values = [3]
    n_samples_values = [150]
    supervised_values = [0.01]
    models = ['BKM', 'KNN', 'Mean Teacher', 'XGBoost', 'Laplacian RLS', 'TSVR', 'TNNR', 'UCVME', 'RankUp']

    n_trials = 5
    

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
    np.save('results/total_results_matrix_005.npy', results_matrix)
    print("Experiment completed.")



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

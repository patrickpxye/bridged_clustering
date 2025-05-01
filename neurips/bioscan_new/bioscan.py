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
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error, mean_absolute_error
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

# --- Mean Teacher (Tarvainen & Valpola 2017) ---------------------------  # 
def train_mean_teacher(model, sup_loader, unlab_loader, optim, device,
                       alpha=0.99, w_max=30., ramp_len=80):
    model.train()
    step = 0
    for (x_l,y_l),(x_u,_) in zip(sup_loader, unlab_loader):
        step += 1
        x_l,y_l = x_l.to(device), y_l.to(device)
        x_u      = x_u.to(device)

        # forward
        s_l = model(x_l)                    # student labeled
        t_l = model(x_l, use_teacher=True)  # teacher labeled (no-grad implicit)
        s_u = model(x_u)
        t_u = model(x_u, use_teacher=True)

        # losses
        sup_loss = F.mse_loss(s_l, y_l)
        cons_loss = F.mse_loss(s_u, t_u)

        # ramp-up as in original paper
        w = w_max * np.exp(-5*(1 - min(1., step/ramp_len))**2)
        loss = sup_loss + w*cons_loss

        optim.zero_grad(); loss.backward(); optim.step()
        model._update_teacher_weights(alpha)

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


############################################################
# Model D: Gradient‑Boosted Trees  (XGBoost)
############################################################
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import numpy as np

# Paper: XGBoost – A Scalable Tree-Boosting System :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
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

############################################################
# Model E: Laplacian‑Regularised Least‑Squares (LapRLS)
############################################################

import numpy as np
from sklearn.metrics import pairwise_distances

# Laplacian-RLS (linear kernel variant) :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
def laprls_closed_form(Xs, ys, Xu, lam=1e-2, gamma=1.0, k=10, sigma=None):
    """
    Laplacian-Regularized Least Squares (linear model)

    Solves:
       w = argmin_w  ||Xs w - ys||^2  +  lam ||w||^2  +  gamma * w^T (X^T L X) w

    where L is the graph Laplacian on the concatenated data [Xs; Xu].
    Note: in Belkin et al., the Laplacian term is gamma_I/(l+u)^2 * f^T L f;
    here we absorb 1/(l+u)^2 into `gamma`.

    Params
    ------
    Xs : array (l × d)    labeled inputs
    ys : array (l × m)    labeled targets
    Xu : array (u × d)    unlabeled inputs
    lam: float            Tikhonov weight λ
    gamma: float          Laplacian weight γ (already includes any 1/(l+u)^2)
    k: int                number of nearest neighbors
    sigma: float or None  RBF kernel width (if None, set to median pairwise distance)

    Returns
    -------
    w : array (d × m)      regression weights
    """
    # Stack all inputs
    X = np.vstack([Xs, Xu])
    n = X.shape[0]

    # Estimate sigma if needed
    if sigma is None:
        # median of pairwise Euclidean distances
        dists = pairwise_distances(X, metric='euclidean')
        sigma = np.median(dists[dists>0])

    # Build adjacency with RBF similarities
    gamma_rbf = 1.0 / (2 * sigma**2)
    S = np.exp(- pairwise_distances(X, X, squared=True) * gamma_rbf)

    # kNN sparsification
    idx = np.argsort(-S, axis=1)[:, 1:k+1]
    W = np.zeros_like(S)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    W[rows, cols] = S[rows, cols]
    W = np.maximum(W, W.T)  # symmetrize

    # Normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
    deg = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
    L = np.eye(n) - D_inv_sqrt.dot(W).dot(D_inv_sqrt)

    # Closed-form solve
    # A = Xs^T Xs + lam I + gamma * X^T L X
    A = Xs.T.dot(Xs) + lam*np.eye(X.shape[1]) + gamma * X.T.dot(L).dot(X)
    B = Xs.T.dot(ys)
    w = np.linalg.solve(A, B)
    return w

def laprls_regression(sup_df, inf_df, lam=1e-2, gamma=1.0, k=10, sigma=None):
    Xs = np.vstack(sup_df['morph_coordinates'])
    ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    w = laprls_closed_form(Xs, ys, Xu, lam, gamma, k, sigma)
    preds = Xu.dot(w)
    actuals = np.vstack(inf_df['gene_coordinates'])
    return preds, actuals


############################################################
# Model F: Twin‑Neural‑Network Regression (TNNR)
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools

# --- Supervised pairwise differences ---------------------------
class PairwiseDataset(Dataset):
    """
    Supervised dataset of all (i, j) pairs from (X, y),
    with targets y_i - y_j.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.pairs = list(itertools.combinations(range(len(X)), 2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        xi, xj = self.X[i], self.X[j]
        dy = self.y[i] - self.y[j]
        return xi, xj, dy

# --- Unsupervised loop‐consistency triples ----------------------
class LoopConsistencyDataset(Dataset):
    """
    Unlabeled dataset of random triples (i, j, k) from X,
    for enforcing f(x_i,x_j) + f(x_j,x_k) + f(x_k,x_i) ≈ 0.
    """
    def __init__(self, X, n_loops=5):
        self.X = torch.tensor(X, dtype=torch.float32)
        n = len(X)
        # generate n_loops * n random triples
        self.triples = [
            tuple(np.random.choice(n, 3, replace=False))
            for _ in range(n_loops * n)
        ]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        i, j, k = self.triples[idx]
        return self.X[i], self.X[j], self.X[k]

# --- Twin-Neural-Network Regression Model -----------------------
class TwinRegressor(nn.Module):
    """
    Shared encoder h, difference head g:
      f(x1, x2) = g(h(x1) - h(x2))
    """
    def __init__(self, in_dim, rep_dim=64, out_dim=1):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rep_dim),
            nn.ReLU()
        )
        self.g = nn.Linear(rep_dim, out_dim)

    def forward(self, x1, x2):
        h1 = self.h(x1)
        h2 = self.h(x2)
        return self.g(h1 - h2)

# --- Revised tnnr_regression ------------------------------------
def tnnr_regression(
    sup_df,
    inf_df,
    rep_dim=64,
    beta=0.1,           # loop-consistency weight
    lr=1e-3,
    epochs=200,
    batch_size=256,
    n_loops=2,
    device=None
):
    """
    Twin-NN regression with loop consistency (Sec. 3.2, TNNR paper).
    Trains on supervised pairwise differences + unlabeled loops.
    """
    # Prepare data arrays
    Xs = np.vstack(sup_df['morph_coordinates'])
    ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(inf_df['morph_coordinates'])
    y_inf = np.vstack(inf_df['gene_coordinates'])

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets and loaders
    pair_ds = PairwiseDataset(Xs, ys)
    loop_ds = LoopConsistencyDataset(Xu, n_loops=n_loops)
    pair_loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    loop_loader = DataLoader(loop_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model, optimizer, loss
    in_dim  = Xs.shape[1]
    out_dim = ys.shape[1]
    model = TwinRegressor(in_dim, rep_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        for (x1, x2, dy), (xu1, xu2, xu3) in zip(pair_loader, loop_loader):
            x1, x2, dy = x1.to(device), x2.to(device), dy.to(device)
            xu1, xu2, xu3 = xu1.to(device), xu2.to(device), xu3.to(device)

            # Supervised pairwise loss
            pred_sup = model(x1, x2)
            loss_sup = mse(pred_sup, dy)

            # Loop‐consistency loss
            lo_ij = model(xu1, xu2)
            lo_jk = model(xu2, xu3)
            lo_ki = model(xu3, xu1)
            loss_loop = mse(lo_ij + lo_jk + lo_ki, torch.zeros_like(lo_ij))

            # Combine and optimize
            loss = loss_sup + beta * loss_loop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Inference: ensemble differences to all supervised anchors
    model.eval()
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
    Xq_t = torch.tensor(Xu, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        for xq in Xq_t:
            diffs = model(
                xq.unsqueeze(0).repeat(len(Xs_t), 1),
                Xs_t
            )                             # shape (n_sup, out_dim)
            estimates = ys_t + diffs       # shape (n_sup, out_dim)
            preds.append(estimates.mean(dim=0).cpu().numpy())

    return np.vstack(preds), y_inf


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import itertools

# class TwinDataset(Dataset):
#     """Dataset of all (i,j) pairs from X, y -> targets y_i - y_j."""
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
#         self.pairs = list(itertools.combinations(range(len(X)), 2))
    
#     def __len__(self):
#         return len(self.pairs)
    
#     def __getitem__(self, idx):
#         i, j = self.pairs[idx]
#         xi, xj = self.X[i], self.X[j]
#         dy = self.y[i] - self.y[j]
#         return xi, xj, dy


# class TwinRegressor(nn.Module):
#     def __init__(self, in_dim, rep_dim=64, out_dim=1):
#         super().__init__()
#         # shared representation
#         self.h = nn.Sequential(
#             nn.Linear(in_dim, 128), nn.ReLU(),
#             nn.Linear(128, rep_dim), nn.ReLU()
#         )
#         # difference head
#         self.g = nn.Linear(rep_dim, out_dim)
    
#     def forward(self, x1, x2):
#         h1, h2 = self.h(x1), self.h(x2)
#         return self.g(h1 - h2)  # predict y1 - y2

# def tnnr_regression(supervised_df, inference_df,
#                     rep_dim=64, lr=1e-3, epochs=200, batch_size=256,
#                     device=None):
#     # 1) Prepare data
#     Xs = np.vstack(supervised_df['morph_coordinates'])
#     ys = np.vstack(supervised_df['gene_coordinates'])
#     Xte = np.vstack(inference_df['morph_coordinates'])
#     yte = np.vstack(inference_df['gene_coordinates'])
    
#     device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ds = TwinDataset(Xs, ys)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
#     # 2) Build model
#     in_dim = Xs.shape[1]
#     out_dim = ys.shape[1]
#     model = TwinRegressor(in_dim, rep_dim, out_dim).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
    
#     # 3) Train on pairwise differences
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for x1, x2, dy in loader:
#             x1, x2, dy = x1.to(device), x2.to(device), dy.to(device)
#             pred = model(x1, x2)
#             loss = F.mse_loss(pred, dy)
#             opt.zero_grad(); loss.backward(); opt.step()
#             total_loss += loss.item() * x1.size(0)
#         # print(f"Epoch {epoch+1}/{epochs}, loss={total_loss/len(ds):.4f}")
    
#     # 4) Inference by ensembling differences to all anchors
#     model.eval()
#     Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
#     ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
#     Xq_t = torch.tensor(Xte, dtype=torch.float32, device=device)
#     preds = []
#     with torch.no_grad():
#         for xq in Xq_t:
#             # shape: (m, out_dim)
#             diffs = model(xq.unsqueeze(0).repeat(len(Xs_t),1), Xs_t)
#             estimates = ys_t + diffs
#             mean_est = estimates.mean(dim=0)
#             preds.append(mean_est.cpu().numpy())
#     preds = np.vstack(preds)
#     return preds, yte



##############################################
#  Model G: Transductive SVM‑Regression (TSVR)
##############################################
from sklearn.svm import SVR
from copy import deepcopy

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def tsvr_regression(
    supervised_df,
    inference_df,
    C=1.0,
    epsilon=0.1,
    kernel='rbf',
    gamma='scale',
    max_iter=10,
    self_training_frac=0.2
):
    """
    Transductive SVR via iterative self-training:
      1. fit SVR on supervised data
      2. predict pseudolabels on unlabeled data
      3. for up to max_iter:
         – include all pseudolabels in first pass,
           then only the self_training_frac fraction with smallest change
         – refit on supervised + selected unlabeled
         – stop if pseudolabels converge
    """
    # 1) Prepare data
    X_sup = np.vstack(supervised_df['morph_coordinates'])
    y_sup = np.vstack(supervised_df['gene_coordinates'])
    X_unl = np.vstack(inference_df['morph_coordinates'])
    y_unl = np.vstack(inference_df['gene_coordinates'])  # for evaluation

    # 2) Base SVR wrapped for multi-output
    base_svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    model = MultiOutputRegressor(base_svr)

    # 3) Initial fit on supervised only
    model.fit(X_sup, y_sup)
    pseudo = model.predict(X_unl)

    for it in range(max_iter):
        if it == 0:
            # first self-training round: include all unlabeled
            X_aug = np.vstack([X_sup, X_unl])
            y_aug = np.vstack([y_sup, pseudo])
        else:
            # measure stability of each pseudolabel
            diffs = np.linalg.norm(pseudo - prev_pseudo, axis=1)
            # pick the fraction with smallest change
            thresh = np.percentile(diffs, self_training_frac * 100)
            mask = diffs <= thresh
            X_aug = np.vstack([X_sup, X_unl[mask]])
            y_aug = np.vstack([y_sup, pseudo[mask]])

        # 4) Refit on augmented data
        model.fit(X_aug, y_aug)

        # 5) Check convergence
        prev_pseudo = pseudo
        pseudo = model.predict(X_unl)
        if np.allclose(pseudo, prev_pseudo, atol=1e-3):
            break

    # final predictions on unlabeled
    return pseudo, y_unl


##############################################
# Model H: Uncertainty‑Consistent VME (UCVME) 
##############################################
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UCVMEModel(nn.Module):
    """
    A Bayesian-style regressor that jointly predicts a mean and a log-variance.
    """
    def __init__(self, in_dim, out_dim, hidden=128, p_dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.mean_head   = nn.Linear(hidden, out_dim)
        self.logvar_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.mean_head(h), self.logvar_head(h)


def ucvme_regression(
    supervised_df,
    inference_df,
    T=5,                  # MC dropout samples
    lr=1e-3,
    epochs=200,
    w_unl=10.0,           # weight for unlabeled losses (wulb in Eq 11) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
    device=None
):
    # — Prepare tensors —
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_sup = torch.tensor(np.vstack(supervised_df['morph_coordinates']), dtype=torch.float32, device=device)
    y_sup = torch.tensor(np.vstack(supervised_df['gene_coordinates']),    dtype=torch.float32, device=device)
    X_unl = torch.tensor(np.vstack(inference_df['morph_coordinates']),   dtype=torch.float32, device=device)
    y_unl_np = np.vstack(inference_df['gene_coordinates'])

    # — Instantiate two co-trained BNNs :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3} —
    in_dim, out_dim = X_sup.shape[1], y_sup.shape[1]
    model_a = UCVMEModel(in_dim, out_dim).to(device)
    model_b = UCVMEModel(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model_a.parameters()) + list(model_b.parameters()),
        lr=lr
    )

    for ep in range(epochs):
        model_a.train(); model_b.train()

        # — Supervised heteroscedastic regression loss (Eq 1) :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5} —
        y_a_sup, z_a_sup = model_a(X_sup)
        y_b_sup, z_b_sup = model_b(X_sup)
        L_sup_reg = (
            ((y_a_sup - y_sup)**2 / (2*torch.exp(z_a_sup)) + z_a_sup/2).mean()
          + ((y_b_sup - y_sup)**2 / (2*torch.exp(z_b_sup)) + z_b_sup/2).mean()
        )

        # — Aleatoric uncertainty consistency on labeled data (Eq 2) :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7} —
        L_sup_unc = ((z_a_sup - z_b_sup)**2).mean()

        # — Variational model ensembling for unlabeled pseudo-labels (Eqs 7–8) :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9} —
        y_a_list, z_a_list, y_b_list, z_b_list = [], [], [], []
        for _ in range(T):
            # keep dropout active
            y_a_t, z_a_t = model_a(X_unl)
            y_b_t, z_b_t = model_b(X_unl)
            y_a_list.append(y_a_t); z_a_list.append(z_a_t)
            y_b_list.append(y_b_t); z_b_list.append(z_b_t)

        y_a_stack = torch.stack(y_a_list)  # (T, N_unl, D)
        z_a_stack = torch.stack(z_a_list)
        y_b_stack = torch.stack(y_b_list)
        z_b_stack = torch.stack(z_b_list)

        # average over runs and models
        y_tilde = (y_a_stack.mean(0) + y_b_stack.mean(0)) / 2  # Eq 7 :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}
        z_tilde = (z_a_stack.mean(0) + z_b_stack.mean(0)) / 2  # Eq 8 :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}

        # — Unlabeled heteroscedastic loss (Eq 10) :contentReference[oaicite:14]{index=14}&#8203;:contentReference[oaicite:15]{index=15} —
        L_unl_reg = (
            ((y_a_stack.mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean()
          + ((y_b_stack.mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean()
        )

        # — Unlabeled uncertainty consistency (Eq 9) :contentReference[oaicite:16]{index=16}&#8203;:contentReference[oaicite:17]{index=17} —
        L_unl_unc = (
            ((z_a_stack.mean(0) - z_tilde)**2).mean()
          + ((z_b_stack.mean(0) - z_tilde)**2).mean()
        )

        # — Total loss (Eq 11) :contentReference[oaicite:18]{index=18}&#8203;:contentReference[oaicite:19]{index=19} —
        loss = L_sup_reg + L_sup_unc + w_unl * (L_unl_reg + L_unl_unc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # — Inference: ensemble means over T dropout runs :contentReference[oaicite:20]{index=20}&#8203;:contentReference[oaicite:21]{index=21} —
    model_a.eval(); model_b.eval()
    preds_list = []
    with torch.no_grad():
        for _ in range(T):
            y_a_u, _ = model_a(X_unl)
            y_b_u, _ = model_b(X_unl)
            preds_list.append((y_a_u + y_b_u) / 2)
        preds = torch.stack(preds_list).mean(0).cpu().numpy()

    return preds, y_unl_np


##############################################
#  Model I: RankUp               #
##############################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class RankUpNet(nn.Module):
    """
    Fully-faithful RankUp network:
      - shared backbone f(x; θ)
      - regression head h(f) → y
      - Auxiliary Ranking Classifier (ARC) g(f) → {0,1}
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # regression head
        self.reg_head = nn.Linear(hidden_dim, out_dim)
        # ARC head
        self.arc_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)
        y_pred = self.reg_head(feat)
        arc_logits = self.arc_head(feat)
        return feat, y_pred, arc_logits


def rankup_regression(
    sup_df,
    inf_df,
    hidden_dim=256,
    lr=1e-3,
    epochs=200,
    batch_size=64,
    alpha_arc=1.0,
    alpha_arc_ulb=1.0,
    alpha_rda=0.1,
    T=0.5,
    tau=0.95,
    ema_m=0.999,
    device=None
):
    """
    Faithful implementation of RankUp (Huang et al. 2024) with:
      - ARC supervised + unsupervised FixMatch-style loss
      - Regression Distribution Alignment (RDA) on unlabeled
      - EMA teacher model for inference

    Args:
      sup_df: DataFrame with 'morph_coordinates', 'gene_coordinates'
      inf_df: DataFrame with 'morph_coordinates', 'gene_coordinates'
    Returns:
      preds_unl (ndarray): predictions for inf_df
      true_unl  (ndarray): ground-truth gene coords for inf_df
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # prepare tensors
    Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32)
    ys = torch.tensor(np.vstack(sup_df['gene_coordinates']),    dtype=torch.float32)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']),   dtype=torch.float32)
    yu = np.vstack(inf_df['gene_coordinates'])

    in_dim  = Xs.size(1)
    out_dim = ys.size(1)

    # DataLoaders
    sup_loader = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=True, drop_last=True)
    unl_loader = DataLoader(TensorDataset(Xu), batch_size=batch_size, shuffle=True, drop_last=True)

    # model + EMA
    model = RankUpNet(in_dim, hidden_dim, out_dim).to(device)
    ema_model = RankUpNet(in_dim, hidden_dim, out_dim).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        sup_iter = iter(sup_loader)
        unl_iter = iter(unl_loader)
        for _ in range(min(len(sup_loader), len(unl_loader))):
            xb, yb = next(sup_iter)
            (xu_w,) = next(unl_iter)
            # augment: weak (identity) and strong (gaussian noise)
            xu_s = xu_w + torch.randn_like(xu_w) * 0.1

            xb, yb, xu_w, xu_s = xb.to(device), yb.to(device), xu_w.to(device), xu_s.to(device)

            # forward sup
            feat_b, pred_b, logit_arc_b = model(xb)
            # compute pairwise ARC targets: torch.sign(y_i - y_j)
            with torch.no_grad():
                y_diff = yb.unsqueeze(0) - yb.unsqueeze(1)
                arc_targets_sup = ((y_diff > 0).long().view(-1)).to(device)

            # sup losses
            loss_reg = F.mse_loss(pred_b, yb)
            # sup ARC logits matrix
            logits_mat_b = (logit_arc_b.unsqueeze(0) - logit_arc_b.unsqueeze(1)).view(-1, 2)
            loss_arc_sup = F.cross_entropy(logits_mat_b, arc_targets_sup)

            # forward unlabeled weak
            _, _, logit_arc_u_w = model(xu_w)
            probs = F.softmax(logit_arc_u_w / T, dim=1)
            maxp, pseudo = probs.max(dim=1)
            mask = (maxp >= tau).float()

            # forward unlabeled strong
            _, _, logit_arc_u_s = model(xu_s)
            loss_arc_unsup = (F.cross_entropy(logit_arc_u_s, pseudo, reduction='none') * mask).mean()

            # RDA on regression outputs (mean & std alignment)
            _, pred_u_w, _ = model(xu_w)
            mu_sup = pred_b.detach().mean(0)
            sigma_sup = pred_b.detach().std(0)
            mu_unl = pred_u_w.mean(0)
            sigma_unl = pred_u_w.std(0)
            loss_rda = ((mu_unl - mu_sup)**2).sum() + ((sigma_unl - sigma_sup)**2).sum()

            # total loss: Eq.6 of RankUp paper
            loss = (loss_reg + alpha_arc * loss_arc_sup
                    + alpha_arc_ulb * loss_arc_unsup
                    + alpha_rda * loss_rda)

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_m).add_(p.data, alpha=1-ema_m)

    # inference with EMA model
    ema_model.eval()
    with torch.no_grad():
        _, preds_unl, _ = ema_model(Xu.to(device))
    return preds_unl.cpu().numpy(), yu


# def _simple_mlp(in_dim, out_dim):
#     return nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(),
#                          nn.Linear(256,128), nn.ReLU(),
#                          nn.Linear(128,out_dim))

# def _pairwise_rank_loss(pred, y):
#     # hinge on pairwise orderings
#     dif_pred = pred[:,None]-pred[None,:]
#     dif_true = y[:,None]-y[None,:]
#     sign_true = np.sign(dif_true)
#     margin = 0.1
#     loss = np.maximum(0, margin - sign_true*dif_pred)
#     return loss.mean()


# def rankup_regression(sup_df, inf_df, lr=1e-3, epochs=200, alpha=0.3):
#     Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32)
#     ys = torch.tensor(np.vstack(sup_df['gene_coordinates']), dtype=torch.float32)
#     Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']), dtype=torch.float32)
#     yu = np.vstack(inf_df['gene_coordinates'])

#     net = _simple_mlp(Xs.size(1), ys.size(1))
#     opt = torch.optim.Adam(net.parameters(), lr=lr)
#     for _ in range(epochs):
#         idx = torch.randperm(Xs.size(0))[:32]
#         xb, yb = Xs[idx], ys[idx]
#         pred_b = net(xb)
#         # basic MSE
#         loss = F.mse_loss(pred_b, yb)
#         # auxiliary ranking on the batch
#         rank_loss = torch.tensor(_pairwise_rank_loss(pred_b.detach().numpy(),
#                                                      yb.detach().numpy()),
#                                  dtype=torch.float32)
#         loss = loss + alpha*rank_loss
#         opt.zero_grad(); loss.backward(); opt.step()

#     with torch.no_grad():
#         preds = net(Xu).numpy()
#     return preds, yu


##################################################
# Model J: AGDN
##################################################


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class ADCConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K=5):
        super().__init__(aggr='add')
        self.K = K
        self.log_t = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, norm = self._normalize(edge_index, x.size(0))
        h = x
        t = torch.exp(self.log_t)
        out = torch.exp(-t) * h
        scale = torch.exp(-t)
        for k in range(1, self.K + 1):
            h = self.propagate(edge_index, x=h, norm=norm)
            scale = scale * t / k
            out = out + scale.view(-1, *([1] * (h.dim() - 1))) * h
        return self.lin(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def _normalize(self, edge_index, num_nodes):
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_index, norm

class AGDN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, K=5, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.last_flags = []
        self.dropouts = []
        self.convs.append(ADCConv(in_channels, hidden_channels, K)); self.last_flags.append(False); self.dropouts.append(dropout)
        for _ in range(num_layers - 2):
            self.convs.append(ADCConv(hidden_channels, hidden_channels, K)); self.last_flags.append(False); self.dropouts.append(dropout)
        self.convs.append(ADCConv(hidden_channels, out_channels, K)); self.last_flags.append(True);  self.dropouts.append(0.0)

    def forward(self, x, edge_index):
        for conv, last, drop in zip(self.convs, self.last_flags, self.dropouts):
            x = conv(x, edge_index)
            if not last:
                x = F.elu(x)
                x = F.dropout(x, p=drop, training=self.training)
        return x


def agdn_regression(supervised_df, inference_df, K=5, hidden=64, num_layers=2, dropout=0.1, epochs=500, lr=1e-3, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    sup_m = np.stack(supervised_df['morph_coordinates'].values)
    sup_g = np.stack(supervised_df['gene_coordinates'].values)
    inf_m = np.stack(inference_df['morph_coordinates'].values)
    inf_g = np.stack(inference_df['gene_coordinates'].values)
    X = torch.tensor(np.vstack([sup_m, inf_m]), dtype=torch.float32, device=device)
    Y_sup = torch.tensor(sup_g, dtype=torch.float32, device=device)
    N_sup = sup_m.shape[0]
    srcs, dsts = [], []
    for i in range(N_sup):
        for j in range(N_sup):
            if i != j:
                srcs.append(i); dsts.append(j)
    for i in range(inf_m.shape[0]):
        u = N_sup + i
        for j in range(N_sup):
            srcs.extend([u, j]); dsts.extend([j, u])
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
    model = AGDN(in_channels=sup_m.shape[1], hidden_channels=hidden, out_channels=sup_g.shape[1], num_layers=num_layers, K=K, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(X, edge_index)
        loss = loss_fn(out[:N_sup], Y_sup)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(X, edge_index)[N_sup:].cpu().numpy()
    return pred, inf_g



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
    mae = mean_absolute_error(predictions, actuals)
    
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

    inference_plus_supervised = pd.concat([inference_samples, supervised_samples], axis=0)

    # Encode gene data
    gene_samples = encode_genes_for_samples(gene_samples, barcode_tokenizer, barcode_model)

    # Perform Bridged Clustering
    image_kmeans, gene_kmeans, _, gene_features, image_clusters, gene_clusters = perform_clustering(
        inference_plus_supervised, gene_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families
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
    knn_predictions, knn_actuals = knn_regression(supervised_samples, inference_samples, n_neighbors=max(1, int(n_samples * supervised)))
    mt_predictions, mt_actuals = mean_teacher_regression(supervised_samples, inference_samples)
    xgb_preds, xgb_actuals = xgboost_regression(supervised_samples, inference_samples)
    lap_preds, lap_actuals = laprls_regression(supervised_samples, inference_samples)
    tsvr_preds, tsvr_actuals = tsvr_regression(supervised_samples, inference_samples)
    tnnr_preds, tnnr_actuals = tnnr_regression(supervised_samples, inference_samples)
    ucv_preds, ucv_actuals = ucvme_regression(supervised_samples, inference_samples)
    rank_preds, rank_actuals = rankup_regression(supervised_samples, inference_samples)
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

    n_families_values = [5]
    n_samples_values = [50]
    supervised_values = [1]
    models = ['BKM', 'KNN', 'Mean Teacher', 'XGBoost', 'Laplacian RLS', 'TSVR', 'TNNR', 'UCVME', 'RankUp', 'AGDN']
    # models = ['BKM', 'XGBoost', 'RankUp', 'AGDN']

    n_trials = 3
    

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
    np.save('results/total_results_matrix_010.npy', results_matrix)
    np.save('results/rs_matrix_009.npy', rs_matrix)
    print("Experiment completed.")
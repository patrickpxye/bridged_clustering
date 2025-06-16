### Script 0: For Bioscan Dataset

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
                       alpha=0.99, w_max=0.1, ramp_len=1):
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


##############################################
# Model D: FixMatch #
##############################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------
#  1) MLP backbone + EMA teacher
# ----------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class FixMatchRegressor:
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=1e-3,
        alpha_ema=0.99,
        lambda_u_max=1.0,
        rampup_length=10,
        conf_threshold=0.1,   # threshold on std of pseudo-labels
        device=None
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Student and EMA teacher
        self.student = MLPRegressor(input_dim, output_dim).to(device)
        self.teacher = MLPRegressor(input_dim, output_dim).to(device)
        self._update_teacher(ema_decay=0.0)  # initialize teacher = student

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.ema_decay = alpha_ema

        # Unsupervised weight schedule
        self.lambda_u_max = lambda_u_max
        self.rampup_length = rampup_length

        # Confidence threshold (we’ll measure std of multiple weak preds)
        self.conf_threshold = conf_threshold

        # MSE criterion
        self.mse_loss = nn.MSELoss(reduction="none")

    @torch.no_grad()
    def _update_teacher(self, ema_decay=None):
        """EMA update: teacher_params = ema_decay * teacher + (1-ema_decay) * student"""
        decay = self.ema_decay if ema_decay is None else ema_decay
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=(1.0 - decay))

    def _get_lambda_u(self, current_epoch):
        """Linear ramp-up from 0 -> lambda_u_max over rampup_length epochs"""
        if current_epoch >= self.rampup_length:
            return self.lambda_u_max
        else:
            return self.lambda_u_max * (current_epoch / self.rampup_length)

    def _augment(self, x, strong=False):
        """
        Return an augmented version of x.
        - strong: heavier noise + random dimension dropout.
        """
        # 1) Additive Gaussian noise (relative to feature scale)
        noise_scale = 0.05 if not strong else 0.2
        x_noisy = x + torch.randn_like(x) * noise_scale

        # 2) Multiplicative jitter (small random scale per-dimension)
        if strong:
            scale = 1.0 + 0.1 * torch.randn_like(x)
            x_noisy = x_noisy * scale

        # 3) Random dimension dropout (only for strong)
        if strong:
            # randomly zero out 10% of dimensions
            mask = (torch.rand_like(x) > 0.1).float()
            x_noisy = x_noisy * mask

        return x_noisy

    def train(
        self,
        sup_loader: DataLoader,
        unl_loader: DataLoader,
        epochs: int = 200
    ):
        """
        Train student+teacher for `epochs` epochs.
        sup_loader yields (x_sup, y_sup)
        unl_loader yields (x_unl, dummy)  [we ignore dummy]
        """
        self.student.train()
        self.teacher.train()

        min_batches = min(len(sup_loader), len(unl_loader))

        for epoch in range(epochs):
            lambda_u = self._get_lambda_u(epoch)
            epoch_losses = {"sup": 0.0, "unsup": 0.0}

            sup_iter = iter(sup_loader)
            unl_iter = iter(unl_loader)

            for _ in range(min_batches):
                # --- 1) Fetch one sup batch and one unl batch
                x_sup, y_sup = next(sup_iter)
                x_unl, _ = next(unl_iter)

                x_sup = x_sup.float().to(self.device)
                y_sup = y_sup.float().to(self.device)
                x_unl = x_unl.float().to(self.device)

                # --- 2) Supervised forward
                preds_sup = self.student(x_sup)                    # (B, out_dim)
                loss_sup = F.mse_loss(preds_sup, y_sup, reduction="mean")

                # --- 3) Unlabeled: generate multiple weak views for confidence
                # We’ll do two weak augmentations per sample.
                x_unl_w1 = self._augment(x_unl, strong=False)
                x_unl_w2 = self._augment(x_unl, strong=False)

                with torch.no_grad():
                    # Teacher produces pseudo-labels
                    p_w1 = self.teacher(x_unl_w1)  # (B, out_dim)
                    p_w2 = self.teacher(x_unl_w2)  # (B, out_dim)

                    # Estimate “confidence” by the two weak preds’ difference
                    pseudo_label = 0.5 * (p_w1 + p_w2)  # average as final pseudo
                    std_weak = (p_w1 - p_w2).pow(2).mean(dim=1).sqrt()  # (B,) L2‐std

                # Mask = 1 if std_weak < threshold, else 0
                mask = (std_weak < self.conf_threshold).float().unsqueeze(1)  # (B,1)

                # --- 4) Strong aug on unlabeled
                x_unl_s = self._augment(x_unl, strong=True)
                preds_s = self.student(x_unl_s)  # (B, out_dim)

                # --- 5) Unsupervised consistency loss (only for “confident” samples)
                # We compute MSE per-sample, then multiply by mask
                loss_unsup_per_sample = self.mse_loss(preds_s, pseudo_label)  # (B, out_dim)
                # average over output_dim, then multiply by mask
                loss_unsup = (loss_unsup_per_sample.mean(dim=1) * mask.squeeze(1)).mean()

                # --- 6) Total loss
                loss = loss_sup + lambda_u * loss_unsup

                # --- 7) Backprop & update student
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # --- 8) EMA update for teacher
                self._update_teacher()

                epoch_losses["sup"] += loss_sup.item()
                epoch_losses["unsup"] += loss_unsup.item()

            # (Optional) print or log epoch stats:
            #  avg_sup = epoch_losses["sup"] / min_batches
            #  avg_unsup = epoch_losses["unsup"] / min_batches
            #  print(f"Epoch {epoch:03d} | L_sup={avg_sup:.4f} | L_unsup={avg_unsup:.4f} | λ_u={lambda_u:.4f}")

    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        """Use the EMA teacher to predict on new data."""
        self.teacher.eval()
        X = X.float().to(self.device)
        return self.teacher(X).cpu().numpy()

# --------------------------------------------
#  2) FixMatch training wrapper
# --------------------------------------------
def fixmatch_regression(
    supervised_df,
    inference_df,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    alpha_ema=0.99,
    lambda_u_max=1.0,
    rampup_length=10,
    conf_threshold=0.1
):
    """
    A stronger FixMatch‐style regressor:
      - Separate EMA teacher to generate pseudo‐labels
      - Confidence masking via two weak‐augment predictions
      - Richer augmentations (additive noise + multiplicative jitter + dropout)
      - Linear ramp‐up of unsupervised weight λ_u
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Prepare numpy arrays
    X_sup = np.vstack(supervised_df["morph_coordinates"])
    y_sup = np.vstack(supervised_df["gene_coordinates"])
    X_unl = np.vstack(inference_df["morph_coordinates"])
    y_unl_true = np.vstack(inference_df["gene_coordinates"])

    input_dim = X_sup.shape[1]
    output_dim = y_sup.shape[1]

    # 2) Build datasets & loaders
    sup_dataset = TensorDataset(
        torch.tensor(X_sup, dtype=torch.float32),
        torch.tensor(y_sup, dtype=torch.float32)
    )
    unl_dataset = TensorDataset(
        torch.tensor(X_unl, dtype=torch.float32),
        torch.zeros((len(X_unl), output_dim), dtype=torch.float32)  # dummy
    )
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    unl_loader = DataLoader(unl_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3) Initialize FixMatchRegressor
    fixmatch = FixMatchRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        lr=lr,
        alpha_ema=alpha_ema,
        lambda_u_max=lambda_u_max,
        rampup_length=rampup_length,
        conf_threshold=conf_threshold,
        device=device
    )

    # 4) Train
    fixmatch.train(sup_loader, unl_loader, epochs=epochs)

    # 5) Final inference on unlabeled
    X_unl_tensor = torch.tensor(X_unl, dtype=torch.float32)
    preds_unl = fixmatch.predict(X_unl_tensor)

    return preds_unl, y_unl_true

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
    Fully‐faithful RankUp network:
      - backbone: f(x; θ) → feat ∈ ℝ^h
      - regression head: h(feat) → y ∈ ℝ^d
      - ARC head: g(feat) → logits ∈ ℝ^2
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.reg_head = nn.Linear(hidden_dim, out_dim)
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
    lr=1e-3,                 # try reducing if batch_size is large 
    epochs=200,
    batch_size=128,
    alpha_arc=1.0,
    alpha_arc_ulb=1.0,
    alpha_rda=0.05,          # RDA weight
    T=0.7,                   # temperature for softmax
    tau=0.90,                # confidence threshold
    ema_m=0.999,
    device=None
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    Xs = torch.tensor(np.vstack(sup_df['morph_coordinates']), dtype=torch.float32).to(device)
    ys = torch.tensor(np.vstack(sup_df['gene_coordinates']),    dtype=torch.float32).to(device)
    Xu = torch.tensor(np.vstack(inf_df['morph_coordinates']),   dtype=torch.float32).to(device)
    yu = np.vstack(inf_df['gene_coordinates'])  # we'll evaluate MSE on this eventually

    in_dim  = Xs.size(1)
    out_dim = ys.size(1)

    # Precompute supervised mean and std for RDA
    sup_mu = ys.mean(0)            # shape (d,)
    sup_sigma = ys.std(0)          # shape (d,)

    sup_loader = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=True,  drop_last=True)
    unl_loader = DataLoader(TensorDataset(Xu),       batch_size=batch_size, shuffle=True,  drop_last=True)

    model = RankUpNet(in_dim, hidden_dim, out_dim).to(device)
    ema_model = RankUpNet(in_dim, hidden_dim, out_dim).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        sup_iter = iter(sup_loader)
        unl_iter = iter(unl_loader)

        if epoch < 10:
            curr_alpha_arc_ulb = 0.0
        else:
            ramp = min((epoch - 10) / 40.0, 1.0)
            curr_alpha_arc_ulb = ramp * alpha_arc_ulb

        for _ in range(min(len(sup_loader), len(unl_loader))):
            xb, yb = next(sup_iter)
            (xu_w,) = next(unl_iter)

            # Summarize yb (d‐dim) into a scalar for ranking:
            # Option A: mean over dimensions
            yb_scalar = yb.mean(dim=1, keepdim=True)  # shape (B, 1)
            # Option B: L2‐norm → uncomment the next line instead:
            # yb_scalar = yb.norm(dim=1, keepdim=True)  # shape (B, 1)

            xb = xb.to(device); yb = yb.to(device)
            xu_w = xu_w.to(device)

            # --- Strong augmentation on xu_w
            noise = torch.randn_like(xu_w) * 0.2
            scale = 1.0 + 0.05 * torch.randn_like(xu_w)
            xu_s = xu_w * scale + noise

            xb, yb_scalar, xu_w, xu_s = xb.to(device), yb_scalar.to(device), xu_w.to(device), xu_s.to(device)

            # ==== Supervised portion ====
            feat_b, pred_b, logit_arc_b = model(xb)
            # (1) Regression loss on actual d‐dim targets
            loss_reg = F.mse_loss(pred_b, yb)

            # (2) Supervised ARC: build all B×B pairs
            y_diff = yb_scalar.unsqueeze(0) - yb_scalar.unsqueeze(1)   # shape (B, B, 1)
            arc_targets_sup = (y_diff.squeeze(-1) > 0).long().view(-1)  # (B*B,)
            logits_mat_b = (logit_arc_b.unsqueeze(0) - logit_arc_b.unsqueeze(1)).view(-1, 2)  # (B*B, 2)
            loss_arc_sup = F.cross_entropy(logits_mat_b, arc_targets_sup)

            # ==== Unlabeled (Weak) portion with EMA teacher ====
            with torch.no_grad():
                _, _, logit_arc_u_w = ema_model(xu_w)  # shape (B, 2)
            logits_mat_u_w = (logit_arc_u_w.unsqueeze(0) - logit_arc_u_w.unsqueeze(1)).view(-1, 2)  # (B*B, 2)
            probs_mat_u_w = F.softmax(logits_mat_u_w / T, dim=1)  # (B*B, 2)
            maxp_mat, pseudo_mat = probs_mat_u_w.max(dim=1)       # (B*B,)
            mask_mat = (maxp_mat >= tau).float()                  # (B*B,)

            # ==== Unlabeled (Strong) portion with student ====
            _, _, logit_arc_u_s = model(xu_s)  # shape (B, 2)
            logits_mat_u_s = (logit_arc_u_s.unsqueeze(0) - logit_arc_u_s.unsqueeze(1)).view(-1, 2)
            loss_arc_unsup = (F.cross_entropy(logits_mat_u_s, pseudo_mat, reduction='none') * mask_mat).mean()

            # ==== RDA: match student’s unlabeled moments to sup_mu/sup_sigma ====
            _, pred_u_w_student, _ = model(xu_w)  # (B, d)
            mu_unl = pred_u_w_student.mean(0)     # (d,)
            sigma_unl = pred_u_w_student.std(0)   # (d,)
            loss_rda = ((mu_unl - sup_mu)**2).sum() + ((sigma_unl - sup_sigma)**2).sum()

            # ==== Combine losses ====
            loss = (loss_reg
                    + alpha_arc * loss_arc_sup
                    + curr_alpha_arc_ulb * loss_arc_unsup
                    + alpha_rda * loss_rda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_m).add_(p.data, alpha=(1.0 - ema_m))

        # Optional: print epoch‐level losses here

    # ==== Final inference on unlabeled via EMA teacher ====
    ema_model.eval()
    with torch.no_grad():
        _, preds_unl, _ = ema_model(Xu)  # shape (n_unlabeled, d)
    return preds_unl.cpu().numpy(), yu


###################################
#  New Baseline: Basic GCN        #
###################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

class GCN(nn.Module):
    """
    Simple 2-layer GCN for regression:
      - conv1: GCNConv(in_channels → hidden_channels)
      - conv2: GCNConv(hidden_channels → out_channels)
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor):
        """
        x: [N, in_channels]       node features
        edge_index: [2, E]        adjacency
        Returns: [N, out_channels] node regression outputs
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def gcn_regression(
    supervised_df,
    inference_df,
    hidden=64,
    dropout=0.1,
    epochs=200,
    lr=1e-3,
    device=None
):
    """
    Train a basic GCN on supervised nodes and predict on inference nodes.
    supervised_df: DataFrame with 'morph_coordinates', 'gene_coordinates'
    inference_df:  DataFrame with 'morph_coordinates', 'gene_coordinates'
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # Stack supervised and inference features
    sup_m = np.vstack(supervised_df['morph_coordinates'].values)
    sup_g = np.vstack(supervised_df['gene_coordinates'].values)
    inf_m = np.vstack(inference_df['morph_coordinates'].values)
    inf_g = np.vstack(inference_df['gene_coordinates'].values)

    # Create node feature tensor and label tensor
    X = torch.tensor(np.vstack([sup_m, inf_m]), dtype=torch.float32, device=device)
    Y_sup = torch.tensor(sup_g, dtype=torch.float32, device=device)
    N_sup = sup_m.shape[0]
    N_inf = inf_m.shape[0]

    # Build edge_index: same scheme as ADC/AGDN (fully connect sup↔sup and sup↔inf)
    srcs, dsts = [], []
    # Sup→Sup edges
    for i in range(N_sup):
        for j in range(N_sup):
            if i != j:
                srcs.append(i); dsts.append(j)
    # Inf↔Sup edges
    for i in range(N_inf):
        u = N_sup + i
        for j in range(N_sup):
            srcs.extend([u, j]); dsts.extend([j, u])
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)

    # Instantiate GCN
    in_dim = sup_m.shape[1]
    out_dim = sup_g.shape[1]
    model = GCN(in_channels=in_dim, hidden_channels=hidden, out_channels=out_dim, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(X, edge_index)        # [N_sup+N_inf, out_dim]
        loss = loss_fn(out[:N_sup], Y_sup)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X, edge_index)[N_sup:].cpu().numpy()  # predictions on inference nodes
    return pred, inf_g


###################################
# New Models for Unmatched Regression #
####################################


import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def monotone_projection_regression(image_df, gene_df, inference_df):
    """
    A “single‐projection + quantile‐matching” baseline for unmatched regression.
    
    Steps:
      1) Let X = [x_i] from image_df['morph_coordinates']  (shape: (N_img, d))
         Let Y = [y_j] from gene_df['gene_coordinates']     (shape: (N_gene, d))
      2) Compute first‐PC direction of X; call the projection values px_i = u_X^T x_i.
         Compute first‐PC direction of Y; call py_j = u_Y^T y_j.
      3) Sort X’s by px, sort Y’s by py.  Create a “pseudo‐paired” list:
            pseudo_X_k  ←  k‐th smallest X (w.r.t. px)
            pseudo_Y_k  ←  k‐th smallest Y (w.r.t. py)
      4) Train a multivariate regressor  R: R( x ) ≈ y  on the pseudo‐pairs { (pseudo_X_k, pseudo_Y_k) }.
      5) Return  R( inference_X )  for each inference_X in inference_df['morph_coordinates'].  
    """
    # 1) Extract data
    X_img = np.vstack(image_df['morph_coordinates'].values)   # shape: (N_img, d)
    Y_gene = np.vstack(gene_df['gene_coordinates'].values)    # shape: (N_gene, d)
    X_inf = np.vstack(inference_df['morph_coordinates'].values)  # (N_inf, d)

    # 2) Fit PCA on each, take first component (1D)
    pca_X = PCA(n_components=1).fit(X_img)
    pca_Y = PCA(n_components=1).fit(Y_gene)
    proj_X = pca_X.transform(X_img).squeeze()   # (N_img,)
    proj_Y = pca_Y.transform(Y_gene).squeeze()  # (N_gene,)

    # 3) Sort indices by projection
    idx_X_sorted = np.argsort(proj_X)  # indices of X in increasing order
    idx_Y_sorted = np.argsort(proj_Y)

    # Keep only min(N_img, N_gene) pairs
    N_pair = min(len(idx_X_sorted), len(idx_Y_sorted))
    idx_X_paired = idx_X_sorted[:N_pair]
    idx_Y_paired = idx_Y_sorted[:N_pair]

    X_pseudo = X_img[idx_X_paired]  # shape: (N_pair, d)
    Y_pseudo = Y_gene[idx_Y_paired] # shape: (N_pair, d)

    # 4) Fit a multivariate linear regressor on those pseudo‐pairs
    linreg = LinearRegression().fit(X_pseudo, Y_pseudo)

    # 5) Make predictions on the true inference set
    Y_pred_inf = linreg.predict(X_inf)  # (N_inf, d)
    Y_true_inf = Y_gene[: len(X_inf) ] if len(Y_gene) >= len(X_inf) else None
    # Note: We do have ground‐truth gene‐vectors for inference_df, so:
    Y_true_inf = np.vstack(inference_df['gene_coordinates'].values)

    return Y_pred_inf, Y_true_inf



from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

def kernel_mean_matching_regression(image_df, gene_df, inference_df, lam=1e-2, sigma=None):
    """
    A “kernel‐mean matching” baseline via RBF‐KRR on pseudo‐paired unordered samples.
    
    1) Let X = [x_i] from image_df['morph_coordinates'],  Y = [y_i] from gene_df['gene_coordinates'].
    2) Standardize X and Y (zero‐mean, unit‐var per dimension).
    3) If sigma is None, set σ = median pairwise distance on X.
    4) Fit KernelRidge (RBF) on {(X_i, Y_i)}_{i=1}^N as if “ordered” (even though they are not).
       Because α is large (lam small), this will try to match means instead of exact pairs.
    5) Predict on inference_df['morph_coordinates'].  
    """
    # 1) Stack data
    X_img = np.vstack(image_df['morph_coordinates'].values)   # (N_img, d)
    Y_gene = np.vstack(gene_df['gene_coordinates'].values)    # (N_gene, d)
    X_inf = np.vstack(inference_df['morph_coordinates'].values)  # (N_inf, d)

    # Truncate to min(N_img, N_gene, N_inf) if shapes differ
    N_pair = min(len(X_img), len(Y_gene))
    X_img = X_img[:N_pair]
    Y_gene = Y_gene[:N_pair]

    # 2) Standardize each space separately
    scaler_X = StandardScaler().fit(X_img)
    scaler_Y = StandardScaler().fit(Y_gene)
    Xs = scaler_X.transform(X_img)   # shape: (N_pair, d)
    Ys = scaler_Y.transform(Y_gene)  # shape: (N_pair, d)

    # 3) Choose sigma by median‐pairwise on Xs if not given
    if sigma is None:
        # compute pairwise Euclidean distances on a random subset (for speed)
        sample_idx = np.random.choice(N_pair, min(N_pair, 200), replace=False)
        sub = Xs[sample_idx]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :])**2, axis=2).ravel()
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med)
    gamma = 1.0 / (2 * sigma**2)

    # 4) Fit a KernelRidge regressor: input = Xs, target = Ys (multi‐output)
    #    We wrap two calls: one KRR per dimension of Y
    preds = []
    for dim in range(Ys.shape[1]):
        kr = KernelRidge(kernel='rbf', alpha=lam, gamma=gamma)
        kr.fit(Xs, Ys[:, dim])
        preds.append(kr.predict(scaler_X.transform(np.vstack(inference_df['morph_coordinates']))))
    # preds is a list of (N_inf,) arrays; stack into (N_inf, d)
    Y_pred_inf = np.vstack(preds).T

    # 5) Un‐standardize
    Y_pred_inf = scaler_Y.inverse_transform(Y_pred_inf)
    Y_true_inf = np.vstack(inference_df['gene_coordinates'].values)

    return Y_pred_inf, Y_true_inf



import numpy as np

def deconvolution_linear_regression(image_df, gene_df, inference_df, eps=1e-6):
    """
    A “covariance‐matching / Wiener‐deconvolution” surrogate (Azadkia & Balabdaoui, 2024),
    modified so that it works even when d_X != d_Y.

    Steps:
      1) Let X = [x_i] from image_df['morph_coordinates']    (shape: N_pair × d_X)
         Let Y = [y_j] from gene_df['gene_coordinates']     (shape: N_pair × d_Y)
         (If len(X_img) != len(Y_gene), truncate both to N_pair = min(N_img, N_gene).)
      2) Compute Σ_XX = (1/N_pair) · X^T X  ∈ ℝ^{d_X×d_X},  Σ_YY = (1/N_pair) · Y^T Y  ∈ ℝ^{d_Y×d_Y}.
         Add eps·I_{d_X} to Σ_XX, and eps·I_{d_Y} to Σ_YY for numerical stability.
      3) Eigendecompose:
         Σ_XX = U_X Λ_X U_X^T,   Σ_YY = U_Y Λ_Y U_Y^T.
      4) Let r = min(d_X, d_Y).  Keep the top‐r eigenpairs:
         U_{X,r} = U_X[:, :r]   (shape: d_X×r),   Λ_{X,r} = diag(λ_X[:r])   (shape: r×r),
         U_{Y,r} = U_Y[:, :r]   (shape: d_Y×r),   Λ_{Y,r} = diag(λ_Y[:r])   (shape: r×r).
      5) Form
         Λ_{X,r}^{-1/2} = diag(1/√(λ_X[:r]))  (r×r),
         Λ_{Y,r}^{1/2}  = diag(√(λ_Y[:r]))     (r×r).
      6) Set
         A_est = U_{Y,r} · Λ_{Y,r}^{1/2} · Λ_{X,r}^{-1/2} · U_{X,r}^T   ∈ ℝ^{d_Y×d_X}.
      7) For each x ∈ inference_df (a d_X‐vector), predict ŷ = A_est · x ∈ ℝ^{d_Y}.
         Return (all predictions, all true gene‐vectors).
    """
    # 1) Stack / truncate to N_pair
    X_img = np.vstack(image_df['morph_coordinates'].values)   # shape: (N_img, d_X)
    Y_gene = np.vstack(gene_df['gene_coordinates'].values)    # shape: (N_gene, d_Y)
    N_pair = min(len(X_img), len(Y_gene))

    X = X_img[:N_pair]   # (N_pair × d_X)
    Y = Y_gene[:N_pair]  # (N_pair × d_Y)

    d_X = X.shape[1]
    d_Y = Y.shape[1]

    # 2) Compute covariance‐like matrices (with eps·I)
    Sigma_XX = (X.T @ X) / N_pair + eps * np.eye(d_X)   # (d_X × d_X)
    Sigma_YY = (Y.T @ Y) / N_pair + eps * np.eye(d_Y)   # (d_Y × d_Y)

    # 3) Eigendecompose each
    λ_x, U_x = np.linalg.eigh(Sigma_XX)   # λ_x: (d_X,), U_x: (d_X × d_X)
    λ_y, U_y = np.linalg.eigh(Sigma_YY)   # λ_y: (d_Y,), U_y: (d_Y × d_Y)

    # 4) Keep top‐r eigenpairs, where r = min(d_X, d_Y)
    r = min(d_X, d_Y)
    # Sort eigenvalues in descending order, then select top r
    idx_x_desc = np.argsort(λ_x)[::-1]
    idx_y_desc = np.argsort(λ_y)[::-1]

    idx_x_r = idx_x_desc[:r]   # indices of top‐r λ_x
    idx_y_r = idx_y_desc[:r]   # indices of top‐r λ_y

    # U_{X,r} : (d_X × r),   Λ_{X,r} : (r,)
    U_x_r = U_x[:, idx_x_r]   # (d_X × r)
    Λ_x_r = λ_x[idx_x_r]      # (r,)

    # U_{Y,r} : (d_Y × r),   Λ_{Y,r} : (r,)
    U_y_r = U_y[:, idx_y_r]   # (d_Y × r)
    Λ_y_r = λ_y[idx_y_r]      # (r,)

    # 5) Form diagonal sqrt/inv‐sqrt matrices (each r×r)
    inv_sqrt_Λx_r = np.diag(1.0 / np.sqrt(Λ_x_r))   # (r×r)
    sqrt_Λy_r     = np.diag(np.sqrt(Λ_y_r))         # (r×r)

    # 6) Build A_est = U_{Y,r} · sqrt(Λ_{Y,r}) · inv_sqrt(Λ_{X,r}) · U_{X,r}^T
    #    Shapes: U_{Y,r} (d_Y×r), sqrt_Λy_r (r×r), inv_sqrt_Λx_r (r×r), U_{X,r}^T (r×d_X)
    A_est = U_y_r @ sqrt_Λy_r @ inv_sqrt_Λx_r @ U_x_r.T   # shape: (d_Y × d_X)

    # 7) Predict on the inference set
    X_inf = np.vstack(inference_df['morph_coordinates'].values)  # shape: (N_inf, d_X)
    Y_pred_inf = (A_est @ X_inf.T).T                               # shape: (N_inf, d_Y)
    Y_true_inf = np.vstack(inference_df['gene_coordinates'].values)  # shape: (N_inf, d_Y)

    return Y_pred_inf, Y_true_inf




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
    print("starting mean teacher")
    mt_predictions, mt_actuals = mean_teacher_regression(supervised_samples, inference_samples)
    print("starting fixmatch")
    fixmatch_predictions, fixmatch_actuals = fixmatch_regression(supervised_samples, inference_samples)
    print("starting laprls")
    lap_preds, lap_actuals = laprls_regression(supervised_samples, inference_samples)
    print("starting tsvr")
    tsvr_preds, tsvr_actuals = tsvr_regression(supervised_samples, inference_samples)
    print("starting tnnr")
    tnnr_preds, tnnr_actuals = tnnr_regression(supervised_samples, inference_samples)
    print("starting ucvme")
    ucv_preds, ucv_actuals = ucvme_regression(supervised_samples, inference_samples)
    print("starting rankup")
    rank_preds, rank_actuals = rankup_regression(supervised_samples, inference_samples)
    print("starting gcn")
    gcn_preds, gcn_actuals = gcn_regression(supervised_samples, inference_samples)
    print("starting monotone‐projection baseline")
    mono_preds, mono_actuals = monotone_projection_regression(image_df=inference_plus_supervised,
                                                            gene_df=gene_plus_supervised, 
                                                            inference_df=inference_samples)
    print("starting kernel mean matching baseline")
    kmm_preds, kmm_actuals = kernel_mean_matching_regression(
        image_df= inference_plus_supervised,  # inference samples with image clusters
        gene_df=gene_plus_supervised,    # same N but “as if” paired
        inference_df=inference_samples,
        lam=1e-2
    )
    print("starting deconvolution linear regression baseline")
    deco_preds, deco_actuals = deconvolution_linear_regression(
        image_df= inference_plus_supervised,  # inference samples with image clusters
        gene_df=gene_plus_supervised, 
        inference_df=inference_samples
    )

    


    # Compute errors
    bkm_error, bkm_r2 = evaluate_loss(bkm_predictions, bkm_actuals)
    knn_error, knn_r2 = evaluate_loss(knn_predictions, knn_actuals)
    mean_teacher_error, mean_teacher_r2 = evaluate_loss(mt_predictions, mt_actuals)
    fixmatch_error, fixmatch_r2 = evaluate_loss(fixmatch_predictions, fixmatch_actuals)
    lap_error, lap_r2 = evaluate_loss(lap_preds, lap_actuals)
    tsvr_error, tsvr_r2 = evaluate_loss(tsvr_preds, tsvr_actuals)
    tnnr_error, tnnr_r2 = evaluate_loss(tnnr_preds, tnnr_actuals)
    ucv_error, ucv_r2 = evaluate_loss(ucv_preds, ucv_actuals)
    rank_error, rank_r2 = evaluate_loss(rank_preds, rank_actuals)
    gcn_error, gcn_r2 = evaluate_loss(gcn_preds, gcn_actuals)
    mono_error, mono_r2 = evaluate_loss(mono_preds, mono_actuals)
    kmm_error, kmm_r2 = evaluate_loss(kmm_preds, kmm_actuals)
    deco_error, deco_r2 = evaluate_loss(deco_preds, deco_actuals)

    # Print results
    print(f"Bridged Clustering Error: {bkm_error}")
    print(f"KNN Error: {knn_error}")
    print(f"Mean Teacher Error: {mean_teacher_error}")
    print(f"FixMatch Error: {fixmatch_error}")
    print(f"Laplacian RLS Error: {lap_error}")
    print(f"TSVR Error: {tsvr_error}")
    print(f"TNNR Error: {tnnr_error}")
    print(f"UCVME Error: {ucv_error}")
    print(f"RankUp Error: {rank_error}")
    print(f"GCN Error: {gcn_error}")
    print(f"Monotone Projection Error: {mono_error}")
    print(f"Kernel Mean Matching Error: {kmm_error}")
    print(f"Deconvolution Linear Regression Error: {deco_error}")
    # Store results in a dictionary

    errors = {
        'BKM': bkm_error,
        'KNN': knn_error,
        'Mean Teacher': mean_teacher_error,
        'FixMatch': fixmatch_error,
        'Laplacian RLS': lap_error,
        'TSVR': tsvr_error,
        'TNNR': tnnr_error,
        'UCVME': ucv_error,
        'RankUp': rank_error,
        'GCN': gcn_error,
        'Monotone Projection': mono_error,
        'Kernel Mean Matching': kmm_error,
        'Deconvolution Linear Regression': deco_error
    }

    rs = {
        'BKM': bkm_r2,
        'KNN': knn_r2,
        'Mean Teacher': mean_teacher_r2,
        'FixMatch': fixmatch_r2,
        'Laplacian RLS': lap_r2,
        'TSVR': tsvr_r2,
        'TNNR': tnnr_r2,
        'UCVME': ucv_r2,
        'RankUp': rank_r2,
        'GCN': gcn_r2,
        'Monotone Projection': mono_r2,
        'Kernel Mean Matching': kmm_r2,
        'Deconvolution Linear Regression': deco_r2
    }
    

    return errors, rs, ami_image, ami_gene, decision_accuracy



if __name__ == '__main__':
    csv_path = 'data/bioscan_data.csv'
    image_folder = 'bioscan_images'

    experiment_key = "039"

    n_families_values = [3, 5, 7]
    n_samples_values = [100]
    supervised_values = [0.01, 0.02, 0.04, 0.08]
    out_only_values = [0.2]
    models = ['BKM', 'KNN', 'Mean Teacher', 'FixMatch', 'Laplacian RLS', 'TSVR', 'TNNR', 'UCVME', 'RankUp','GCN', 'Monotone Projection', 'Kernel Mean Matching', 'Deconvolution Linear Regression']

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
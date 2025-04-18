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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.optim import Adam


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
    n_supervised = max(n_supervised, 1)
    # n_supervised = 1
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
# GCN and AGDN #
##################################
class AGDNConv(MessagePassing):
    """
    An Adaptive Graph Diffusion Network layer.
    Each layer performs multi-hop diffusion with either:
      (A) Hop-wise Attention (HA), or
      (B) Hop-wise Convolution (HC).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 K=3,
                 aggr_mode='attention',  # 'attention' or 'convolution'
                 bias=True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K  # Diffusion depth

        # Main linear transform (applied to the 0-hop)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # Optional residual transform
        self.res_lin = nn.Linear(in_channels, out_channels, bias=False)

        # If you want to add a bias after the diffusion:
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Two possible modes for adaptive weighting:
        #  1) Hop-wise attention: (K+1)-sized attention for each node
        #  2) Hop-wise convolution: (K+1)-sized learnable kernel for each channel
        self.aggr_mode = aggr_mode
        if self.aggr_mode == 'attention':
            # Single learnable vector for query
            # dimension = 2*out_channels, since we compare hop0 vs hopK embeddings
            # but in practice we can keep it simpler:
            self.att_query = nn.Parameter(torch.Tensor(2 * out_channels))
        elif self.aggr_mode == 'convolution':
            # Learnable kernel of size (K+1, out_channels).
            self.hop_kernel = nn.Parameter(torch.Tensor(K+1, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.res_lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.aggr_mode == 'attention':
            nn.init.zeros_(self.att_query)   # or xavier, up to you
        elif self.aggr_mode == 'convolution':
            nn.init.xavier_uniform_(self.hop_kernel)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [N, in_channels] node features
        edge_index: [2, E] graph edges
        edge_weight: [E], if your adjacency is weighted
        Returns: [N, out_channels]
        """
        # 0-hop transform
        x0 = self.lin(x)  # shape: [N, out_channels]

        # Generate multi-hop embeddings: h^0, h^1, h^2, ... h^K
        # h^k = A @ h^(k-1)
        # We store them in a list, for subsequent combination:
        multi_hop_list = [x0]  # k=0
        cur_x = x0
        for k in range(1, self.K+1):
            cur_x = self.propagate(edge_index, x=cur_x, edge_weight=edge_weight)
            multi_hop_list.append(cur_x)

        # Summation with learned weighting
        #  Option A: Hop-wise Attention
        #  Option B: Hop-wise Convolution
        if self.aggr_mode == 'attention':
            # We produce attention weights for each hop, node i
            # For hop k, we compare [h^0_i || h^k_i] with self.att_query
            # Then do a softmax across k.  
            # shape of multi_hop_list: (K+1) items each [N, out_channels]
            # We'll stack them into a single [N, (K+1), out_channels].
            # Then compute attention scores.

            # Stack all hops: shape => [N, K+1, out_channels]
            hstack = torch.stack(multi_hop_list, dim=1)
            # Build "reference" h^0 repeated across each hop
            # shape => [N, K+1, out_channels]
            ref = hstack[:, 0:1, :].expand(-1, self.K+1, -1)

            # Then for each hop k: concat [ref_i, hstack_i,k] => dim out_channels*2
            # Then do inner product with self.att_query
            # shape => [N, K+1, 2*out_channels]
            cat_r = torch.cat([ref, hstack], dim=-1)
            
            # shape => [1, 1, 2*out_channels]
            q = self.att_query.view(1, 1, -1)
            # scores => [N, K+1]
            scores = torch.einsum('nkh,nkh->nk',
                                  [cat_r, q.expand(cat_r.shape[0], cat_r.shape[1], -1)])
            # apply activation => e.g. LeakyReLU
            scores = F.leaky_relu(scores, negative_slope=0.2)
            alpha = F.softmax(scores, dim=1)  # softmax across hops

            # Weighted sum across hops
            # alpha => [N, K+1], hstack => [N, K+1, out_channels]
            alpha = alpha.unsqueeze(-1)  # => [N, K+1, 1]
            out = torch.sum(alpha * hstack, dim=1)  # => [N, out_channels]

        elif self.aggr_mode == 'convolution':
            # For each channel c, we have a kernel of length (K+1).
            # multi_hop_list[k] shape => [N, out_channels]
            # hop_kernel => [K+1, out_channels]
            # So out[i, c] = sum_{k=0..K} [ multi_hop_list[k][i, c] * hop_kernel[k, c] ]
            # We'll combine them easily by stacking.

            # shape => [N, K+1, out_channels]
            hstack = torch.stack(multi_hop_list, dim=1)
            # shape => [K+1, out_channels] => broadcast to [N, K+1, out_channels]
            kernel = self.hop_kernel.unsqueeze(0)  # shape => [1, K+1, out_channels]
            # elementwise product => [N, K+1, out_channels], then sum over dimension=1
            out = (hstack * kernel).sum(dim=1)

        # Add residual linear connection from x
        out = out + self.res_lin(x)
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight):
        """
        x_j: neighbor features from j->i
        edge_weight: optional edge weights
        The aggregator is 'add', so we'll sum over messages after message() returns.
        """
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

class AGDN(nn.Module):
    """
    A full multi-layer AGDN, stacking AGDNConv with optional MLP heads.
    """
    def __init__(self,
                 num_features,     # dimension of input x
                 hidden_dim,
                 out_dim,
                 num_layers=2,
                 K=3,
                 aggr_mode='attention'):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.K = K
        self.aggr_mode = aggr_mode

        # Input layer
        self.convs.append(AGDNConv(num_features, hidden_dim, K=self.K, aggr_mode=self.aggr_mode))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(AGDNConv(hidden_dim, hidden_dim, K=self.K, aggr_mode=self.aggr_mode))
        # Output layer
        if num_layers > 1:
            self.convs.append(AGDNConv(hidden_dim, out_dim, K=self.K, aggr_mode=self.aggr_mode))
        else:
            # If only 1 layer, the first conv is also output conv
            self.convs[0] = AGDNConv(num_features, out_dim, K=self.K, aggr_mode=self.aggr_mode)

    def forward(self, data):
        """
        data: a PyG Data object: data.x, data.edge_index, data.edge_weight, ...
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < (self.num_layers - 1):  # hidden layers
                x = F.relu(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def build_graph(image_features, supervised_gene_features, image_indices, supervised_indices, inference_indices, threshold=0.7):
    """
    Build a graph for GCN training with image features and supervised gene features.
    """
    # Create the similarity matrix based on image features
    sim_matrix = cosine_similarity(image_features)
    num_nodes = len(image_features)

    # Create edge list based on similarity threshold
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(image_features, dtype=torch.float)

    # Initialize gene features with zeros (for all nodes initially)
    y = torch.zeros((num_nodes, supervised_gene_features.shape[1]), dtype=torch.float)

    # Assign supervised gene features to the correct nodes
    for local_idx, global_idx in enumerate(supervised_indices):
        y[global_idx] = torch.tensor(supervised_gene_features[local_idx], dtype=torch.float)

    # Create masks for training (supervised) and evaluation (inference)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    eval_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[supervised_indices] = True
    eval_mask[inference_indices] = True

    # Return graph data with train and eval masks
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, eval_mask=eval_mask)
    return data

def train_gcn(model, data, epochs=500, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)

        # Calculate loss only on the supervised training points
        loss = loss_fn(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"GCN Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    
    return model

def train_agdn(model, data, epochs=500, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)

        # Calculate loss only on the supervised training points
        loss = loss_fn(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"GCN Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    
    return model

def evaluate_gcn(model, data, inference_gene_features):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        
        # Calculate loss only on the inference points
        pred = output[data.eval_mask]
        true = torch.tensor(inference_gene_features, dtype=torch.float)

        # Calculate Euclidean distances between predictions and true values
        distances = torch.norm(pred - true, dim=1).cpu().numpy()
        loss = np.mean(distances)
        print(f"GCN Test Error (Euclidean Distance): {loss}")
        
    return loss, distances

def evaluate_adgn(model, data, inference_gene_features):
    model.eval()
    with torch.no_grad():
        output = model(data)
        
        # Calculate loss only on the inference points
        pred = output[data.eval_mask]
        true = torch.tensor(inference_gene_features, dtype=torch.float)

        # Calculate Euclidean distances between predictions and true values
        distances = torch.norm(pred - true, dim=1).cpu().numpy()
        loss = np.mean(distances)
        print(f"GCN Test Error (Euclidean Distance): {loss}")
        
    return loss, distances

def gcn_regression(image_features, supervised_gene_features, inference_gene_features, image_indices, supervised_indices, inference_indices):
    # Prepare graph data correctly
    data = build_graph(image_features, supervised_gene_features, image_indices, supervised_indices, inference_indices)

    # Create and train the GCN model
    model = GCN(in_channels=image_features.shape[1], hidden_channels=64, out_channels=supervised_gene_features.shape[1])
    model = train_gcn(model, data, epochs=1500, lr=0.01)

    # Evaluate the model on inference data
    loss, distances = evaluate_gcn(model, data, inference_gene_features)
    return loss, distances

def agdn_regression(image_features, supervised_gene_features, inference_gene_features, image_indices, supervised_indices, inference_indices):
    # Prepare graph data correctly
    data = build_graph(image_features, supervised_gene_features, image_indices, supervised_indices, inference_indices, threshold = 0.8)

    # Create and train the GCN model
    model = AGDN(num_features=image_features.shape[1], hidden_dim=256, out_dim=supervised_gene_features.shape[1], num_layers=2, K=2, aggr_mode='attention')
    model = train_agdn(model, data, epochs=1500, lr=0.01)

    # Evaluate the model on inference data
    loss, distances = evaluate_adgn(model, data, inference_gene_features)
    return loss, distances


###########################
# Main Experiment Routine #
###########################
def run_experiment(csv_path, image_folder, n_families, n_samples=50, supervised=0.05):
    # Load dataset and images
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    image_samples, gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised)

    # Load pre-trained models
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    # Encode image features for each sample group
    image_features_image = encode_images(image_samples['processid'].values, images, image_model, image_transform)
    image_features_supervised = encode_images(supervised_samples['processid'].values, images, image_model, image_transform)
    image_features_inference = encode_images(inference_samples['processid'].values, images, image_model, image_transform)

    # Combine encoded features into one matrix
    image_features = np.concatenate([image_features_image, image_features_supervised, image_features_inference])

    # Encode gene features for supervised samples only
    supervised_gene_features = encode_genes(supervised_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    inference_gene_features = encode_genes(inference_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)

    # Get indices for supervised and inference samples
    image_indices = list(range(len(image_features)))
    supervised_indices = list(range(len(image_features_image), len(image_features_image) + len(image_features_supervised)))
    inference_indices = list(range(len(image_features_image) + len(image_features_supervised), len(image_features)))

    # Perform Bridged Clustering
    image_kmeans, gene_kmeans, _, gene_features, image_clusters, gene_clusters = perform_clustering(
        image_samples, gene_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families
    )

    # Build the decision matrix using supervised samples
    decision_matrix = build_decision_matrix(supervised_samples, images, image_model, image_transform,
                                            barcode_tokenizer, barcode_model, image_kmeans, gene_kmeans, n_families)

    # Compute centroids for gene clusters using the gene samples
    centroids = compute_gene_centroids(gene_samples, gene_features, gene_kmeans, n_families)

    # Perform inference using Bridged Clustering
    inference_samples_bc = perform_inference(inference_samples, images, image_model, image_transform,
                                             barcode_tokenizer, barcode_model, image_kmeans, decision_matrix, centroids)

    # Compute Bridged Clustering error on inference samples
    bkm_error = compute_error_on_inference(inference_samples_bc)

    # Train and evaluate GCN using correctly assigned samples
    loss, gcn_error = gcn_regression(image_features, supervised_gene_features, inference_gene_features, image_indices, supervised_indices, inference_indices)
    # Train and evaluate AGDN using correctly assigned samples
    loss, agdn_error = agdn_regression(image_features, supervised_gene_features, inference_gene_features, image_indices, supervised_indices, inference_indices)

    # Print results
    print(f"Bridged Clustering Error: Mean={np.mean(bkm_error)}, Std={np.std(bkm_error)}")
    print(f"GCN Error: Mean={np.mean(gcn_error)}, Std={np.std(gcn_error)}")
    print(f"AGDN Error: Mean={np.mean(agdn_error)}, Std={np.std(agdn_error)}")

    return np.mean(bkm_error), np.mean(gcn_error), np.mean(agdn_error)


if __name__ == '__main__':
    csv_path = '../test_data.csv'
    image_folder = '../test_images'


    # Initialize a 3D matrix to store results for each experiment
    # Dimensions: [n_families, n_samples, 3] where 3 corresponds to [BKM, KNN, Mean Teacher]
    results_matrix = np.empty((2, 3, 3))
    bkm_gcn_matrix = np.empty((2, 3))
    success_rate_matrix = np.empty((2, 3))

    # Map indices for n_families and n_samples
    n_families_values = [3]
    n_samples_values = [100]

    for n_families_idx, n_families in enumerate(n_families_values):
        for n_samples_idx, n_samples in enumerate(n_samples_values):
            bkm_total, gcn_total, agdn_total = [], [], []
            n_trials = 50
            success = 1
            for i in range(n_trials):
                print(f"Experiment {i+1} for n_families={n_families}, n_samples={n_samples}")
                bkm, gcn, agdn = run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=0.15)
                if bkm < gcn:
                    success += 1
                bkm_total.append(bkm)
                gcn_total.append(gcn)
                agdn_total.append(agdn)
            results_matrix[n_families_idx, n_samples_idx, 0] = np.mean(np.array(bkm_total))
            results_matrix[n_families_idx, n_samples_idx, 1] = np.mean(np.array(gcn_total))
            results_matrix[n_families_idx, n_samples_idx, 2] = np.mean(np.array(agdn_total))
            print("=====================================")
            print(f"With {n_families} families, {n_samples} samples per family, and 1 sample per family:")
            print("Bridged Clustering Errors Average:", np.mean(np.array(bkm_total)))
            print("GCN Errors Average:", np.mean(np.array(gcn_total)))
            print("AGDN Errors Average:", np.mean(np.array(agdn_total)))
            bkm_gcn_matrix[n_families_idx, n_samples_idx] = np.mean(np.mean(np.array(bkm_total)/np.array(gcn_total)))
            print("Success Rate of Bridged Clustering:", success/n_trials)
            success_rate_matrix[n_families_idx, n_samples_idx] = success/n_trials
            print("=====================================")


    print("=====================================")
    print("Results Matrix:")
    print(results_matrix)
    print("=====================================")
    print("BKM-GCN Matrix:")
    print(bkm_gcn_matrix)
    print("=====================================")
    print("Success Rate Matrix:")
    print(success_rate_matrix)
    print("=====================================")



# With 3 families, 100 samples per family, and 1 sample per family:
# Bridged Clustering Errors Average: 10.12025743560363
# GCN Errors Average: 17.085516
# AGDN Errors Average: 17.789927
# Success Rate of Bridged Clustering: 0.92

## in 26 trials
## BKM > AGDN > GCN in 15 cases
## BKM > GCN > AGDN in 8 cases
## GCN > BKM > AGDN in 3 cases
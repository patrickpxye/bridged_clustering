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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

############################################################
# AGDN Layer Implementation
############################################################


import pandas as pd
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import Adam

# AGDN Layer Implementation
class AGDNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K=3, aggr_mode='attention', bias=True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.res_lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if aggr_mode == 'attention':
            self.att_query = nn.Parameter(torch.Tensor(2 * out_channels))
        elif aggr_mode == 'convolution':
            self.hop_kernel = nn.Parameter(torch.Tensor(K+1, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.res_lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if hasattr(self, 'att_query'):
            nn.init.zeros_(self.att_query)
        if hasattr(self, 'hop_kernel'):
            nn.init.xavier_uniform_(self.hop_kernel)

    def forward(self, x, edge_index, edge_weight=None):
        x0 = self.lin(x)
        multi_hop_list = [x0]
        cur_x = x0
        for _ in range(self.K):
            cur_x = self.propagate(edge_index, x=cur_x, edge_weight=edge_weight)
            multi_hop_list.append(cur_x)
        
        hstack = torch.stack(multi_hop_list, dim=1)
        if hasattr(self, 'att_query'):
            ref = hstack[:, 0:1, :].expand(-1, self.K+1, -1)
            cat_r = torch.cat([ref, hstack], dim=-1)
            q = self.att_query.view(1, 1, -1)
            scores = torch.einsum('nkh,nkh->nk', [cat_r, q.expand(cat_r.shape[0], cat_r.shape[1], -1)])
            scores = F.leaky_relu(scores, negative_slope=0.2)
            alpha = F.softmax(scores, dim=1)
            alpha = alpha.unsqueeze(-1)
            out = torch.sum(alpha * hstack, dim=1)
        elif hasattr(self, 'hop_kernel'):
            kernel = self.hop_kernel.unsqueeze(0)
            out = (hstack * kernel).sum(dim=1)
        
        out = out + self.res_lin(x)
        if self.bias is not None:
            out = out + self.bias
        return out

class AGDN(nn.Module):
    def __init__(self, num_features, hidden_dim, out_dim, num_layers=2, K=3, aggr_mode='attention'):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.K = K
        self.aggr_mode = aggr_mode
        self.convs.append(AGDNConv(num_features, hidden_dim, K=self.K, aggr_mode=self.aggr_mode))
        for _ in range(num_layers - 2):
            self.convs.append(AGDNConv(hidden_dim, hidden_dim, K=self.K, aggr_mode=self.aggr_mode))
        self.convs.append(AGDNConv(hidden_dim, out_dim, K=self.K, aggr_mode=self.aggr_mode))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < (self.num_layers - 1):
                x = F.relu(x)
        return x

############################################################
# AGDN Regression: analog of your gcn_regression
############################################################

def agdn_regression(supervised_df, test_df, image_folder, image_model, image_transform, barcode_tokenizer, barcode_model, num_features=1000, hidden_dim=64, out_dim=768, num_layers=2, K=3, aggr_mode='attention', epochs=500, lr=0.01, threshold=0.7):
    supervised_df = encode_images_for_samples(supervised_df, image_folder, image_model, image_transform)
    supervised_df = encode_genes_for_samples(supervised_df, barcode_tokenizer, barcode_model)
    test_df = encode_images_for_samples(test_df, image_folder, image_model, image_transform)
    test_df = encode_genes_for_samples(test_df, barcode_tokenizer, barcode_model)

    image_features = np.concatenate([supervised_df['morph_coordinates'].tolist(), test_df['morph_coordinates'].tolist()], axis=0)
    supervised_gene = np.array(supervised_df['gene_coordinates'].tolist())
    test_gene = np.array(test_df['gene_coordinates'].tolist())

    supervised_indices = np.arange(len(supervised_df))
    test_indices = np.arange(len(test_df)) + len(supervised_df)
    X = torch.tensor(np.concatenate([supervised_df['morph_coordinates'].tolist(), test_df['morph_coordinates'].tolist()]), dtype=torch.float)
    Y = torch.zeros((len(X), out_dim), dtype=torch.float)

    Y[supervised_indices] = torch.tensor(supervised_gene, dtype=torch.float)

    data = build_graph(image_features, supervised_gene, supervised_indices, test_indices, threshold=threshold)
    data.x = X
    data.y = Y
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[supervised_indices] = True
    eval_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    eval_mask[test_indices] = True
    data.train_mask = train_mask
    data.eval_mask = eval_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = AGDN(num_features=num_features, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, K=K, aggr_mode=aggr_mode).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(data)
        pred_gene = pred[data.eval_mask]
        true_gene = data.y[data.eval_mask]

    distances = (pred_gene - true_gene).norm(dim=1).cpu().numpy()
    mean_dist = distances.mean()
    return distances


def bridged_agdn_regression(supervised_df, test_df, image_folder, image_model, image_transform, barcode_tokenizer, barcode_model, num_features=768, hidden_dim=64, out_dim=768, num_layers=2, K=3, aggr_mode='attention', epochs=500, lr=0.01, threshold=0.7):
    supervised_df = encode_images_for_samples(supervised_df, image_folder, image_model, image_transform)
    supervised_df = encode_genes_for_samples(supervised_df, barcode_tokenizer, barcode_model)
    test_df = encode_images_for_samples(test_df, image_folder, image_model, image_transform)
    test_df = encode_genes_for_samples(test_df, barcode_tokenizer, barcode_model)

    # 1. Build Gene Graph and propagate info
    gene_data = build_gene_graph(supervised_df, test_df, barcode_tokenizer, barcode_model)
    model_gene = AGDN(num_features=num_features, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, K=K, aggr_mode=aggr_mode)
    train_agdn_model(model_gene, gene_data, epochs=epochs, lr=lr)

    # Extract global gene information from supervised data
    model_gene.eval()
    final_gene_embed = model_gene(gene_data)
    supervised_embed_indices = get_supervised_gene_indices(gene_data)
    supervised_gene_embeddings = final_gene_embed[supervised_embed_indices]
    supervised_df['global_gene_coords'] = supervised_gene_embeddings.cpu().detach().numpy()

    # 2. Build Image Graph and propagate gene information
    image_data = build_image_graph(supervised_df, test_df, images, image_model, image_transform, combine_gene_embedding=True):
    model_image = AGDN(num_features=num_features, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, K=K, aggr_mode=aggr_mode)
    train_agdn_model(model_image, image_data, epochs=epochs, lr=lr)

    # 3. Evaluate on test set
    model_image.eval()
    final_image_embed = model_image(image_data)

    return final_image_embed

############################################################
# Example "build_graph" for AGDN
############################################################

def build_graph(image_features, gene_features, supervised_indices, test_indices, threshold=0.7):
    """
    Example method for building a PyG 'Data' object for AGDN, 
    similar to your bridging or gcn code's build_graph.

    You might adapt your existing method to produce:
      data.edge_index: edges in COO format
      data.edge_weight: if needed
    """
    import numpy as np
    import torch
    from torch_geometric.data import Data

    N = len(image_features)
    # For demonstration, let's create a naive adjacency by thresholding pairwise cos similarity:
    # In your code, you might incorporate morphological similarity or some biology-based adjacency.

    # shape => [N, D]
    X_ = image_features
    # (just an example of building adjacency - you can adapt your own logic)
    # caution: for large N, do NOT do a full NxN similarity in real practice 
    # or it may blow up memory. This is just a conceptual snippet. 
    # For actual code, see your existing "build_graph" logic or neighbor-based approach.
    sim = (X_ @ X_.T) / (
        np.linalg.norm(X_, axis=1, keepdims=True) * 
        np.linalg.norm(X_, axis=1, keepdims=True).T + 1e-9
    )
    adj = (sim > threshold).astype(int)
    # remove self-loops
    np.fill_diagonal(adj, 0)

    # Build edge index
    edge_index = np.array(np.nonzero(adj), dtype=np.int64)
    # optional edge_weight
    edge_weight = sim[edge_index[0], edge_index[1]]

    # Create Data
    data = Data()
    data.edge_index = torch.from_numpy(edge_index)
    data.edge_weight = torch.from_numpy(edge_weight).float()

    return data

def get_supervised_gene_indices(gene_data):
    """
    This function retrieves the indices of the supervised nodes in the gene graph.
    These are the nodes that have both image and gene data (the supervised set).
    
    Parameters:
        gene_data (Data): A PyG Data object containing the node features, edge_index, etc.
    
    Returns:
        torch.Tensor: Indices of the supervised nodes.
    """
    # The supervised nodes are marked in the dataset by having non-zero labels in 'gene_coordinates'
    # This assumes that the 'gene_coordinates' column has valid values for supervised nodes.
    supervised_indices = torch.nonzero(gene_data.y != 0).squeeze().tolist()
    return supervised_indices

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


def build_gene_graph(gene_samples, supervised_samples, barcode_tokenizer, barcode_model):
    """
    Build a graph for AGDN with gene data, including unsupervised gene nodes and supervised ones.
    The supervised samples are the bridge between gene and image data.
    Args:
        gene_samples (DataFrame): Gene data of all unsupervised samples.
        supervised_samples (DataFrame): Supervised samples containing both gene and image data.
    Returns:
        data (Data): A PyTorch Geometric Data object with graph information for gene propagation.
    """

    gene_features = encode_genes(gene_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    supervised_gene_features = encode_genes(supervised_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)

    # Merge supervised and unsupervised gene features
    all_gene_features = np.concatenate([gene_features, supervised_gene_features], axis=0)

    # Build adjacency based on gene similarity (you can adjust this)
    adj = cosine_similarity(all_gene_features)
    np.fill_diagonal(adj, 0)  # Remove self-loops

    # Create edge indices
    edge_index = np.array(np.nonzero(adj > 0.7), dtype=np.int64)

    # Create PyTorch Geometric data
    data = Data(x=torch.tensor(all_gene_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(np.concatenate([gene_features, supervised_gene_features], axis=0), dtype=torch.float))  # Add y

    return data


def build_image_graph(image_samples, supervised_samples, images, image_model, image_transform, combine_gene_embedding=True):
    """
    Build a graph for AGDN with image data, including unsupervised image nodes and supervised nodes.
    The supervised samples carry global gene embeddings.

    Args:
        image_samples (DataFrame): Image data for all unsupervised samples.
        supervised_samples (DataFrame): Supervised samples containing both gene and image data.
        images (dict): A mapping of image IDs to image paths.
        combine_gene_embedding (bool): Whether to combine the global gene embeddings to supervised samples' image features.

    Returns:
        data (Data): A PyTorch Geometric Data object with graph information for image propagation.
    """
    # Encode image features
    image_features = encode_images(image_samples['processid'].values, images, image_model, image_transform)
    supervised_image_features = encode_images(supervised_samples['processid'].values, images, image_model, image_transform)

    # Optionally combine global gene embeddings to image features for supervised nodes
    if combine_gene_embedding:
        global_gene_coords = supervised_samples['global_gene_coords'].values
        supervised_image_features = np.concatenate([supervised_image_features, global_gene_coords], axis=1)

    # Merge supervised and unsupervised image features
    all_image_features = np.concatenate([image_features, supervised_image_features], axis=0)

    # Build adjacency based on image similarity
    adj = cosine_similarity(all_image_features)
    np.fill_diagonal(adj, 0)  # Remove self-loops

    # Create edge indices
    edge_index = np.array(np.nonzero(adj > 0.7), dtype=np.int64)

    # Create PyTorch Geometric data
    data = Data(x=torch.tensor(all_image_features, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long))

    return data


def train_agdn_model(model, data, epochs=500, lr=0.001):
    """
    Train the AGDN model with graph data.

    Args:
        model (nn.Module): The AGDN model to be trained.
        data (Data): The PyTorch Geometric Data object containing graph information.
        epochs (int): The number of epochs for training.
        lr (float): The learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def encode_images_for_samples(df, image_folder, image_model, image_transform):
    """
    Encode image features for each sample in the dataframe.

    Args:
        df (DataFrame): The data frame containing the sample IDs.
        image_folder (str): Path to the folder containing image files.
        image_model (nn.Module): Pre-trained image model.
        image_transform (Transform): Preprocessing transform for images.

    Returns:
        df (DataFrame): The original dataframe with added image features.
    """
    features = encode_images(df['processid'].values, image_folder, image_model, image_transform)
    df['morph_coordinates'] = features.tolist()
    return df

def encode_genes_for_samples(df, barcode_tokenizer, barcode_model):
    """
    Encode gene barcode sequences for each sample in the dataframe.

    Args:
        df (DataFrame): The data frame containing the sample IDs and gene barcodes.
        barcode_tokenizer (AutoTokenizer): Pre-trained tokenizer for barcode data.
        barcode_model (AutoModel): Pre-trained model for encoding gene barcodes.

    Returns:
        df (DataFrame): The original dataframe with added gene features.
    """
    gene_features = encode_genes(df['dna_barcode'].values, barcode_tokenizer, barcode_model)
    df['gene_coordinates'] = gene_features.tolist()
    return df



def run_experiment(csv_path, image_folder, n_families, n_samples=50, supervised=0.05, epochs=500, lr=0.01, threshold=0.7):

    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    image_samples, gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised)

    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    # Compare AGDN vs Bridged AGDN
    print("Running AGDN...")
    agdn_distances = agdn_regression(supervised_samples, inference_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, epochs=epochs, lr=lr, threshold=threshold)
    print("Running Bridged AGDN...")
    bridged_agdn_distances = bridged_agdn_regression(supervised_samples, inference_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, epochs=epochs, lr=lr, threshold=threshold)

    print("AGDN Error:", np.mean(agdn_distances))
    print("Bridged AGDN Error:", np.mean(bridged_agdn_distances))
    return np.mean(agdn_distances), np.mean(bridged_agdn_distances)






# Example usage:
csv_path = '../test_data.csv'
image_folder = '../test_images'
n_families = 3
n_samples = 50
supervised = 0.02
num_layers = 2
lr = 0.001
epochs = 100

agdn_error, bridged_agdn_error_mean = run_experiment(
    csv_path, image_folder, n_families, n_samples, supervised, epochs, lr
)

print(f"AGDN Error: {agdn_error}")
print(f"Bridged AGDN Error: {bridged_agdn_error_mean}")

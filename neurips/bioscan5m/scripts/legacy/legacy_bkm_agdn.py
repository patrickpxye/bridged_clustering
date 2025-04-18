import pandas as pd
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from torchvision import models, transforms
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.optim import Adam

############################################################
# AGDN Layer Implementation
############################################################

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


############################################################
# AGDN Regression: analog of your gcn_regression
############################################################

def agdn_regression(supervised_df,
                    test_df,
                    image_model=None,
                    image_transform=None,
                    barcode_tokenizer=None,
                    barcode_model=None,
                    image_folder=None,
                    num_features=128,
                    hidden_dim=64,
                    out_dim=128,
                    num_layers=2,
                    K=3,
                    aggr_mode='attention',
                    epochs=500,
                    lr=0.01,
                    threshold=0.7):
    """
    This function parallels 'gcn_regression', but uses AGDN for the graph-based
    regression from morphological space to gene space.

    Steps:
      1) Encode images, gene features (if not already done).
      2) Build PyG data object from 'build_graph' or similar function.
      3) Split nodes into supervised mask and inference mask (like your code).
      4) Train AGDN with MSE loss on supervised nodes.
      5) Evaluate on test/inference nodes.

    Returns:
      distances (numpy array): the per-sample Euclidean distance
                               between predicted gene coords and ground truth.
    """

    # 1) Possibly encode image / gene features for supervised_df and test_df
    #    Similar to your approach for bridging code.
    #    If not done upstream, do it now:
    supervised_df = encode_images_for_samples(supervised_df, image_folder, image_model, image_transform)
    supervised_df = encode_genes_for_samples(supervised_df, barcode_tokenizer, barcode_model)
    test_df       = encode_images_for_samples(test_df, image_folder, image_model, image_transform)
    test_df       = encode_genes_for_samples(test_df, barcode_tokenizer, barcode_model)

    # 2) Build graph the same way you do for GCN:
    #    We have "supervised_indices" and "inference_indices" from your "build_graph"
    #    or "build_graph(...) + train_mask + eval_mask".
    #    For demonstration, we reuse your build_graph(...) approach:
    #    data = build_graph(...)

    image_features = np.concatenate([supervised_df['morph_coordinates'].tolist(),
                                     test_df['morph_coordinates'].tolist()], axis=0)
    # Suppose we keep track of supervised gene coords for training
    supervised_gene = np.array(supervised_df['gene_coordinates'].tolist())
    # We'll create a big zero array for the unsupervised portion
    test_gene       = np.array(test_df['gene_coordinates'].tolist())
    true_test_gene = torch.tensor(test_gene, dtype=torch.float)


    # Suppose we have indices:
    supervised_indices = np.arange(len(supervised_df))
    test_indices       = np.arange(len(test_df)) + len(supervised_df)
    X = torch.tensor(np.concatenate([supervised_df['morph_coordinates'].tolist(),
                                     test_df['morph_coordinates'].tolist()]), dtype=torch.float)
    Y = torch.zeros((len(X), out_dim), dtype=torch.float)

    # Put actual gene coords for supervised portion
    Y[supervised_indices] = torch.tensor(supervised_gene, dtype=torch.float)

    # Edges: from your 'build_graph' approach.  Example:
    data = build_graph(image_features, threshold=threshold)

    data.x = X
    data.y = Y
    # data.edge_index is set from build_graph
    # data.edge_weight is optional, if you have weighted edges

    # We define train_mask and test_mask
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[supervised_indices] = True
    eval_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    eval_mask[test_indices] = True
    data.train_mask = train_mask
    data.eval_mask  = eval_mask

    # 3) Move the data onto GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 4) Initialize AGDN model
    model = AGDN(num_features=num_features,
                 hidden_dim=hidden_dim,
                 out_dim=out_dim,
                 num_layers=num_layers,
                 K=K,
                 aggr_mode=aggr_mode).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    # 5) Train loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    final_loss = loss.item()
    print(f"Final training loss after {epochs} epochs = {final_loss:.4f}")

    # 5) If loss is still > 1, add another 1500 epochs, up to three tries of this
    iteration = 0
    while final_loss > 1 and iteration < 3:
        print(f"Final loss > 1, adding another 1500 epochs.")
        for epoch in range(1500):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        print(f"Final training loss after {epochs + 1500} epochs = {final_loss:.4f}")
        iteration += 1

    # 6) Inference
    model.eval()
    with torch.no_grad():
        pred = model(data)
        pred_gene = pred[data.eval_mask]   # shape: [#test, out_dim]

    # 7) Compute Euclidean distance
    distances = (pred_gene - true_test_gene).norm(dim=1).cpu().numpy()
    mean_dist = distances.mean()
    print(f"AGDN on test set: Mean Dist = {mean_dist:.4f}")

    return distances


############################################################
# Example "build_graph" for AGDN
############################################################

def build_graph(image_features, threshold=0.7):
    """
    Example method for building a PyG 'Data' object for AGDN, 
    similar to your bridging or gcn code's build_graph.

    You might adapt your existing method to produce:
      data.edge_index: edges in COO format
      data.edge_weight: if needed
    """

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
    # n_supervised = 1
    # Use remaining samples for inference to ensure full coverage
    n_inference = n - (n_gene + n_supervised)

    gene_samples = family_data.iloc[:n_gene]
    supervised_samples = family_data.iloc[n_gene:n_gene + n_supervised]
    inference_samples = family_data.iloc[n_gene + n_supervised:n_gene + n_supervised + n_inference]

    return gene_samples, supervised_samples, inference_samples

def get_data_splits(df, supervised):
    """
    Loop over families in the DataFrame and concatenate splits from each family.
    Returns four DataFrames: image_samples, gene_samples, supervised_samples, inference_samples.
    """
    gene_list, sup_list, inf_list = [], [], []
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
def perform_clustering(image_samples, supervised_samples, gene_samples, images, image_model, image_transform, barcode_tokenizer, barcode_model, n_families):
    """
    Perform KMeans clustering on image and gene samples.
    Returns the trained KMeans objects and raw features.
    """
    image_features = encode_images(image_samples['processid'].values, images, image_model, image_transform)
    supervised_features = encode_images(supervised_samples['processid'].values, images, image_model, image_transform)
    concatenated_features = np.concatenate((image_features, supervised_features), axis=0)
    image_kmeans = KMeans(n_clusters=n_families, random_state=42).fit(concatenated_features)
    
    gene_features = encode_genes(gene_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    gene_kmeans = KMeans(n_clusters=n_families, random_state=42).fit(gene_features)
    
    return image_kmeans, gene_kmeans, image_features, gene_features

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
    



###########################
# Main Experiment Routine #
###########################
def run_experiment(csv_path, image_folder, n_families, n_samples=50, supervised=0.05):
    """
    Compares Bridged Clustering to AGDN on the BIOSCAN dataset.
    """

    # 1) Load dataset and images
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised)


    # 2) Load pre-trained models
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    # 3) Perform Bridged Clustering
    image_kmeans, gene_kmeans, _, gene_features = perform_clustering(
            inference_samples, supervised_samples, gene_samples, images,
            image_model, image_transform,
            barcode_tokenizer, barcode_model,
            n_families
        )

    decision_matrix = build_decision_matrix(
        supervised_samples, images,
        image_model, image_transform,
        barcode_tokenizer, barcode_model,
        image_kmeans, gene_kmeans, n_families
    )
    centroids = compute_gene_centroids(
        gene_samples, gene_features,
        gene_kmeans, n_families
    )
    inference_samples_bc = perform_inference(
        inference_samples, images,
        image_model, image_transform,
        barcode_tokenizer, barcode_model,
        image_kmeans, decision_matrix, centroids
    )
    bkm_error = compute_error_on_inference(inference_samples_bc)

    # 4) Prepare data for AGDN
    #    We'll gather the features for the "supervised" subset and the "inference" subset.
    #    Then feed them to our new function 'agdn_regression' that you pasted in.

    # Encode morphological and gene embeddings
    # for the supervised and inference sets:
    supervised_df = supervised_samples.copy()
    inference_df  = inference_samples.copy()

    # 5) Train & evaluate AGDN
    #    This returns the Euclidean distances for each inference sample
    agdn_distances = agdn_regression(
        supervised_df,    # includes morphological + gene
        inference_df,     # includes morphological + gene
        # plus any other relevant arguments...
        image_model=image_model,
        image_transform=image_transform,
        barcode_tokenizer=barcode_tokenizer,
        barcode_model=barcode_model,
        image_folder=images,
        # optional hyperparams:
        num_features=1000,   # dimension of morphological embeddings
        hidden_dim=256,
        out_dim=768,        # dimension of gene embeddings
        num_layers=2,
        K=3,
        aggr_mode='attention',
        epochs=1500,
        lr=0.05,
        threshold=0.8       # if build adjacency via threshold
    )

    # Compute mean ± std error for AGDN
    agdn_error = np.mean(agdn_distances)
    agdn_std   = np.std(agdn_distances)

    # 6) Print results
    print(f"\nBridged Clustering Error  => Mean = {np.mean(bkm_error):.4f}, Std = {np.std(bkm_error):.4f}")
    print(f"AGDN Error               => Mean = {agdn_error:.4f}, Std = {agdn_std:.4f}\n")

    return np.mean(bkm_error), agdn_error



if __name__ == '__main__':
    csv_path = '../test_data.csv'
    image_folder = '../test_images'


    results_matrix = np.empty((1, 11, 2))
    bkm_agdn_matrix = np.empty((1, 11))
    success_rate_matrix = np.empty((1, 11))

    # Map indices for n_families and n_samples
    n_families_values = [3]
    n_samples_values = [50]

    for n_families_idx, n_families in enumerate(n_families_values):
        for n_samples_idx, n_samples in enumerate(n_samples_values):
            bkm_total, agdn_total = [], []
            n_trials = 1
            success = 0
            for i in range(n_trials):
                print(f"Experiment {i+1} for n_families={n_families}, n_samples={n_samples}")
                bkm, agdn = run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=0.01)
                if bkm < agdn:
                    success += 1
                bkm_total.append(bkm)
                agdn_total.append(agdn)
            results_matrix[n_families_idx, n_samples_idx, 0] = np.mean(np.array(bkm_total))
            results_matrix[n_families_idx, n_samples_idx, 1] = np.mean(np.array(agdn_total))
            print("=====================================")
            print(f"With {n_families} families, {n_samples} samples per family, and 1 sample per family:")
            print("Bridged Clustering Errors Average:", np.mean(np.array(bkm_total)))
            print("AGDN Errors Average:", np.mean(np.array(agdn_total)))
            bkm_agdn_matrix[n_families_idx, n_samples_idx] = np.mean(np.mean(np.array(bkm_total)/np.array(agdn_total)))
            print("Success Rate of Bridged Clustering:", success/n_trials)
            success_rate_matrix[n_families_idx, n_samples_idx] = success/n_trials
            print("=====================================")


    print("=====================================")
    print("Results Matrix:")
    print(results_matrix)
    print("=====================================")
    print("BKM-AGDN Matrix:")
    print(bkm_agdn_matrix)
    print("=====================================")
    print("Success Rate Matrix:")
    print(success_rate_matrix)
    print("=====================================")
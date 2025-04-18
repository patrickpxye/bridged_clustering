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
import torch_geometric
import math

############################################################
# AGDN Layer Implementation
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class AGDNConvHC(MessagePassing):
    """
    AGDN layer with Hop-wise Convolution (HC) in PyTorch Geometric style.
    K hops of GAT-based diffusion are aggregated with a learnable hop-wise kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        K=2,
        heads=1,
        negative_slope=0.2,
        dropout=0.0,
        residual=True
    ):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.K = K
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual

        # Linear transformation before attention (shared across hops)
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)

        # GAT attention parameters (shared across hops)
        # We’ll do standard “additive” attention (att_l, att_r).
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Residual projection if shape mismatch
        if residual and (in_channels != heads * out_channels):
            self.res_fc = nn.Linear(in_channels, heads * out_channels, bias=False)
        else:
            self.res_fc = None

        # Bias
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))

        # ---- Hop-wise Convolution Kernel ----
        #  We have K+1 “hop slots” (0..K),
        #  each slot has a weight vector of size [out_channels].
        #  We broadcast them across nodes.
        self.theta = nn.Parameter(torch.Tensor(self.K + 1, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        nn.init.xavier_uniform_(self.att_l, gain=1.0)
        nn.init.xavier_uniform_(self.att_r, gain=1.0)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=1.0)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.theta)  # Start small so training can find good combination

    def forward(self, x, edge_index):
        """
        x:          Node features [N, in_channels]
        edge_index: Graph connectivity [2, E]
        """
        # 1) Transform input features
        x_lin = self.linear(x).view(-1, self.heads, self.out_channels)

        # 2) (Optional) Precompute residual
        if self.res_fc is not None:
            x_res = self.res_fc(x).view(-1, self.heads, self.out_channels)
        else:
            x_res = x_lin if self.residual else None

        # 3) Gather multi-hop representations
        reps = []  # list of size (K+1), each of shape [N, heads, out_channels]

        # Hop 0: Just the transformed input
        reps.append(x_lin)

        # Hops 1..K: repeatedly apply message passing
        cur = x_lin
        for _ in range(self.K):
            cur = self.propagate(edge_index, x=cur)
            reps.append(cur)

        # 4) Combine with hop-wise convolution kernel
        #    reps shape: [(N, heads, out_channels), ... K+1 times ...]
        #    stack -> (K+1, N, heads, out_channels)
        all_reps = torch.stack(reps, dim=0)

        # We'll multiply each hop’s representation by theta[k, :] across channels
        # We need to broadcast theta from shape [K+1, out_channels]
        # to [K+1, 1, 1, out_channels], then multiply.
        theta_view = self.theta.view(self.K+1, 1, 1, self.out_channels)
        # out shape => [K+1, N, heads, out_channels]
        out = all_reps * theta_view

        # Sum over hops
        out = out.sum(dim=0)  # now shape is [N, heads, out_channels]

        # 5) Flatten heads
        out = out.view(-1, self.heads * self.out_channels)

        # 6) Add residual
        if self.residual and (x_res is not None):
            out += x_res.view(-1, self.heads * self.out_channels)

        # 7) Add bias
        out = out + self.bias
        return out

    def message(self, x_j):
        """
        GAT-style 'message' function with no explicit edge-level gating.
        We rely on out-of-loop attention computation below in 'edge_updater'
        or we can do it inside 'message'.
        """
        return x_j

    def edge_updater(self, edge_index, x):
        """
        Torch Geometric 2.x approach: we can define
        attention coefficients in an edge_updater or inside 'message'.
        For clarity, let's do it in 'message' if we want standard GAT.
        """
        # This method is optional: in some PyG versions,
        # you explicitly define how to update edges. We'll keep a simpler approach.
        pass

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Overwrite how messages are aggregated at the destination node
        if we want to incorporate the GAT attention explicitly.
        Otherwise, we can rely on the standard 'add' aggregator.
        """
        # For standard GAT, we compute attention before summation (softmax).
        # We'll do that here.  Let's get alpha_l, alpha_r from x_j, x_i if needed.
        # Alternatively, we can do a simpler aggregator if you prefer direct adjacency.
        #
        # A simpler approach: if we always want
        # row-stochastic adjacency = 1/deg(i),
        # we can let PyG handle it. For full GAT attention, see below.
        return super().aggregate(inputs, index, ptr, dim_size)

    def message(self, x_j, x_i, index):
        """
        Overwrites the default message function to include GAT-style attention scores.
        """
        # x_j, x_i have shape [E, heads, out_channels].
        # 1) Compute attention scores e_ij = (x_j * att_l + x_i * att_r).sum(dim=-1)
        alpha_l_j = (x_j * self.att_l).sum(dim=-1, keepdim=True)  # [E, heads, 1]
        alpha_r_i = (x_i * self.att_r).sum(dim=-1, keepdim=True)  # [E, heads, 1]
        alpha = alpha_l_j + alpha_r_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # 2) Normalize by softmax over edges with same target index
        alpha = softmax(alpha, index)  # shape [E, heads, 1]
        alpha = self.dropout(alpha)
        # 3) Multiply features by attention
        return x_j * alpha

class AGDN(nn.Module):
    """
    Stacks multiple AGDNConvHC layers into a deeper network.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2,
                 K=2,
                 heads=1,
                 dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(AGDNConvHC(in_channels, hidden_channels,
                                     K=K, heads=heads, dropout=dropout))
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(AGDNConvHC(hidden_channels * heads,
                                         hidden_channels,
                                         K=K,
                                         heads=heads,
                                         dropout=dropout))
        # Final layer (heads=1 for simpler output)
        self.convs.append(AGDNConvHC(hidden_channels * heads,
                                     out_channels,
                                     K=K,
                                     heads=1,
                                     dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=conv.dropout.p, training=self.training)
        return x


############################################################
# AGDN Regression
############################################################

def agdn_regression(supervised_df, inference_df, morph_col='morph_coordinates', gene_col='gene_coordinates',
                    hidden=64, num_layers=2, heads=1, dropout=0.1,
                    epochs=50, lr=1e-3):
    """
    Train an AGDN to map morphological embeddings -> gene embeddings,
    using 'supervised_df' as labeled data, then predict gene embeddings
    for 'inference_df'. Return the per-sample Euclidean distance.
    
    :param supervised_df: DataFrame with columns [morph_col, gene_col]
    :param inference_df:  DataFrame with columns [morph_col, gene_col]
    :param morph_col:     Name of the morphological embedding column in the DataFrame
    :param gene_col:      Name of the gene embedding column in the DataFrame
    :param hidden:        Hidden dimension
    :param num_layers:    Number of AGDN layers
    :param heads:         Number of heads
    :param dropout:       Dropout rate
    :param epochs:        Training epochs
    :param lr:            Learning rate
    :return: Numpy array of distances for each row in inference_df
    """

    # 1) Extract morphological & gene embeddings as arrays
    sup_morph = np.stack(supervised_df[morph_col].values, axis=0)
    sup_gene  = np.stack(supervised_df[gene_col].values, axis=0)
    inf_morph = np.stack(inference_df[morph_col].values, axis=0)
    inf_gene  = np.stack(inference_df[gene_col].values, axis=0)

    sup_morph_t = torch.FloatTensor(sup_morph)
    sup_gene_t  = torch.FloatTensor(sup_gene)
    inf_morph_t = torch.FloatTensor(inf_morph)
    inf_gene_t  = torch.FloatTensor(inf_gene)

    N_sup = sup_morph.shape[0]
    N_inf = inf_morph.shape[0]

    # 2) Build adjacency for supervised portion
    #    For demonstration, we do a full mesh among supervised nodes
    srcs, dsts = [], []
    for i in range(N_sup):
        for j in range(N_sup):
            if i != j:
                srcs.append(i)
                dsts.append(j)
    edge_index_sup = torch.tensor([srcs, dsts], dtype=torch.long)

    data_sup = Data(x=sup_morph_t, edge_index=edge_index_sup)

    # 3) Build the AGDN
    in_channels  = sup_morph.shape[1]
    out_channels = sup_gene.shape[1]
    model = AGDN(in_channels, hidden, out_channels, num_layers=num_layers, heads=heads, dropout=dropout)

    # 4) Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # 5) Train on supervised
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data_sup.x, data_sup.edge_index)   # [N_sup, out_channels]
        loss = criterion(out, sup_gene_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"AGDN epoch={epoch+1}, MSE={loss.item():.4f}")

    # 6) Build a combined graph that includes inference nodes,
    #    so that we can propagate morphological info from sup->inf
    #    We'll label the new nodes from [N_sup..N_sup+N_inf-1]
    srcs_inf, dsts_inf = [], []
    for i in range(N_inf):
        inf_node_id = N_sup + i
        # Connect to all supervised nodes (bidirectional):
        for j in range(N_sup):
            srcs_inf.append(inf_node_id)
            dsts_inf.append(j)
            srcs_inf.append(j)
            dsts_inf.append(inf_node_id)
    edge_index_inf = torch.tensor([srcs_inf, dsts_inf], dtype=torch.long)

    big_x = torch.cat([sup_morph_t, inf_morph_t], dim=0)
    big_edge = torch.cat([edge_index_sup, edge_index_inf], dim=1)
    data_inf = Data(x=big_x, edge_index=big_edge)

    # 7) Infer for the last portion
    model.eval()
    with torch.no_grad():
        big_out = model(data_inf.x, data_inf.edge_index) # shape [N_sup+N_inf, out_channels]
    inf_pred = big_out[N_sup:, :]  # shape [N_inf, out_channels]

    # 8) Euclidean distances
    distances = torch.norm(inf_pred - inf_gene_t, dim=1)  # shape [N_inf]
    return distances.numpy()


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
    Compare Bridged Clustering (BKM) vs. AGDN on the dataset.
    """
    # 1) Load dataset
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)
    gene_samples, supervised_samples, inference_samples = get_data_splits(df, supervised=supervised)

    # 2) Pre-trained models (image + barcode)
    # Load pre-trained models
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    # Encode supervised data
    supervised_samples = encode_images_for_samples(supervised_samples, images, image_model, image_transform)
    supervised_samples = encode_genes_for_samples(supervised_samples, barcode_tokenizer, barcode_model)
    supervised_samples = supervised_samples.reset_index(drop=True)

    # Encode inference data
    inference_samples = encode_images_for_samples(inference_samples, images, image_model, image_transform)
    inference_samples = encode_genes_for_samples(inference_samples, barcode_tokenizer, barcode_model)
    inference_samples = inference_samples.reset_index(drop=True)


    # 3) BKM
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
    
    # 5) Train & Evaluate AGDN
    agdn_distances = agdn_regression(
        supervised_df=supervised_samples,
        inference_df=inference_samples,
        morph_col='morph_coordinates',
        gene_col='gene_coordinates',
        hidden=64,
        num_layers=2,
        heads=1,
        dropout=0.1,
        epochs=2000,
        lr=1e-3
    )
    agdn_error = np.mean(agdn_distances)
    agdn_std   = np.std(agdn_distances)

    # Print results
    print(f"\nBridged Clustering Error => Mean = {np.mean(bkm_error):.4f}, Std = {np.std(bkm_error):.4f}")
    print(f"AGDN Error               => Mean = {agdn_error:.4f}, Std = {agdn_std:.4f}\n")

    return np.mean(bkm_error), agdn_error



if __name__ == '__main__':
    csv_path = '../test_data.csv'
    image_folder = '../test_images'


    results_matrix = np.empty((1, 1, 2))
    bkm_agdn_matrix = np.empty((1, 1))
    success_rate_matrix = np.empty((1, 1))

    # Map indices for n_families and n_samples
    n_families_values = [3]
    n_samples_values = [50]

    for n_families_idx, n_families in enumerate(n_families_values):
        for n_samples_idx, n_samples in enumerate(n_samples_values):
            bkm_total, agdn_total = [], []
            n_trials = 5
            success = 0
            for i in range(n_trials):
                print(f"Experiment {i+1} for n_families={n_families}, n_samples={n_samples}")
                bkm, agdn = run_experiment(csv_path, image_folder, n_families=n_families, n_samples=n_samples, supervised=0.1)
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
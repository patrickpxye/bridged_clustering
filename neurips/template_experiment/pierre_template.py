import os
import random
import warnings

import numpy as np
import itertools
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import pairwise_distances
import torch
import torchvision
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing

from torch.optim import Adam
import torch_geometric.utils
from tqdm import tqdm
from collections import Counter




from sklearn.neighbors import NearestNeighbors
from k_means_constrained import KMeansConstrained

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#############################
# Configuration & Hyperparams
#############################
BASE_DIR        = "input/ingredients_classifier"
IMAGE_FOLDER    = os.path.join(BASE_DIR, "images")
INGREDIENTS_FILE= os.path.join(BASE_DIR, "ingredients.txt")
RECIPES_FILE    = os.path.join(BASE_DIR, "recipes.txt")
ALL_IMAGES_FILE = os.path.join(BASE_DIR, "all_images.txt")
ALL_LABELS_FILE = os.path.join(BASE_DIR, "all_labels.txt")

IMAGE_SIZE      = 224
N_CLUSTERS      = [2, 3, 4, 5, 6, 10] # number of clusters for X (for deterministic clustering pick same amount of cuisines)
SUP_FRACS       = [0.0205, 0.07, 0.15, 0.3] # supervised sample fractions (0.0205 = 1/49)
OUT_FRAC        = 0.55    # fraction of "output-only" samples for Y-clustering
N_TRIALS        = 10
Y_DIM_REDUCED   = 128   # target dim for random projection

#############################
# I/O Helpers
#############################
def load_paths(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_ingredients():
    with open(INGREDIENTS_FILE) as f:
        vocab = [ing.strip() for ing in f.read().split(",") if ing.strip()]
    return vocab, {ing: i for i, ing in enumerate(vocab)}

def load_recipes():
    recipes = []
    with open(RECIPES_FILE) as f:
        for line in f:
            row = line.strip()
            recipes.append([] if not row else [ing.strip() for ing in row.split(",") if ing.strip()])
    return recipes

#############################
# Image Encoder
#############################
def get_image_encoder():
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, tf

def encode_images(df, image_root, model, tfm):
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding images"):
        full = os.path.join(image_root, row["img_path"])
        if not os.path.exists(full):
            feats.append(np.zeros(2048, float)); continue
        img = Image.open(full).convert("RGB")
        x   = tfm(img).unsqueeze(0)
        with torch.no_grad(): feats.append(model(x).squeeze().cpu().numpy())
    return np.vstack(feats)

#############################
# Label Encoder
#############################
def make_multihot(recipe_id, recipes, ing2idx, D):
    vec = np.zeros(D, int)
    for ing in recipes[recipe_id]:
        idx = ing2idx.get(ing)
        if idx is not None: vec[idx] = 1
    return vec

#############################
# Splits: stratified supervised, output-only, inference
#############################
def stratified_split_masks(x_lab, sup_frac, out_frac, n_clusters):
    """
    Create boolean masks for three disjoint subsets:
      - sup_mask:   stratified supervised samples (at least one per X-cluster)
      - out_mask:   "output-only" samples used for Y-clustering but not supervised
      - inf_mask:   inference/test samples (neither used for supervision nor Y-clustering)

    Args:
        x_lab (array-like): cluster labels for all samples in X-space, shape (N,)
        sup_frac (float):   target fraction of supervised samples (e.g. 0.05)
        out_frac (float):   fraction of "output-only" samples for Y-clustering (e.g. 0.4)

    Returns:
        sup_mask (np.ndarray[bool]): shape (N,), True where sample is supervised
        out_mask (np.ndarray[bool]): shape (N,), True where sample is output-only
        inf_mask (np.ndarray[bool]): shape (N,), True where sample is inference-only
    """
    N = len(x_lab)

    # 1) Determine total number of supervised samples, at least one per cluster
    n_sup = max(n_clusters, int(sup_frac * N))

    # 2) Pick one supervised index from each cluster to ensure coverage
    sup_idx = []
    for c in range(n_clusters):
        indices = np.where(x_lab == c)[0]
        if len(indices) > 0:
            sup_idx.append(int(np.random.choice(indices)))

    # 3) Fill remaining supervised slots by random selection from the rest
    remaining = list(set(range(N)) - set(sup_idx))
    to_pick = n_sup - len(sup_idx)
    if to_pick > 0 and remaining:
        extra = np.random.choice(remaining, min(to_pick, len(remaining)), replace=False)
        sup_idx.extend(map(int, extra))

    # Build supervised mask
    sup_mask = np.zeros(N, bool)
    sup_mask[sup_idx] = True

    # 4) From the ``rest`` (non-supervised), pick output-only samples for Y-clustering
    rest = np.where(~sup_mask)[0]
    n_out = int(out_frac * N)
    out_idx = np.random.choice(rest, min(n_out, len(rest)), replace=False)

    # Build output-only mask
    out_mask = np.zeros(N, bool)
    out_mask[out_idx] = True

    # 5) The remaining samples are reserved for inference/testing
    inf_mask = ~(sup_mask | out_mask)

    return sup_mask, out_mask, inf_mask

#############################
# Clustering + Bridging
#############################
def purity_score(true_labels, cluster_labels):
    # true_labels: array-like of length N (e.g. cuisine names)
    # cluster_labels: array-like of length N
    total = len(true_labels)
    score = 0
    for c in np.unique(cluster_labels):
        mask = (cluster_labels == c)
        if mask.sum() == 0: 
            continue
        most_common = Counter(true_labels[mask]).most_common(1)[0][1]
        score += most_common
    return score / total

def cluster_features(X, n_clusters):
    # N = X.shape[0]
    # base_size = N // n_clusters
    # size_min = base_size
    # size_max = size_min + (1 if N % n_clusters else 0)  # ensure all samples are used
    # return KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=42).fit_predict(X)
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

def cluster_ingredients(Y_proj, n_clusters):
    # N = Y_proj.shape[0]
    # base_size = N // n_clusters
    # size_min = base_size
    # size_max = size_min + (1 if N % n_clusters else 0)  # ensure all samples are used
    # return KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=42).fit_predict(Y_proj)
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(Y_proj)
def cluster_features_constrained(X, n_clusters):
    N = len(X)
    base_size = N // n_clusters
    size_min = base_size
    size_max = size_min + (1 if N % n_clusters else 0)  # ensure all samples are used
    return KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=42).fit_predict(X)


def cluster_ingredients_constrained(Y_proj, n_clusters):
    N = len(Y_proj)
    base_size = N // n_clusters
    size_min = base_size
    size_max = size_min + (1 if N % n_clusters else 0)  # ensure all samples are used
    return KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=42).fit_predict(Y_proj)


def learn_bridge(x_lab, y_lab, sup_mask, n_clusters):
    mapping = np.zeros(n_clusters, int)
    for c in range(n_clusters):
        mask = (x_lab == c) & sup_mask
        if mask.any(): mapping[c] = np.bincount(y_lab[mask], minlength=n_clusters).argmax()
    return mapping

def compute_centroids(Y, y_lab, n_clusters):
    centroids = np.zeros((n_clusters, Y.shape[1]), float)
    for c in range(n_clusters):
        mem = (y_lab == c)
        if mem.sum(): centroids[c] = Y[mem].mean(axis=0)
    return centroids

def predict_bridge(x_lab, mapping, centroids):
    return centroids[mapping[x_lab]]

def responsibilities(y, centroids, tau):
    # y:   (D,)   one sample’s bridged‐continuous prediction
    # centroids: (K, D)
    dists  = np.linalg.norm(centroids - y[None], axis=1)      # (K,)
    logits = - dists / tau
    exp    = np.exp(logits - logits.max())
    return exp / exp.sum()       

#############################
# Baseline: KNN
#############################
def knn_baseline(X_train, Y_train, X_test, k=1):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)
    return knn.predict(X_test)

#############################
# Baseline: GNN
#############################

class GNNClusterBridge(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_clusters, tau=1.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_clusters)
        self.tau = tau

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv3(h, edge_index, edge_weight=edge_weight))
        logits = self.classifier(h)             # [N, K]
        return F.softmax(logits / self.tau, dim=-1)  # soft assignments p

def run_gnn_bridged( X, y_lab, sup_mask,
    n_clusters,
    edge_index,
    edge_weights,
    centroids,
    tau,
    hidden_dim,
    lr,
    epochs):
    """
    X: np.ndarray [N×D_x]
    y_lab: array-like [N]
    sup_mask: boolean mask [N]
    n_clusters: int, number of clusters in X
    edge_index: LongTensor [2×E] for PyG
    centroids: np.ndarray [K×D_y], where K is the number of clusters in Y
    tau: float, temperature for softmax
    hidden_dim: int, hidden dimension for GNN
    lr: float, learning rate for optimizer
    epochs: int, number of training epochs
    returns Y_pred: np.ndarray [N×D_y]
    """

    # 2) Build PyG Data and train GNN
    data = Data(
        x=torch.from_numpy(X).float(),
        edge_index=edge_index,
        edge_attr=edge_weights if edge_weights is not None else None
    )
    model = GNNClusterBridge(in_dim=X.shape[1], hidden_dim=hidden_dim, n_clusters=n_clusters, tau=tau)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # loss_history = []
    for epoch in range(epochs):
        model.train()
        p = model(data.x, data.edge_index)        # [N,K]
        loss = F.cross_entropy(
            torch.log(p.clamp(min=1e-10)[sup_mask]), 
            torch.from_numpy(y_lab[sup_mask]).long()
        )
        # loss_history.append(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    
    # # after training:
    # plt.figure(figsize=(6,4))
    # plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-')
    # plt.xlabel("Epoch",  fontsize=12)
    # plt.ylabel("Training Loss", fontsize=12)
    # plt.title("Training Loss vs. Epoch", fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # 3) Inference
    model.eval()
    with torch.no_grad():
        p = model(data.x, data.edge_index)       # [N,K]
    # 4) Bridge: weighted sum of centroids
    Y_pred = p @ torch.from_numpy(centroids).float()  # [N, D_y]
    return Y_pred.cpu().numpy()

# import skfuzzy as fuzz
# def fuzzy_cluster(X, n_clusters, m=2.0, error=1e-5, maxiter=1000):
#     """
#     Run fuzzy c‑means on X (N×D) and return:
#       - centers: (n_clusters×D)
#       - u: membership matrix (n_clusters×N)
#       - hard_labels: argmax over u (N,)
#     """
#     # skfuzzy wants features×samples
#     X_T = X.T
#     centers, u, _, _, _, _, _ = fuzz.cluster.cmeans(
#         X_T, c=n_clusters, m=m,
#         error=error, maxiter=maxiter
#     )
#     hard = np.argmax(u, axis=0)
#     return centers, u, hard

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
    input_dim = len(supervised_samples['image_coordinates'].sample().iloc[0])
    output_dim = len(supervised_samples['ingredient_coordinates'].sample().iloc[0])
    model = MeanTeacherModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prepare data loaders (labeled and unlabeled)
    supervised_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(supervised_samples['image_coordinates'].tolist(), dtype=torch.float32),
            torch.tensor(supervised_samples['ingredient_coordinates'].tolist(), dtype=torch.float32))),
        batch_size=32, shuffle=True)

    unlabeled_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(inference_samples['image_coordinates'].tolist(), dtype=torch.float32),
            torch.zeros_like(torch.tensor(inference_samples['image_coordinates'].tolist(), dtype=torch.float32)))),
        batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(inference_samples['image_coordinates'].tolist(), dtype=torch.float32),
            torch.tensor(inference_samples['ingredient_coordinates'].tolist(), dtype=torch.float32))),
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
    Xtr = np.vstack(sup_df['image_coordinates'])
    ytr = np.vstack(sup_df['ingredient_coordinates'])
    Xte = np.vstack(inf_df['image_coordinates'])
    yte = np.vstack(inf_df['ingredient_coordinates'])

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
    Xs = np.vstack(sup_df['image_coordinates'])
    ys = np.vstack(sup_df['ingredient_coordinates'])
    Xu = np.vstack(inf_df['image_coordinates'])
    w = laprls_closed_form(Xs, ys, Xu, lam, gamma, k, sigma)
    preds = Xu.dot(w)
    actuals = np.vstack(inf_df['ingredient_coordinates'])
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
    Xs = np.vstack(sup_df['image_coordinates'])
    ys = np.vstack(sup_df['ingredient_coordinates'])
    Xu = np.vstack(inf_df['image_coordinates'])
    y_inf = np.vstack(inf_df['ingredient_coordinates'])

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
#     Xs = np.vstack(supervised_df['image_coordinates'])
#     ys = np.vstack(supervised_df['ingredient_coordinates'])
#     Xte = np.vstack(inference_df['image_coordinates'])
#     yte = np.vstack(inference_df['ingredient_coordinates'])
    
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
    X_sup = np.vstack(supervised_df['image_coordinates'])
    y_sup = np.vstack(supervised_df['ingredient_coordinates'])
    X_unl = np.vstack(inference_df['image_coordinates'])
    y_unl = np.vstack(inference_df['ingredient_coordinates'])  # for evaluation

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
    X_sup = torch.tensor(np.vstack(supervised_df['image_coordinates']), dtype=torch.float32, device=device)
    y_sup = torch.tensor(np.vstack(supervised_df['ingredient_coordinates']),    dtype=torch.float32, device=device)
    X_unl = torch.tensor(np.vstack(inference_df['image_coordinates']),   dtype=torch.float32, device=device)
    y_unl_np = np.vstack(inference_df['ingredient_coordinates'])

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
      sup_df: DataFrame with 'image_coordinates', 'ingredient_coordinates'
      inf_df: DataFrame with 'image_coordinates', 'ingredient_coordinates'
    Returns:
      preds_unl (ndarray): predictions for inf_df
      true_unl  (ndarray): ground-truth gene coords for inf_df
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # prepare tensors
    Xs = torch.tensor(np.vstack(sup_df['image_coordinates']), dtype=torch.float32)
    ys = torch.tensor(np.vstack(sup_df['ingredient_coordinates']),    dtype=torch.float32)
    Xu = torch.tensor(np.vstack(inf_df['image_coordinates']),   dtype=torch.float32)
    yu = np.vstack(inf_df['ingredient_coordinates'])

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
#     Xs = torch.tensor(np.vstack(sup_df['image_coordinates']), dtype=torch.float32)
#     ys = torch.tensor(np.vstack(sup_df['ingredient_coordinates']), dtype=torch.float32)
#     Xu = torch.tensor(np.vstack(inf_df['image_coordinates']), dtype=torch.float32)
#     yu = np.vstack(inf_df['ingredient_coordinates'])

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
    sup_m = np.stack(supervised_df['image_coordinates'].values)
    sup_g = np.stack(supervised_df['ingredient_coordinates'].values)
    inf_m = np.stack(inference_df['image_coordinates'].values)
    inf_g = np.stack(inference_df['ingredient_coordinates'].values)
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

#############################
# Single‐trial Experiment
#############################
def run_experiment(df, model, tfm, recipes, ing2idx, vocab_size, X, sup_frac, out_frac, n_clusters, edge_index, edge_weights):
    Y_true = np.vstack(df['ingredient_vec'].values)               # (N, D)
    rp     = GaussianRandomProjection(n_components=Y_DIM_REDUCED, random_state=42)
    Y_proj = rp.fit_transform(Y_true)                             # (N, D′)

    # 1) stratify & split
    x_lab  = cluster_features(X, n_clusters)
    sup_mask, out_mask, inf_mask = stratified_split_masks(x_lab, sup_frac, out_frac, n_clusters)

    # 2) cluster Y only on sup+out
    y_lab = np.empty(len(Y_proj), int)
    y_lab[sup_mask | out_mask] = cluster_ingredients(Y_proj[sup_mask | out_mask], n_clusters)

    # 3) prepare supervised/inference sets for Mean Teacher
    X_sup = X[sup_mask];    Y_sup = Y_true[sup_mask]
    X_inf = X[inf_mask];    Y_inf = Y_true[inf_mask]

    true = df['cuisine_type'].values

    # 1) Purity of X-clusters (on *all* samples)
    x_p = purity_score(true, x_lab)
    nmi_x = normalized_mutual_info_score(true, x_lab)

    # 2) Purity of Y-clusters (only on sup+out)
    mask = sup_mask | out_mask
    true_y = true[mask]
    y_p   = purity_score(true_y, y_lab[mask])
    nmi_y = normalized_mutual_info_score(true_y, y_lab[mask])

    print(f"  → X-cluster purity: {x_p:.3f},  NMI: {nmi_x:.3f}")
    print(f"  → Y-cluster purity: {y_p:.3f},  NMI: {nmi_y:.3f}")

    # wrap into the small DataFrames MT expects
    df_sup = pd.DataFrame({
        'image_coordinates': list(X_sup),
        'ingredient_coordinates' : list(Y_sup)
    })
    df_inf = pd.DataFrame({
        'image_coordinates': list(X_inf),
        'ingredient_coordinates' : list(Y_inf)
    })

    # 4) compute our three baselines
    # — BKM
    mapping   = learn_bridge(x_lab, y_lab, sup_mask, n_clusters)
    centroids = compute_centroids(Y_true, y_lab, n_clusters)
    Yb_all    = predict_bridge(x_lab, mapping, centroids)

    Y_hard_inf = Yb_all[inf_mask]
    Y_true_inf = Y_true[inf_mask]

    # # choose tau heuristically
    # from sklearn.metrics import pairwise_distances
    # cd   = pairwise_distances(centroids)
    tau  = 0.2

    # compute Y_soft for the inference set
    Y_soft_inf = []
    for yc in Yb_all[inf_mask]:
        r = responsibilities(yc, centroids, tau)
        Y_soft_inf.append(r.dot(centroids))
    Y_soft_inf = np.vstack(Y_soft_inf)

    # - GNN-bridge
    Y_gnn = run_gnn_bridged(
        X, y_lab, sup_mask, n_clusters, edge_index, edge_weights, centroids, tau=0.5, hidden_dim=128, lr=3e-3, epochs=47
    )
    Y_gnn_inf = Y_gnn[inf_mask]
    # — KNN
    Yk_inf    = knn_baseline(X_sup, Y_sup, X_inf, k=max(1, int(n_clusters * sup_frac)))

    # — Mean Teacher
    mt_preds, mt_actuals = mean_teacher_regression(df_sup, df_inf)

    # # 3) XGBoost
    # # xgb_preds,   xgb_y = xgboost_regression( df_sup, df_inf )

    # 4) LapRLS
    lap_preds,   lap_y = laprls_regression( df_sup, df_inf )

    # 5) Twin-NN regressor
    tnnr_preds,  tnnr_y = tnnr_regression( df_sup, df_inf )

    # 6) Transductive SVR
    tsvr_preds,  tsvr_y = tsvr_regression( df_sup, df_inf )

    # 7) UCVME
    ucv_preds,   ucv_y  = ucvme_regression( df_sup, df_inf )

    # 8) RankUp
    rank_preds, rank_y  = rankup_regression( df_sup, df_inf )

    # 9) AGDN
    agdn_preds, agdn_y  = agdn_regression( df_sup, df_inf )

    # now compute MAE on each:
    results = {
      "BKM":         mean_absolute_error(Y_true_inf,    Y_hard_inf),
      "BKM_soft":    mean_absolute_error(Y_true_inf,    Y_soft_inf),
      "GNNBridge":   mean_absolute_error(Y_true_inf,    Y_gnn_inf),
      "KNN":         mean_absolute_error(Y_true_inf,    Yk_inf),
      "MeanTeacher": mean_absolute_error(mt_actuals,    mt_preds),
    #   "XGBoost":   mean_absolute_error(xgb_y,         xgb_preds),
      "LapRLS":      mean_absolute_error(lap_y,         lap_preds),
      "TNNR":        mean_absolute_error(tnnr_y,        tnnr_preds),
      "TSVR":        mean_absolute_error(tsvr_y,        tsvr_preds),
      "UCVME":       mean_absolute_error(ucv_y,         ucv_preds),
      "RankUp":      mean_absolute_error(rank_y,        rank_preds),
      "AGDN":        mean_absolute_error(agdn_y,        agdn_preds),
    }

    # # GNN‐bridge
    # Y_gnn = run_gnn_bridged(
    #     X, Y_true, y_lab, mask, n_clusters, edge_index, centroids, tau=0.5
    # )
    # results["GNNBridge"] = mean_absolute_error(Y_true[inf_mask], Y_gnn[inf_mask])

    # # 5) Fuzzy‐bridge (soft)
    # centers_x, u_x, hard_x = fuzzy_cluster(X, n_clusters, m=2.0)
    # # re‑use the SAME Y‑centroids above; membership u_x tells us soft weights
    # # for each sample i, cluster c: u_x[c,i]
    # # so the soft prediction is ∑₍c₌0→K₋₁₎ u_x[c,i] * centroids[c]
    # Y_soft = u_x.T.dot(centroids)

    # results["FuzzyBridge"] = mean_absolute_error(Y_true[inf_mask], Y_soft[inf_mask])

    return results, sup_mask

#############################
# Single‐trial Experiment
#############################
def run_experiment_constrained(df, model, tfm, recipes, ing2idx, vocab_size, X, sup_frac, out_frac, n_clusters, edge_index, edge_weights):
    Y_true = np.vstack(df['ingredient_vec'].values)               # (N, D)
    rp     = GaussianRandomProjection(n_components=Y_DIM_REDUCED, random_state=42)
    Y_proj = rp.fit_transform(Y_true)                             # (N, D′)

    # 1) stratify & split
    x_lab  = cluster_features_constrained(X, n_clusters)
    sup_mask, out_mask, inf_mask = stratified_split_masks(x_lab, sup_frac, out_frac, n_clusters)

    # 2) cluster Y only on sup+out
    y_lab = np.empty(len(Y_proj), int)
    y_lab[sup_mask | out_mask] = cluster_ingredients_constrained(Y_proj[sup_mask | out_mask], n_clusters)

    # 3) prepare supervised/inference sets for Mean Teacher
    X_sup = X[sup_mask];    Y_sup = Y_true[sup_mask]
    X_inf = X[inf_mask];    Y_inf = Y_true[inf_mask]

    true = df['cuisine_type'].values

    # 1) Purity of X-clusters (on *all* samples)
    x_p = purity_score(true, x_lab)
    nmi_x = normalized_mutual_info_score(true, x_lab)

    # 2) Purity of Y-clusters (only on sup+out)
    mask = sup_mask | out_mask
    true_y = true[mask]
    y_p   = purity_score(true_y, y_lab[mask])
    nmi_y = normalized_mutual_info_score(true_y, y_lab[mask])

    print(f"  → X-cluster purity (constrained): {x_p:.3f},  NMI: {nmi_x:.3f}")
    print(f"  → Y-cluster purity (constrained): {y_p:.3f},  NMI: {nmi_y:.3f}")

    # wrap into the small DataFrames MT expects
    df_sup = pd.DataFrame({
        'image_coordinates': list(X_sup),
        'ingredient_coordinates' : list(Y_sup)
    })
    df_inf = pd.DataFrame({
        'image_coordinates': list(X_inf),
        'ingredient_coordinates' : list(Y_inf)
    })

    # 4) compute our three baselines
    # — BKM
    mapping   = learn_bridge(x_lab, y_lab, sup_mask, n_clusters)
    centroids = compute_centroids(Y_true, y_lab, n_clusters)
    Yb_all    = predict_bridge(x_lab, mapping, centroids)

    Y_hard_inf = Yb_all[inf_mask]
    Y_true_inf = Y_true[inf_mask]

    # # choose tau heuristically
    # from sklearn.metrics import pairwise_distances
    # cd   = pairwise_distances(centroids)
    tau  = 0.2

    # compute Y_soft for the inference set
    Y_soft_inf = []
    for yc in Yb_all[inf_mask]:
        r = responsibilities(yc, centroids, tau)
        Y_soft_inf.append(r.dot(centroids))
    Y_soft_inf = np.vstack(Y_soft_inf)

    # - GNN-bridge
    Y_gnn = run_gnn_bridged(
        X, y_lab, sup_mask, n_clusters, edge_index, edge_weights, centroids, tau=0.5, hidden_dim=128, lr=3e-3, epochs=47
    )
    Y_gnn_inf = Y_gnn[inf_mask]
    # — KNN
    Yk_inf    = knn_baseline(X_sup, Y_sup, X_inf, k=max(1, int(n_clusters * sup_frac)))

    # — Mean Teacher
    mt_preds, mt_actuals = mean_teacher_regression(df_sup, df_inf)

    # # 3) XGBoost
    # # xgb_preds,   xgb_y = xgboost_regression( df_sup, df_inf )

    # 4) LapRLS
    lap_preds,   lap_y = laprls_regression( df_sup, df_inf )

    # 5) Twin-NN regressor
    tnnr_preds,  tnnr_y = tnnr_regression( df_sup, df_inf )

    # 6) Transductive SVR
    tsvr_preds,  tsvr_y = tsvr_regression( df_sup, df_inf )

    # 7) UCVME
    ucv_preds,   ucv_y  = ucvme_regression( df_sup, df_inf )

    # 8) RankUp
    rank_preds, rank_y  = rankup_regression( df_sup, df_inf )

    # 9) AGDN
    agdn_preds, agdn_y  = agdn_regression( df_sup, df_inf )

    # now compute MAE on each:
    results = {
      "BC":          mean_absolute_error(Y_true_inf,    Y_hard_inf),
      "Softmax-BC":  mean_absolute_error(Y_true_inf,    Y_soft_inf),
      "GNN-BC":      mean_absolute_error(Y_true_inf,    Y_gnn_inf),
      "KNN":         mean_absolute_error(Y_true_inf,    Yk_inf),
      "MeanTeacher": mean_absolute_error(mt_actuals,    mt_preds),
    #   "XGBoost":   mean_absolute_error(xgb_y,         xgb_preds),
      "LapRLS":      mean_absolute_error(lap_y,         lap_preds),
      "TNNR":        mean_absolute_error(tnnr_y,        tnnr_preds),
      "TSVR":        mean_absolute_error(tsvr_y,        tsvr_preds),
      "UCVME":       mean_absolute_error(ucv_y,         ucv_preds),
      "RankUp":      mean_absolute_error(rank_y,        rank_preds),
      "AGDN":        mean_absolute_error(agdn_y,        agdn_preds),
    }

    # # GNN‐bridge
    # Y_gnn = run_gnn_bridged(
    #     X, Y_true, y_lab, mask, n_clusters, edge_index, centroids, tau=0.5
    # )
    # results["GNNBridge"] = mean_absolute_error(Y_true[inf_mask], Y_gnn[inf_mask])

    # # 5) Fuzzy‐bridge (soft)
    # centers_x, u_x, hard_x = fuzzy_cluster(X, n_clusters, m=2.0)
    # # re‑use the SAME Y‑centroids above; membership u_x tells us soft weights
    # # for each sample i, cluster c: u_x[c,i]
    # # so the soft prediction is ∑₍c₌0→K₋₁₎ u_x[c,i] * centroids[c]
    # Y_soft = u_x.T.dot(centroids)

    # results["FuzzyBridge"] = mean_absolute_error(Y_true[inf_mask], Y_soft[inf_mask])

    return results, sup_mask

#############################
# Multi‐trial Evaluation
#############################
# def run_all_trials(df, model, tfm, recipes, ing2idx, vocab_size, n_clusters, edge_index, X_all):
#     # summary of all the models for this cluster number
#     # 1) initialize an empty DataFrame
#     summary = {}
#     # X       = encode_images(df, IMAGE_FOLDER, model, tfm)
#     X     = X_all

#     model_names = [
#       "BKM",
#       "BKM_soft",
#       "KNN",
#       "MeanTeacher",
#       "XGBoost",
#       "LapRLS",
#       "TSVR",
#       "TNNR",
#       "UCVME",
#       "RankUp",
#       "AGDN",
#       "GNNBridge",

#     ]

#     for sup_frac in SUP_FRACS:
#             # summary keys look like "sup=2.05%, out=55.00%"
#             key    = f"sup={sup_frac:.2%}, out={OUT_FRAC:.2%}"
#             # 2) initialize an empty list for each model
#             errors = { name: [] for name in model_names }

#             print(f"\n=== Supervised {sup_frac:.2%} / Output-only {OUT_FRAC:.2%} ===")
#             # run N_TRIALS trials
#             for t in range(N_TRIALS):
#                 # 3) run your experiment; assume it now returns (results_dict, sup_mask)
#                 results, sup_mask = run_experiment(
#                     df, model, tfm, recipes, ing2idx, vocab_size,
#                     X, sup_frac, OUT_FRAC, n_clusters, edge_index
#                 )
#                 # results is a dict: {"BKM":0.01, "KNN":0.02, ...}

#                 # collect each MAE into its list
#                 for name in model_names:
#                     errors[name].append(results[name])

#                 # print them per‐trial
#                 print(f"--- Trial {t+1} ---")
#                 for name in model_names:
#                     print(f"{name:12s} MAE = {results[name]:.4f}")
#                 print()

#             # 4) compute & print averages
#             print("=== Averages ===")
#             avg_errors = {}
#             for name in model_names:
#                 avg = np.mean(errors[name])
#                 avg_errors[name] = avg
#                 print(f"{name:12s} avg MAE = {avg:.4f}")

#             # stash into summary
#             summary[key] = { f"{name}_MAE": avg for name, avg in avg_errors.items() }

#     return summary

def collect_mae_trials(df, model, tfm, recipes, ing2idx, D,
                       cluster_vals, sup_fracs, out_frac, n_trials):
    """
    Runs `run_experiment` n_trials times for every (n_clusters, sup_frac) combo,
    returns:
      - loss_data: shape (len(cluster_vals), len(sup_fracs), num_models, n_trials)
      - model_names: the list of model keys, in the order used in loss_data
    """
    model_names = [
      "BC",
      "Softmax-BC",
      "GNN-BC",
      "KNN",
      "MeanTeacher",
    #   "XGBoost",
      "LapRLS",
      "TSVR",
      "TNNR",
      "UCVME",
      "RankUp",
      "AGDN",
    ]
    num_h1 = len(cluster_vals)
    num_h2 = len(sup_fracs)
    num_m  = len(model_names)
    # loss_data = np.zeros((num_h1, num_h2, num_m, n_trials), float)
    loss_data_constrained = np.zeros((num_h1, num_h2, num_m, n_trials), float)

    for i, n_clusters in enumerate(cluster_vals):
        print(f"\n=== RUNNING for {n_clusters} clusters ===")
        # pick cuisines at random (or deterministic if you prefer)
        all_cuisines = df['cuisine_type'].unique()
        cuisines = np.random.choice(all_cuisines, size=n_clusters, replace=False).tolist()
        df_run = df[df['cuisine_type'].isin(cuisines)].reset_index(drop=True)
        print(f"→ Using cuisines for {n_clusters} clusters:", cuisines)
        # pre‐encode once per df_run
        X_all = encode_images(df_run, IMAGE_FOLDER, model, tfm)
        # build your GNN graph once
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=22, algorithm="auto").fit(X_all)
        adj = knn.kneighbors_graph(X_all, mode="connectivity").tocoo()
        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        edge_weights = None
        # edge_weights = None  # not used in this experiment
        # sigma = adj.data.mean()
        # sim   = np.exp(- (adj.data**2) / (2 * sigma**2))
        # edge_weights = torch.tensor(sim, dtype=torch.float)
        # print(f"→ Edge index shape: {edge_index.shape}, weights: {edge_weights is not None}")

        for j, sup_frac in enumerate(sup_fracs):
            print(f"\n-- Supervision = {sup_frac:.2%} --")
            for t in range(n_trials):
                print(f" Trial {t+1}/{n_trials}:")
                # results, _ = run_experiment(
                #     df_run, model, tfm, recipes, ing2idx, D,
                #     X_all, sup_frac, out_frac, n_clusters, edge_index, edge_weights
                # )
                # constrained version
                results_constrained, _ = run_experiment_constrained(
                    df_run, model, tfm, recipes, ing2idx, D,
                    X_all, sup_frac, out_frac, n_clusters, edge_index, edge_weights
                )

                for m, name in enumerate(model_names):
                    # mae = results[name]
                    mae_constrained = results_constrained[name]
                    # loss_data[i, j, m, t] = mae
                    loss_data_constrained[i, j, m, t] = mae_constrained
                    # print(f"    {name:12s} MAE = {mae:.4f} (constrained: {mae_constrained:.4f})")
                    print(f"    {name:12s} MAE (constrained) = {mae_constrained:.4f}")
                    
            # after all trials for this sup_frac, print the averages
            print(f" → AVERAGES for {sup_frac:.2%}:")
            for m, name in enumerate(model_names):
                # avg = loss_data[i,j,m,:].mean()
                avg_constrained = loss_data_constrained[i,j,m,:].mean()
                # print(f"    {name:12s} avg MAE = {avg:.4f} (constrained: {avg_constrained:.4f})")
                print(f"    {name:12s} avg MAE (constrained) = {avg_constrained:.4f}")

        print(f"\n=== SUMMARY TABLE for {n_clusters} clusters ===")
        # shape (sup_fracs × models)
        # rows = []
        rows_constrained = []
        for j, sup_frac in enumerate(sup_fracs):
            # row = {'sup_frac': f"{sup_frac:.2%}"}
            row_constrained = {'sup_frac': f"{sup_frac:.2%} (constrained)"}
            for m, name in enumerate(model_names):
                # row[name] = loss_data[i,j,m,:].mean()
                row_constrained[name] = loss_data_constrained[i,j,m,:].mean()
            # rows.append(row)
            rows_constrained.append(row_constrained)
        # summary_df = pd.DataFrame(rows).set_index('sup_frac')
        summary_df_constrained = pd.DataFrame(rows_constrained).set_index('sup_frac')
        # print(summary_df)
        print(summary_df_constrained)
        output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    for j, sup_frac in enumerate(sup_fracs):
        # build a dict of “cluster size → average MAE per method”
        table_data = {
            str(c): loss_data_constrained[i, j].mean(axis=1)
            for i, c in enumerate(cluster_vals)
        }
        df = pd.DataFrame(table_data, index=model_names)
        df.insert(0, "Method", df.index)  # add the “Method” column
        for col in df.columns.drop("Method"):
            df[col] = df[col].map(lambda x: f"{x:.6f}")
        # render with matplotlib
        fig, ax = plt.subplots(
            figsize=(len(cluster_vals)*1.2 + 2, len(model_names)*0.4 + 1)
        )
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        fig.tight_layout()
        
        # save it
        pct = int(sup_frac * 100)
        fname = os.path.join(output_dir, f"10trials_summary_supervised_{pct}pct.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # also print it out here in the notebook
        print(f"\nSupervision = {pct}%")
        print(df)

    return loss_data_constrained, model_names

#############################
# Plot
#############################
# def bubble_plot(df_results,
#                             model_order=None,
#                             desired_max_area=2000,
#                             offset_radius=0.15,
#                             alpha=3.0):
#     """
#     Exponential‐scaled bubble plot:
#       size ∝ (exp(alpha*norm_mae) - 1)/(exp(alpha) - 1) * desired_max_area
#     where norm_mae = (MAE - min_mae)/(max_mae - min_mae).
#     """
#     xf = sorted(df_results['sup_frac'].unique())
#     yf = sorted(df_results['n_clusters'].unique())
#     if model_order is None:
#         model_order = df_results['model'].unique()
#     colors = plt.cm.tab10(np.linspace(0,1,len(model_order)))

#     # normalize MAE to [0,1]
#     maes = df_results["MAE"].values
#     min_mae, max_mae = maes.min(), maes.max()
#     df_results = df_results.assign(
#         norm_mae = (df_results["MAE"] - min_mae) / (max_mae - min_mae)
#     )

#     fig, ax = plt.subplots(figsize=(8,6))
#     for mi, model_name in enumerate(model_order):
#         sub = df_results[df_results['model'] == model_name]
#         xs, ys, ss = [], [], []
#         for _, row in sub.iterrows():
#             x0 = xf.index(row.sup_frac)
#             y0 = yf.index(row.n_clusters)
#             angle = 2*np.pi * mi/len(model_order)
#             dx, dy = offset_radius*np.cos(angle), offset_radius*np.sin(angle)
#             xs.append(x0 + dx)
#             ys.append(y0 + dy)

#             # exponential scaling
#             e = np.exp(alpha * row.norm_mae) - 1
#             scale = (np.exp(alpha) - 1)
#             area = (e / scale) * desired_max_area
#             ss.append(area)

#         ax.scatter(xs, ys, s=ss,
#                    color=colors[mi], alpha=0.7,
#                    label=model_name, edgecolor='k', linewidth=0.5)

#     # axis formatting
#     ax.set_xticks(range(len(xf)))
#     ax.set_xticklabels([f"{f*100:.2f}%" for f in xf])
#     ax.set_yticks(range(len(yf)))
#     ax.set_yticklabels([str(c) for c in yf])
#     ax.set_xlabel("Supervision fraction")
#     ax.set_ylabel("Number of clusters")
#     ax.set_title("MAE of models over clusters & supervision\n(exponential bubble‐size scaling)")
#     ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
#     plt.tight_layout()
#     plt.show()

# def bubble_plot_for_fraction(df_subset,
#                              model_order=None,
#                              desired_max_area=2000,
#                              offset_radius=0.15,
#                              alpha=3.0):
#     """
#     Plot just one sup_frac slice of your results.
#     df_subset: filtered DataFrame where all rows share the same sup_frac.
#     """
#     # extract the single fraction and cast to string for the title
#     sup_frac = df_subset['sup_frac'].iloc[0]
#     title_frac = f"{sup_frac*100:.2f}% supervision"

#     # keep n_clusters variety on y
#     yf = sorted(df_subset['n_clusters'].unique())
#     model_order = model_order or df_subset['model'].unique()
#     colors = plt.cm.tab10(np.linspace(0,1,len(model_order)))

#     # normalize MAE to [0,1] *within this slice*
#     maes = df_subset["MAE"].values
#     min_mae, max_mae = maes.min(), maes.max()
#     df_subset = df_subset.assign(
#         norm_mae = (df_subset["MAE"] - min_mae) / (max_mae - min_mae + 1e-12)
#     )

#     fig, ax = plt.subplots(figsize=(6,4))
#     for mi, model_name in enumerate(model_order):
#         sub = df_subset[df_subset['model'] == model_name]
#         xs, ys, ss = [], [], []
#         for _, row in sub.iterrows():
#             # x fixed at 0 (only one column)
#             x0 = 0
#             # y index of cluster count
#             y0 = yf.index(row.n_clusters)
#             angle = 2*np.pi * mi/len(model_order)
#             dx, dy = offset_radius*np.cos(angle), offset_radius*np.sin(angle)
#             xs.append(x0 + dx)
#             ys.append(y0 + dy)

#             # exponential sizing
#             e = np.exp(alpha * row.norm_mae) - 1
#             scale = (np.exp(alpha) - 1)
#             area = (e / scale) * desired_max_area
#             ss.append(area)

#         ax.scatter(xs, ys, s=ss,
#                    color=colors[mi], alpha=0.7,
#                    label=model_name, edgecolor='k', linewidth=0.5)

#     # axis formatting: only one x‐label
#     ax.set_xticks([0])
#     ax.set_xticklabels([title_frac])
#     ax.set_yticks(range(len(yf)))
#     ax.set_yticklabels([str(c) for c in yf])
#     ax.set_xlabel("Supervision fraction")
#     ax.set_ylabel("Number of clusters")
#     ax.set_title(f"MAE bubbles @ {title_frac}")
#     ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small", ncol=1)
#     plt.tight_layout()
#     plt.show()


# def plot_one_per_fraction(df_results, model_order=None,
#                           desired_max_area=2000,
#                           offset_radius=0.15,
#                           alpha=3.0):
#     """
#     Loop over each sup_frac and plot its own bubble‐chart.
#     """
#     for sup_frac in sorted(df_results['sup_frac'].unique()):
#         subset = df_results[df_results['sup_frac'] == sup_frac]
#         bubble_plot_for_fraction(
#             subset,
#             model_order=model_order,
#             desired_max_area=desired_max_area,
#             offset_radius=offset_radius,
#             alpha=alpha
#         )

# def run_and_plot_all(df, model, tfm, recipes, ing2idx, D):
#     all_rows = []
#     # if want to do randomization:
#     all_cuisines = df['cuisine_type'].unique()
#     for n_clusters in N_CLUSTERS:
#         # select exactly the cuisines you want for this number of clusters
#         # determinstic
#         cuisines = []
#         # if n_clusters == 3:
#         #     cuisines = ['beef_tacos', 'pizza', 'ramen']
#         # elif n_clusters == 4:
#         #     cuisines = ['beef_tacos', 'pizza', 'ramen', 'apple_pie']
#         # else:  # n_clusters == 5
#         #     cuisines = ['beef_tacos', 'pizza', 'ramen', 'apple_pie', 'strawberry_shortcake']
#         # or randomize
#         cuisines = np.random.choice(all_cuisines, size=n_clusters, replace=False).tolist()
#         print(f"→ Using cuisines for {n_clusters} clusters:", cuisines)

#         # create a fresh DataFrame for this run
#         df_run = df[df['cuisine_type'].isin(cuisines)].reset_index(drop=True)

#         X_all = encode_images(df_run, IMAGE_FOLDER, model, tfm)
#         knn = NearestNeighbors(n_neighbors=10, algorithm="auto").fit(X_all)
#         adj = knn.kneighbors_graph(X_all, mode="connectivity").tocoo()
#         edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        
#         print(f"Running with {n_clusters} clusters on {len(df_run)} recipes")

#         summary = run_all_trials(df_run, model, tfm, recipes, ing2idx, D, n_clusters, edge_index, X_all)
        
#         print(f"\n=== Summary for all proportions ({n_clusters} clusters) ===")
#         print(pd.DataFrame(summary).T)
#         # summary keys look like "sup=2.05%, out=55.00%"
#         for key, metrics in summary.items():
#             sup_frac = float(key.split(",")[0].split("=")[1].strip("%"))/100
#             for metric_name, mae in metrics.items():
#                 all_rows.append({
#                     "n_clusters": n_clusters,
#                     "sup_frac":   sup_frac,
#                     "model":      metric_name.replace("_MAE",""),
#                     "MAE":        mae
#                 })

#     df_results = pd.DataFrame(all_rows)
#     plot_one_per_fraction(df_results,
#                 model_order=["BKM","BKM_soft","GNNBridge","KNN","MeanTeacher","XGBoost","LapRLS","TSVR","TNNR","UCVME","RankUp","AGDN"])

import numpy as np
import matplotlib.pyplot as plt

def plot_boxplots_by_hyper1(loss_data: np.ndarray,
                            hyperparam1_labels=["30", "60", "90", "120", "150"],
                            hyperparam2_labels=["1% supervision", "5% supervision", "10% supervision", "15% supervision"],
                            model_labels=["Bridged Clustering", "KNN", "Mean Teacher", "GCN", "AGDN", "Bridged AGDN"],
                            log_scale=False):
    """
    Create a figure with 5 subplots (one for each value of hyperparameter 1).
    In each subplot the loss distributions are shown as grouped boxplots for the
    4 hyperparameter 2 configurations. Within each group, the 5 boxplots correspond
    to the 5 models.

    Parameters:
      - loss_data: NumPy array of shape (1, 5, 4, 5, 20) or (5, 4, 5, 20).
                   Each entry represents the loss for a particular trial.
      - hyperparam1_labels: List of labels for hyperparameter 1 (default: "H1-0", "H1-1", ...).
      - hyperparam2_labels: List of labels for hyperparameter 2 (default: "H2-0", "H2-1", ...).
      - model_labels: List of labels for models (default: "Model 1", "Model 2", ...).
      - log_scale: Boolean, if True the y-axis will be set to a logarithmic scale.
    """
    # # Remove the wrapper dimension, if present.
    # if loss_data.shape[0] == 1:
    #     loss_data = loss_data[0]  # Now shape should be (5, 4, 5, 20)
        
    # Unpack shape
    num_h1, num_h2, num_models, num_trials = loss_data.shape  # 5,4,5,20

    # Set default labels if needed.
    if hyperparam1_labels is None:
        hyperparam1_labels = [f'H1-{i}' for i in range(num_h1)]
    if hyperparam2_labels is None:
        hyperparam2_labels = [f'H2-{i}' for i in range(num_h2)]
    if model_labels is None:
        model_labels = [f'Model {i+1}' for i in range(num_models)]
    
    # Define a color for each model (consistent across all subplots)
    colors = [
        'lightblue', 
        'lightgreen', 
        'salmon', 
        'violet', 
        'wheat', 
        'lightgrey', 
        'lightpink', 
        'lightyellow', 
        'lightcyan', 
        'lightcoral', 
        'lightsteelblue', 
        'lightgoldenrodyellow', 
        'lightseagreen', 
        'lightsalmon', 
        'lightblue', 
        'lightgreen'
    ]

    # Create the figure with one subplot per hyperparameter 1 value.
    fig, axes = plt.subplots(nrows=num_h1, ncols=1, figsize=(12, 4 * num_h1), sharex=False)
    
    # In case there is only one subplot (num_h1==1), force axes to be iterable.
    if num_h1 == 1:
        axes = [axes]

    # Loop over hyperparameter 1 values, each gets its own subplot.
    for i in range(num_h1):
        ax = axes[i]
        data_all = []    # to collect the loss arrays for each group/model
        positions = []   # the x positions at which to put each boxplot
        group_centers = []  # to mark center of each hyperparam2 group for labeling
        
        gap = 1  # gap between groups (in x-axis units)
        # Loop over the 4 hyperparameter 2 values for this hyperparameter 1 configuration.
        for j in range(num_h2):
            base = j * (num_models + gap)  # starting position for this group
            # positions for each of the 5 models within this group.
            pos = list(range(base + 1, base + 1 + num_models))
            # For each model in this hyper 2 group, extract the 20 loss values.
            for m in range(num_models):
                data_all.append(loss_data[i, j, m, :])
                positions.append(pos[m])
            group_centers.append(np.mean(pos))
        
        # Create the boxplots for this subplot.
        bp = ax.boxplot(data_all, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True)
        
        # Color each box based on its model. Since for each group the boxes
        # appear in order (model 0 through model 4), we use modulo arithmetic.
        for idx, box in enumerate(bp['boxes']):
            model_idx = idx % num_models  # cycles from 0 to 4
            box.set(facecolor=colors[model_idx])
        
        # Set the x-axis ticks at the centers of each hyperparam2 group.
        ax.set_xticks(group_centers)
        ax.set_xticklabels(hyperparam2_labels, fontsize=10)
        ax.set_ylabel("MAE", fontsize=12)
        ax.set_title(f"MAE Distributions for {hyperparam1_labels[i]}", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
    
    # Create a custom legend for the models (appears outside the subplots)
    # from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[i], edgecolor='k', label=model_labels[i])
                      for i in range(num_models)]
    # Place legend to the right of the plots.
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space on the right for the legend
    plt.show()


#############################
# Fine-Tuning
#############################
def train_and_eval_gnn(
    X, y_lab, sup_mask,
    edge_index,
    centroids,
    n_clusters=5,
    tau=0.5,
    hidden_dim=64,
    lr=0.01,
    epochs=1000
):
    # 1) train the GNN
    Y_gnn = run_gnn_bridged(
        X, y_lab, sup_mask, n_clusters,
        edge_index, centroids,
        tau=tau,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs
    )

    return Y_gnn

def eval_one_config(
    X, Y_true, Y_proj, x_lab, y_lab,
    n_clusters, sup_frac, out_frac,
    edge_index, tau,
    hidden_dim, lr, epochs
):
    # replicate exactly your stratified / clustering / centroids logic
    sup_mask, out_mask, inf_mask = stratified_split_masks(
        x_lab, sup_frac, out_frac, n_clusters
    )
    # cluster Y on sup+out
    y_lab[sup_mask|out_mask] = cluster_ingredients(
        Y_proj[sup_mask|out_mask], n_clusters
    )
    centroids = compute_centroids(Y_true, y_lab, n_clusters)

    # train & eval the GNN for this config
    Y_pred = train_and_eval_gnn(
        X, y_lab, sup_mask,
        edge_index,
        centroids,
        n_clusters=n_clusters,
        tau=tau,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs
    )
    return mean_absolute_error(Y_true[inf_mask], Y_pred[inf_mask])

def find_general_gnn_hyperparams(
    df,                    # your DataFrame for one dataset
    param_grid,                # dict of lists for tau, hidden_dim, lr, epochs
    model, tfm, recipes, ing2idx, D,
        cluster_vals, sup_fracs, out_frac, n_trials
):

    best = None
    for tau, hidden_dim, lr, n_neighbors, epochs in itertools.product(
        param_grid['tau'],
        param_grid['hidden_dim'],
        param_grid['lr'],
        param_grid['n_neighbors'],
        param_grid['epochs']
    ):
        maes = []
        print(f"\n=== RUNNING for tau={tau}, hidden_dim={hidden_dim}, lr={lr}, n_neighbors={n_neighbors}, epochs={epochs} ===")
        # pick cuisines at random (or deterministic if you prefer)
        for i, n_clusters in enumerate(cluster_vals):
            print(f"\n=== RUNNING for {n_clusters} clusters ===")
            # pick cuisines at random (or deterministic if you prefer)
            all_cuisines = df['cuisine_type'].unique()
            cuisines = np.random.choice(all_cuisines, size=n_clusters, replace=False).tolist()
            df_run = df[df['cuisine_type'].isin(cuisines)].reset_index(drop=True)
            print(f"→ Using cuisines for {n_clusters} clusters:", cuisines)

            # pre‐encode once per df_run
            X_all = encode_images(df_run, IMAGE_FOLDER, model, tfm)
            # build your GNN graph once
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(X_all)
            adj = knn.kneighbors_graph(X_all, mode="connectivity").tocoo()
            edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)

            Y_true = np.vstack(df_run['ingredient_vec'].values)               # (N, D)
            rp     = GaussianRandomProjection(n_components=Y_DIM_REDUCED, random_state=42)
            Y_proj = rp.fit_transform(Y_true)                             # (N, D′)
            y_lab  = np.empty(len(Y_proj), int)
            # 1) stratify & split
            x_lab  = cluster_features(X_all, n_clusters)

            for j, sup_frac in enumerate(sup_fracs):
                print(f"\n-- Supervision = {sup_frac:.2%} --")
                mae = eval_one_config(
                    X_all, Y_true,
                    Y_proj, x_lab, y_lab,
                    n_clusters=n_clusters,
                    sup_frac=sup_frac,
                    out_frac=OUT_FRAC,
                    edge_index=edge_index,
                    tau=tau,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    epochs=epochs
                )
                maes.append(mae)
                print(f"    MAE = {mae:.4f}")
        avg_mae = np.mean(maes)
        print(f"→ AVERAGE MAE = {avg_mae:.4f}")
        if best is None or avg_mae < best[0]:
            best = (avg_mae, dict(
                tau=tau, hidden_dim=hidden_dim,
                lr=lr, n_neighbors=n_neighbors, epochs=epochs
            ))

    return best  # (best_avg_mae, best_config)
    


#############################
# Main
#############################
if __name__ == '__main__':
    vocab, ing2idx = load_ingredients()
    recipes        = load_recipes()
    D              = len(vocab)

    imgs = load_paths(ALL_IMAGES_FILE)
    df   = pd.DataFrame({'img_path':imgs})
    df['recipe_id']    = pd.read_csv(ALL_LABELS_FILE,header=None).iloc[:,0]
    df['ingredient_vec'] = df['recipe_id'].apply(lambda r: make_multihot(r, recipes, ing2idx, D))
    df['cuisine_type']  = df['img_path'].str.split('/').str[0]
    # df = df[df['cuisine_type'].isin({'beef_tacos','pizza','ramen'})].reset_index(drop=True)

    model, tfm = get_image_encoder()
    # summary = run_all_trials(df, model, tfm, recipes, ing2idx, D)
    # print("\n=== Summary for all proportions ===")
    # print(pd.DataFrame(summary).T)

     # collect
    cluster_vals = N_CLUSTERS              # [3,4,5]
    sup_fracs    = SUP_FRACS               # [0.0205,0.05,0.1]
    out_frac     = OUT_FRAC
    n_trials     = N_TRIALS                # e.g. 3 or bump to 20 for nicer boxes

    loss_data, model_labels = collect_mae_trials(
        df, model, tfm, recipes, ing2idx, D,
        cluster_vals, sup_fracs, out_frac, n_trials
    )

    # labels for the axes
    hyperparam1_labels = [f"{c} clusters"    for c in cluster_vals]
    hyperparam2_labels = [f"{int(s*100)}% sup" for s in sup_fracs]
    print(hyperparam1_labels)
    print(hyperparam2_labels)

    # finally, plot
    # plot_boxplots_by_hyper1(
    #     loss_data,
    #     hyperparam1_labels=hyperparam1_labels,
    #     hyperparam2_labels=hyperparam2_labels,
    #     model_labels=model_labels,
    #     log_scale=False
    # )
    # run_and_plot_all(df, model, tfm, recipes, ing2idx, D)
    # param_grid = {
    # 'tau':        [0.4, 0.5, 0.75],
    # 'hidden_dim': [64, 128, 256],
    # 'lr':         [1e-4, 5e-4, 1e-5],
    # 'n_neighbors': [5, 10, 20],
    # 'epochs':     [100]
    # }
    # best_mae, best_cfg = find_general_gnn_hyperparams(df, param_grid, model, tfm, recipes, ing2idx, D,
    #     cluster_vals, sup_fracs, out_frac, n_trials)
    # print("GENERAL GNN hyperparams →", best_cfg, "avg MAE =", best_mae)

import os
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsRegressor
from sklearn.random_projection import GaussianRandomProjection
import torch
import torchvision
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.optim import Adam
import torch_geometric.utils
from tqdm import tqdm
from collections import Counter

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
N_CLUSTERS      = 3
SUP_FRACS       = [0.0205, 0.05, 0.1]
OUT_FRAC        = 0.55    # fraction of "output-only" samples for Y-clustering
N_TRIALS        = 3
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
def stratified_split_masks(x_lab, sup_frac, out_frac):
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
    n_sup = max(N_CLUSTERS, int(sup_frac * N))

    # 2) Pick one supervised index from each cluster to ensure coverage
    sup_idx = []
    for c in range(N_CLUSTERS):
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
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

def cluster_ingredients(Y_proj, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(Y_proj)

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

#############################
# Baseline: KNN
#############################
def knn_baseline(X_train, Y_train, X_test, k=3):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)
    return knn.predict(X_test)

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
def run_experiment(df, model, tfm, recipes, ing2idx, vocab_size, X, sup_frac, out_frac):
    Y_true = np.vstack(df['ingredient_vec'].values)               # (N, D)
    rp     = GaussianRandomProjection(n_components=Y_DIM_REDUCED, random_state=42)
    Y_proj = rp.fit_transform(Y_true)                             # (N, D′)

    # 1) stratify & split
    x_lab  = cluster_features(X, N_CLUSTERS)
    sup_mask, out_mask, inf_mask = stratified_split_masks(x_lab, sup_frac, out_frac)

    # 2) cluster Y only on sup+out
    y_lab = np.empty(len(Y_proj), int)
    y_lab[sup_mask | out_mask] = cluster_ingredients(Y_proj[sup_mask | out_mask], N_CLUSTERS)

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
    mapping   = learn_bridge(x_lab, y_lab, sup_mask, N_CLUSTERS)
    centroids = compute_centroids(Y_true, y_lab, N_CLUSTERS)
    Yb_all    = predict_bridge(x_lab, mapping, centroids)
    # bkm_mae   = mean_absolute_error(Y_inf, Yb_all[inf_mask])

    # — KNN
    Yk_inf    = knn_baseline(X_sup, Y_sup, X_inf)
    # knn_mae   = mean_absolute_error(Y_inf, Yk_inf)

    # — Mean Teacher
    mt_preds, mt_actuals = mean_teacher_regression(df_sup, df_inf)

    # 3) XGBoost
    xgb_preds,   xgb_y = xgboost_regression( df_sup, df_inf )

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
      "BKM":       mean_absolute_error(Y_inf,    Yb_all[inf_mask]),
      "KNN":       mean_absolute_error(Y_true[inf_mask],    Yk_inf),
      "MeanTeacher": mean_absolute_error(mt_actuals,    mt_preds),
      "XGBoost":   mean_absolute_error(xgb_y,   xgb_preds),
      "LapRLS":    mean_absolute_error(lap_y,   lap_preds),
      "TNNR":      mean_absolute_error(tnnr_y, tnnr_preds),
      "TSVR":      mean_absolute_error(tsvr_y, tsvr_preds),
      "UCVME":     mean_absolute_error(ucv_y,  ucv_preds),
      "RankUp":    mean_absolute_error(rank_y, rank_preds),
      "AGDN":      mean_absolute_error(agdn_y,agdn_preds),
    }



    return results, sup_mask

#############################
# Multi‐trial Evaluation
#############################
def run_all_trials(df, model, tfm, recipes, ing2idx, vocab_size):
    summary = {}
    X       = encode_images(df, IMAGE_FOLDER, model, tfm)

    model_names = [
      "BKM",
      "KNN",
      "MeanTeacher",
      "XGBoost",
      "LapRLS",
      "TSVR",
      "TNNR",
      "UCVME",
      "RankUp",
      "AGDN",
    ]

    for sup_frac in SUP_FRACS:
            key    = f"sup={sup_frac:.2%}, out={OUT_FRAC:.2%}"
            # 2) initialize an empty list for each model
            errors = { name: [] for name in model_names }

            print(f"\n=== Supervised {sup_frac:.2%} / Output-only {OUT_FRAC:.2%} ===")
            for t in range(N_TRIALS):
                # 3) run your experiment; assume it now returns (results_dict, sup_mask)
                results, sup_mask = run_experiment(
                    df, model, tfm, recipes, ing2idx, vocab_size,
                    X, sup_frac, OUT_FRAC
                )
                # results is a dict: {"BKM":0.01, "KNN":0.02, ...}

                # collect each MAE into its list
                for name in model_names:
                    errors[name].append(results[name])

                # print them per‐trial
                print(f"--- Trial {t+1} ---")
                for name in model_names:
                    print(f"{name:12s} MAE = {results[name]:.4f}")
                print()

            # 4) compute & print averages
            print("=== Averages ===")
            avg_errors = {}
            for name in model_names:
                avg = np.mean(errors[name])
                avg_errors[name] = avg
                print(f"{name:12s} avg MAE = {avg:.4f}")

            # stash into summary
            summary[key] = { f"{name}_MAE": avg for name, avg in avg_errors.items() }

    return summary

#############################
# Main
#############################
def bubble_plot(df_results,
                cluster_labels=[3,4,5],
                sup_labels=["2.05%","5%","10%"],
                model_order=None,
                bubble_scale=2000,
                offset_radius=0.15):
    """
    df_results: DataFrame with columns
      ['n_clusters','sup_frac','model','MAE']
    """
    # convert sup_frac back to % strings for tick labels
    xf = sorted(df_results['sup_frac'].unique())
    yf = sorted(df_results['n_clusters'].unique())
    if model_order is None:
        model_order = df_results['model'].unique()

    # prepare color map
    colors = plt.cm.tab10(np.linspace(0,1,len(model_order)))

    fig, ax = plt.subplots(figsize=(8,6))
    # for each model, plot its bubbles
    for mi, model_name in enumerate(model_order):
        sub = df_results[df_results['model']==model_name]
        xs, ys, ss = [], [], []
        for _, row in sub.iterrows():
            # base coords: index into lists
            x0 = xf.index(row.sup_frac)
            y0 = yf.index(row.n_clusters)
            # small radial offset
            angle = 2*np.pi * mi/len(model_order)
            dx = offset_radius*np.cos(angle)
            dy = offset_radius*np.sin(angle)
            xs.append(x0 + dx)
            ys.append(y0 + dy)
            # scale MAE -> bubble area
            ss.append(row.MAE * bubble_scale)
        ax.scatter(xs, ys, s=ss, color=colors[mi],
                   alpha=0.7, label=model_name, edgecolor='k', linewidth=0.5)

    # axis ticks
    ax.set_xticks(range(len(xf)))
    ax.set_xticklabels([f"{int(f*100)}%" for f in xf])
    ax.set_yticks(range(len(yf)))
    ax.set_yticklabels([str(c) for c in yf])
    ax.set_xlabel("Supervision fraction")
    ax.set_ylabel("Number of clusters")
    ax.set_title("MAE of models over clusters & supervision")
    ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()

def run_and_plot_all(df, model, tfm, recipes, ing2idx, D):
    all_rows = []
    for n_clusters in [3,4,5]:
        # override global for each run
        global N_CLUSTERS
        N_CLUSTERS = n_clusters
        if n_clusters == 3:
            df = df[df['cuisine_type'].isin({'beef_tacos','pizza','ramen'})].reset_index(drop=True)
        elif n_clusters == 4:
            df = df[df['cuisine_type'].isin({'beef_tacos','pizza','ramen', 'apple_pie'})].reset_index(drop=True)
        else:
            df = df[df['cuisine_type'].isin({'beef_tacos','pizza','ramen', 'apple_pie', 'strawberry_shortcake'})].reset_index(drop=True)
        summary = run_all_trials(df, model, tfm, recipes, ing2idx, D)
        print(f"\n=== Summary for all proportions ({n_clusters} clusters) ===")
        print(pd.DataFrame(summary).T)
        # summary keys look like "sup=2.05%, out=55.00%"
        for key, metrics in summary.items():
            sup_frac = float(key.split(",")[0].split("=")[1].strip("%"))/100
            for metric_name, mae in metrics.items():
                all_rows.append({
                    "n_clusters": n_clusters,
                    "sup_frac":   sup_frac,
                    "model":      metric_name.replace("_MAE",""),
                    "MAE":        mae
                })

    df_results = pd.DataFrame(all_rows)
    bubble_plot(df_results,
                cluster_labels=[3,4,5],
                sup_labels=["2.05%","5%","10%"],
                model_order=["BKM","KNN","MeanTeacher","XGBoost","LapRLS","TSVR","TNNR","UCVME","RankUp","AGDN"])
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
    run_and_plot_all(df, model, tfm, recipes, ing2idx, D)

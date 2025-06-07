# Bridged Clustering for WikiArt: Predicting Year from Paintings using Style as Latent Space

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from torchvision import models, transforms
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
from sklearn.manifold import TSNE
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Parameters
N_CLUSTERS = [2]  #, 3, 4, 5, 6] 
SUPERVISED_VALUES = [1] #, 3, 5, 10]
N_TRIALS = 10
IMAGE_SIZE = 224

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

def encode_images(df, image_folder):
    features = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(image_folder, row['style'].replace(" ", "_"), row['filename'])
        if not os.path.exists(path):
            continue
        try:
            image = Image.open(path).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                feat = resnet(image).squeeze().numpy()
            features.append(feat)
            valid_indices.append(idx)
        except:
            continue
    return np.array(features), valid_indices

from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import dense_to_sparse

class GNNClusterBridge(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_clusters, tau=1.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_clusters)
        self.tau = tau

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        logits = self.classifier(h)
        return F.softmax(logits / self.tau, dim=-1)

def run_gnn_bridged(
    X, y_lab, sup_mask, n_clusters,
    edge_index, edge_weights,
    centroids, tau=1.0,
    hidden_dim=64, lr=1e-3, epochs=100
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y_lab, dtype=torch.long, device=device)
    mask = torch.tensor(sup_mask, dtype=torch.bool, device=device)

    model = GNNClusterBridge(x.shape[1], hidden_dim, n_clusters, tau).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[mask], y[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)  # shape (N, K)
        preds = out @ torch.tensor(centroids, dtype=torch.float32, device=device)  # (N, 1)
    return preds.cpu().numpy()

from itertools import product
from sklearn.metrics import mean_absolute_error

def grid_search_gnn(
    X, y_lab, sup_mask,
    n_clusters, edge_index, centroids,
    test_indices, true_years, 
    hidden_dims=[64], taus=[1.0], lrs=[1e-3], epochs=100
):
    results = []
    best_mae = float('inf')
    best_config = None
    best_preds = None

    for hidden_dim, tau, lr in product(hidden_dims, taus, lrs):
        print(f"Running GNN with hidden_dim={hidden_dim}, tau={tau}, lr={lr}")
        preds = run_gnn_bridged(
            X=X,
            y_lab=y_lab,
            sup_mask=sup_mask,
            n_clusters=n_clusters,
            edge_index=edge_index,
            edge_weights=None,
            centroids=centroids.reshape(-1, 1),
            tau=tau,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs
        )
        pred_vals = preds[test_indices].flatten()
        mae = mean_absolute_error(true_years, pred_vals)

        print(f"  → MAE: {mae:.4f}")
        results.append(((hidden_dim, tau, lr), mae))

        if mae < best_mae:
            best_mae = mae
            best_config = (hidden_dim, tau, lr)
            best_preds = pred_vals

    print("\n Best config:")
    print(f"  hidden_dim={best_config[0]}, tau={best_config[1]}, lr={best_config[2]}, MAE={best_mae:.4f}")
    return best_config, best_mae, best_preds, results


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

def mean_teacher_regression(supervised_df, inference_df, feature_col='image_coordinates', label_col='year'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(supervised_df[feature_col].iloc[0])
    output_dim = 1  # year is scalar

    model = MeanTeacherModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Labeled data loader
    X_train = torch.tensor(np.vstack(supervised_df[feature_col]), dtype=torch.float32)
    y_train = torch.tensor(supervised_df[label_col].values.reshape(-1, 1), dtype=torch.float32)
    labeled_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)

    # Unlabeled inputs
    X_unlabeled = torch.tensor(np.vstack(inference_df[feature_col]), dtype=torch.float32)
    dummy_labels = torch.zeros_like(X_unlabeled[:, :1])  # Dummy labels
    unlabeled_loader = torch.utils.data.DataLoader(list(zip(X_unlabeled, dummy_labels)), batch_size=32, shuffle=True)

    # Test loader (same as inference)
    y_test = torch.tensor(inference_df[label_col].values.reshape(-1, 1), dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(list(zip(X_unlabeled, y_test)), batch_size=32)

    for epoch in range(100):
        train_mean_teacher(model, labeled_loader, unlabeled_loader, optimizer, device)

    preds, actuals = evaluate_mean_teacher(model, test_loader, device)
    return preds.flatten(), actuals.flatten()

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

# def xgboost_regression(sup_df, inf_df,
#                        **xgb_params):               # e.g. n_estimators=200, max_depth=6
#     if not xgb_params:                              # sensible defaults
#         xgb_params = dict(n_estimators=300,
#                           learning_rate=0.05,
#                           max_depth=6,
#                           subsample=0.8,
#                           colsample_bytree=0.8,
#                           objective='reg:squarederror',
#                           verbosity=0,
#                           n_jobs=-1)
#     Xtr = np.vstack(sup_df['morph_coordinates'])
#     ytr = np.vstack(sup_df['gene_coordinates'])
#     Xte = np.vstack(inf_df['morph_coordinates'])
#     yte = np.vstack(inf_df['gene_coordinates'])

#     # multi‑output wrapper trains one model per dimension
#     model = MultiOutputRegressor(XGBRegressor(**xgb_params))
#     model.fit(Xtr, ytr)
#     preds = model.predict(Xte)
#     return preds, yte

def xgboost_regression(sup_df, inf_df, feature_col='image_coordinates', label_col='year', **xgb_params):
    if not xgb_params:
        xgb_params = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', verbosity=0)

    X_train = np.vstack(sup_df[feature_col])
    y_train = sup_df[label_col].values
    X_test = np.vstack(inf_df[feature_col])
    y_test = inf_df[label_col].values

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test

############################################################
# Model E: Laplacian‑Regularised Least‑Squares (LapRLS)
############################################################

import numpy as np
from sklearn.metrics import pairwise_distances

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

# def laprls_regression(sup_df, inf_df, lam=1e-2, gamma=1.0, k=10, sigma=None):
#     Xs = np.vstack(sup_df['morph_coordinates'])
#     ys = np.vstack(sup_df['gene_coordinates'])
#     Xu = np.vstack(inf_df['morph_coordinates'])
#     w = laprls_closed_form(Xs, ys, Xu, lam, gamma, k, sigma)
#     preds = Xu.dot(w)
#     actuals = np.vstack(inf_df['gene_coordinates'])
#     return preds, actuals
def laprls_regression(sup_df, inf_df, feature_col='image_coordinates', label_col='year', lam=1e-2, gamma=1.0, k=10, sigma=None):
    Xs = np.vstack(sup_df[feature_col])
    ys = sup_df[label_col].values.reshape(-1, 1)
    Xu = np.vstack(inf_df[feature_col])
    yu = inf_df[label_col].values

    w = laprls_closed_form(Xs, ys, Xu, lam, gamma, k, sigma)
    preds = Xu @ w
    return preds.flatten(), yu


############################################################
# Model F: Twin‑Neural‑Network Regression (TNNR)
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools

class TwinDataset(Dataset):
    """Dataset of all (i,j) pairs from X, y -> targets y_i - y_j."""
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

class TwinRegressor(nn.Module):
    def __init__(self, in_dim, rep_dim=64, out_dim=1):
        super().__init__()
        # shared representation
        self.h = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, rep_dim), nn.ReLU()
        )
        # difference head
        self.g = nn.Linear(rep_dim, out_dim)
    
    def forward(self, x1, x2):
        h1, h2 = self.h(x1), self.h(x2)
        return self.g(h1 - h2)  # predict y1 - y2

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
    def tnnr_regression(supervised_df, inference_df, feature_col='image_coordinates', label_col='year',
                        rep_dim=64, lr=1e-3, epochs=200, batch_size=256, device=None):
        Xs = np.vstack(supervised_df[feature_col])
        ys = supervised_df[label_col].values.reshape(-1, 1)
        Xq = np.vstack(inference_df[feature_col])
        yq = inference_df[label_col].values

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds = TwinDataset(Xs, ys)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = TwinRegressor(in_dim=Xs.shape[1], rep_dim=rep_dim, out_dim=1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            for x1, x2, dy in loader:
                x1, x2, dy = x1.to(device), x2.to(device), dy.to(device)
                pred = model(x1, x2)
                loss = F.mse_loss(pred, dy)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
        ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
        Xq_t = torch.tensor(Xq, dtype=torch.float32, device=device)

        preds = []
        with torch.no_grad():
            for xq in Xq_t:
                diffs = model(xq.unsqueeze(0).repeat(len(Xs_t),1), Xs_t)
                estimates = ys_t + diffs
                mean_est = estimates.mean(dim=0)
                preds.append(mean_est.cpu().numpy())
        return np.array(preds).flatten(), yq


##############################################
#  Model G: Transductive SVM‑Regression (TSVR)
##############################################
from sklearn.svm import SVR
from copy import deepcopy

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

# def tsvr_regression(
#     supervised_df,
#     inference_df,
#     C=1.0,
#     epsilon=0.1,
#     kernel='rbf',
#     gamma='scale',
#     max_iter=10,
#     self_training_frac=0.2
# ):
#     """
#     Transductive SVR via iterative self-training:
#       1. fit SVR on supervised data
#       2. predict pseudolabels on unlabeled data
#       3. for up to max_iter:
#          – include all pseudolabels in first pass,
#            then only the self_training_frac fraction with smallest change
#          – refit on supervised + selected unlabeled
#          – stop if pseudolabels converge
#     """
#     # 1) Prepare data
#     X_sup = np.vstack(supervised_df['morph_coordinates'])
#     y_sup = np.vstack(supervised_df['gene_coordinates'])
#     X_unl = np.vstack(inference_df['morph_coordinates'])
#     y_unl = np.vstack(inference_df['gene_coordinates'])  # for evaluation

#     # 2) Base SVR wrapped for multi-output
#     base_svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
#     model = MultiOutputRegressor(base_svr)

#     # 3) Initial fit on supervised only
#     model.fit(X_sup, y_sup)
#     pseudo = model.predict(X_unl)

#     for it in range(max_iter):
#         if it == 0:
#             # first self-training round: include all unlabeled
#             X_aug = np.vstack([X_sup, X_unl])
#             y_aug = np.vstack([y_sup, pseudo])
#         else:
#             # measure stability of each pseudolabel
#             diffs = np.linalg.norm(pseudo - prev_pseudo, axis=1)
#             # pick the fraction with smallest change
#             thresh = np.percentile(diffs, self_training_frac * 100)
#             mask = diffs <= thresh
#             X_aug = np.vstack([X_sup, X_unl[mask]])
#             y_aug = np.vstack([y_sup, pseudo[mask]])

#         # 4) Refit on augmented data
#         model.fit(X_aug, y_aug)

#         # 5) Check convergence
#         prev_pseudo = pseudo
#         pseudo = model.predict(X_unl)
#         if np.allclose(pseudo, prev_pseudo, atol=1e-3):
#             break

#     # final predictions on unlabeled
#     return pseudo, y_unl

def tsvr_regression(
    supervised_df,
    inference_df,
    feature_col='image_coordinates',
    label_col='year',
    C=1.0,
    epsilon=0.1,
    kernel='rbf',
    gamma='scale',
    max_iter=10,
    self_training_frac=0.2
):
    X_sup = np.vstack(supervised_df[feature_col])
    y_sup = supervised_df[label_col].values
    X_unl = np.vstack(inference_df[feature_col])
    y_unl = inference_df[label_col].values  # for eval

    base_svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    model = MultiOutputRegressor(base_svr) if y_sup.ndim > 1 else base_svr
    model.fit(X_sup, y_sup)
    pseudo = model.predict(X_unl)

    for it in range(max_iter):
        if it == 0:
            X_aug = np.vstack([X_sup, X_unl])
            y_aug = np.hstack([y_sup, pseudo])
        else:
            diffs = np.abs(pseudo - prev_pseudo)
            thresh = np.percentile(diffs, self_training_frac * 100)
            mask = diffs <= thresh
            X_aug = np.vstack([X_sup, X_unl[mask]])
            y_aug = np.hstack([y_sup, pseudo[mask]])

        model.fit(X_aug, y_aug)
        prev_pseudo = pseudo
        pseudo = model.predict(X_unl)
        if np.allclose(pseudo, prev_pseudo, atol=1e-3):
            break

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


# def ucvme_regression(
#     supervised_df,
#     inference_df,
#     T=5,                  # MC dropout samples
#     lr=1e-3,
#     epochs=200,
#     w_unl=10.0,           # weight for unlabeled losses (wulb in Eq 11) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
#     device=None
# ):
#     # — Prepare tensors —
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     X_sup = torch.tensor(np.vstack(supervised_df['morph_coordinates']), dtype=torch.float32, device=device)
#     y_sup = torch.tensor(np.vstack(supervised_df['gene_coordinates']),    dtype=torch.float32, device=device)
#     X_unl = torch.tensor(np.vstack(inference_df['morph_coordinates']),   dtype=torch.float32, device=device)
#     y_unl_np = np.vstack(inference_df['gene_coordinates'])

#     # — Instantiate two co-trained BNNs :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3} —
#     in_dim, out_dim = X_sup.shape[1], y_sup.shape[1]
#     model_a = UCVMEModel(in_dim, out_dim).to(device)
#     model_b = UCVMEModel(in_dim, out_dim).to(device)
#     optimizer = torch.optim.Adam(
#         list(model_a.parameters()) + list(model_b.parameters()),
#         lr=lr
#     )

#     for ep in range(epochs):
#         model_a.train(); model_b.train()

#         # — Supervised heteroscedastic regression loss (Eq 1) :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5} —
#         y_a_sup, z_a_sup = model_a(X_sup)
#         y_b_sup, z_b_sup = model_b(X_sup)
#         L_sup_reg = (
#             ((y_a_sup - y_sup)**2 / (2*torch.exp(z_a_sup)) + z_a_sup/2).mean()
#           + ((y_b_sup - y_sup)**2 / (2*torch.exp(z_b_sup)) + z_b_sup/2).mean()
#         )

#         # — Aleatoric uncertainty consistency on labeled data (Eq 2) :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7} —
#         L_sup_unc = ((z_a_sup - z_b_sup)**2).mean()

#         # — Variational model ensembling for unlabeled pseudo-labels (Eqs 7–8) :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9} —
#         y_a_list, z_a_list, y_b_list, z_b_list = [], [], [], []
#         for _ in range(T):
#             # keep dropout active
#             y_a_t, z_a_t = model_a(X_unl)
#             y_b_t, z_b_t = model_b(X_unl)
#             y_a_list.append(y_a_t); z_a_list.append(z_a_t)
#             y_b_list.append(y_b_t); z_b_list.append(z_b_t)

#         y_a_stack = torch.stack(y_a_list)  # (T, N_unl, D)
#         z_a_stack = torch.stack(z_a_list)
#         y_b_stack = torch.stack(y_b_list)
#         z_b_stack = torch.stack(z_b_list)

#         # average over runs and models
#         y_tilde = (y_a_stack.mean(0) + y_b_stack.mean(0)) / 2  # Eq 7 :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}
#         z_tilde = (z_a_stack.mean(0) + z_b_stack.mean(0)) / 2  # Eq 8 :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}

#         # — Unlabeled heteroscedastic loss (Eq 10) :contentReference[oaicite:14]{index=14}&#8203;:contentReference[oaicite:15]{index=15} —
#         L_unl_reg = (
#             ((y_a_stack.mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean()
#           + ((y_b_stack.mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean()
#         )

#         # — Unlabeled uncertainty consistency (Eq 9) :contentReference[oaicite:16]{index=16}&#8203;:contentReference[oaicite:17]{index=17} —
#         L_unl_unc = (
#             ((z_a_stack.mean(0) - z_tilde)**2).mean()
#           + ((z_b_stack.mean(0) - z_tilde)**2).mean()
#         )

#         # — Total loss (Eq 11) :contentReference[oaicite:18]{index=18}&#8203;:contentReference[oaicite:19]{index=19} —
#         loss = L_sup_reg + L_sup_unc + w_unl * (L_unl_reg + L_unl_unc)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # — Inference: ensemble means over T dropout runs :contentReference[oaicite:20]{index=20}&#8203;:contentReference[oaicite:21]{index=21} —
#     model_a.eval(); model_b.eval()
#     preds_list = []
#     with torch.no_grad():
#         for _ in range(T):
#             y_a_u, _ = model_a(X_unl)
#             y_b_u, _ = model_b(X_unl)
#             preds_list.append((y_a_u + y_b_u) / 2)
#         preds = torch.stack(preds_list).mean(0).cpu().numpy()

#     return preds, y_unl_np

def ucvme_regression(
    supervised_df,
    inference_df,
    feature_col='image_coordinates',
    label_col='year',
    T=5,
    lr=1e-3,
    epochs=200,
    w_unl=10.0,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_sup = torch.tensor(np.vstack(supervised_df[feature_col]), dtype=torch.float32, device=device)
    y_sup = torch.tensor(supervised_df[label_col].values.reshape(-1, 1), dtype=torch.float32, device=device)
    X_unl = torch.tensor(np.vstack(inference_df[feature_col]), dtype=torch.float32, device=device)
    y_unl_np = inference_df[label_col].values

    in_dim, out_dim = X_sup.shape[1], 1
    model_a = UCVMEModel(in_dim, out_dim).to(device)
    model_b = UCVMEModel(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(list(model_a.parameters()) + list(model_b.parameters()), lr=lr)

    for ep in range(epochs):
        model_a.train(); model_b.train()
        y_a_sup, z_a_sup = model_a(X_sup)
        y_b_sup, z_b_sup = model_b(X_sup)

        L_sup_reg = (((y_a_sup - y_sup)**2 / (2*torch.exp(z_a_sup)) + z_a_sup/2).mean() +
                     ((y_b_sup - y_sup)**2 / (2*torch.exp(z_b_sup)) + z_b_sup/2).mean())
        L_sup_unc = ((z_a_sup - z_b_sup)**2).mean()

        y_a_list, z_a_list, y_b_list, z_b_list = [], [], [], []
        for _ in range(T):
            ya, za = model_a(X_unl); yb, zb = model_b(X_unl)
            y_a_list.append(ya); z_a_list.append(za)
            y_b_list.append(yb); z_b_list.append(zb)

        y_tilde = (torch.stack(y_a_list).mean(0) + torch.stack(y_b_list).mean(0)) / 2
        z_tilde = (torch.stack(z_a_list).mean(0) + torch.stack(z_b_list).mean(0)) / 2

        L_unl_reg = (((torch.stack(y_a_list).mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean() +
                     ((torch.stack(y_b_list).mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean())
        L_unl_unc = (((torch.stack(z_a_list).mean(0) - z_tilde)**2).mean() +
                     ((torch.stack(z_b_list).mean(0) - z_tilde)**2).mean())

        loss = L_sup_reg + L_sup_unc + w_unl * (L_unl_reg + L_unl_unc)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model_a.eval(); model_b.eval()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            ya_u, _ = model_a(X_unl)
            yb_u, _ = model_b(X_unl)
            preds.append((ya_u + yb_u)/2)
        final_preds = torch.stack(preds).mean(0).cpu().numpy().flatten()

    return final_preds, y_unl_np

##############################################
#  Model I: RankUp-simplified                #
##############################################

def _simple_mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(),
                         nn.Linear(256,128), nn.ReLU(),
                         nn.Linear(128,out_dim))

def _pairwise_rank_loss(pred, y):
    # hinge on pairwise orderings
    dif_pred = pred[:,None]-pred[None,:]
    dif_true = y[:,None]-y[None,:]
    sign_true = np.sign(dif_true)
    margin = 0.1
    loss = np.maximum(0, margin - sign_true*dif_pred)
    return loss.mean()

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
def rankup_regression(
    sup_df,
    inf_df,
    feature_col='image_coordinates',
    label_col='year',
    lr=1e-3,
    epochs=200,
    alpha=0.3
):
    Xs = torch.tensor(np.vstack(sup_df[feature_col]), dtype=torch.float32)
    ys = torch.tensor(sup_df[label_col].values.reshape(-1, 1), dtype=torch.float32)
    Xu = torch.tensor(np.vstack(inf_df[feature_col]), dtype=torch.float32)
    yu = inf_df[label_col].values

    net = _simple_mlp(Xs.size(1), 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for _ in range(epochs):
        idx = torch.randperm(Xs.size(0))[:32]
        xb, yb = Xs[idx], ys[idx]
        pred_b = net(xb)
        loss = F.mse_loss(pred_b, yb)

        with torch.no_grad():
            rank_loss = torch.tensor(_pairwise_rank_loss(
                pred_b.squeeze().numpy(), yb.squeeze().numpy()), dtype=torch.float32)
        loss = loss + alpha * rank_loss

        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        preds = net(Xu).numpy().flatten()
    return preds, yu

class AGDNConv(MessagePassing):
    """
    A simplified AGDN-like convolution layer, in PyTorch Geometric style.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        negative_slope=0.2,
        dropout=0.0,
        residual=True,
        **kwargs
    ):
        super().__init__(aggr='add', node_dim=0)  # Force node_dim=0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual

        # Linear transformation for inputs
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Residual connection
        if residual and in_channels != heads * out_channels:
            self.res_fc = nn.Linear(in_channels, heads * out_channels, bias=False)
        else:
            self.res_fc = None

        # Bias
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        nn.init.xavier_uniform_(self.att_l, gain=1.0)
        nn.init.xavier_uniform_(self.att_r, gain=1.0)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=1.0)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        """
        # Transform input
        x_lin = self.linear(x).view(-1, self.heads, self.out_channels)

        # Residual part (optional)
        if self.res_fc is not None:
            x_res = self.res_fc(x).view(-1, self.heads, self.out_channels)
        else:
            x_res = x_lin if self.residual else None

        # Compute alpha_l and alpha_r per node
        alpha_l = (x_lin * self.att_l).sum(dim=-1, keepdim=True)  # shape: [N, heads, 1]
        alpha_r = (x_lin * self.att_r).sum(dim=-1, keepdim=True)  # shape: [N, heads, 1]

        # Propagate messages; the target indices are passed as "index" to the message function.
        out = self.propagate(edge_index, x=x_lin, alpha_l=alpha_l, alpha_r=alpha_r)

        # Flatten heads
        out = out.view(-1, self.heads * self.out_channels)

        # Add residual if available
        if self.residual and (x_res is not None):
            out += x_res.view(-1, self.heads * self.out_channels)

        # Add bias
        out = out + self.bias
        return out

    def message(self, x_j, alpha_l_j, alpha_r_i, index):
        """
        x_j: Neighbor features [E, heads, out_channels]
        alpha_l_j: Attention term from neighbor [E, heads, 1]
        alpha_r_i: Attention term from center node [E, heads, 1]
        index:   Target indices for each edge from propagate
        """
        alpha = alpha_l_j + alpha_r_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # Use the provided "index" for softmax computation.
        alpha = torch_geometric.utils.softmax(alpha, index, num_nodes=x_j.size(0))
        alpha = self.dropout(alpha)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out

class AGDN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=1,
        dropout=0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            AGDNConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                AGDNConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        # Last layer
        self.convs.append(
            AGDNConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# def agdn_regression(supervised_df, inference_df, morph_col='morph_coordinates', gene_col='gene_coordinates',
#                     hidden=64, num_layers=2, heads=1, dropout=0.1,
#                     epochs=500, lr=1e-3):
#     """
#     Train an AGDN to map morphological embeddings -> gene embeddings,
#     using 'supervised_df' as labeled data, then predict gene embeddings
#     for 'inference_df'. Return the per-sample Euclidean distance.
    
#     :param supervised_df: DataFrame with columns [morph_col, gene_col]
#     :param inference_df:  DataFrame with columns [morph_col, gene_col]
#     :param morph_col:     Name of the morphological embedding column in the DataFrame
#     :param gene_col:      Name of the gene embedding column in the DataFrame
#     :param hidden:        Hidden dimension
#     :param num_layers:    Number of AGDN layers
#     :param heads:         Number of heads
#     :param dropout:       Dropout rate
#     :param epochs:        Training epochs
#     :param lr:            Learning rate
#     :return: Numpy array of distances for each row in inference_df
#     """

#     # 1) Extract morphological & gene embeddings as arrays
#     sup_morph = np.stack(supervised_df[morph_col].values, axis=0)
#     sup_gene  = np.stack(supervised_df[gene_col].values, axis=0)
#     inf_morph = np.stack(inference_df[morph_col].values, axis=0)
#     inf_gene  = np.stack(inference_df[gene_col].values, axis=0)

#     sup_morph_t = torch.FloatTensor(sup_morph)
#     sup_gene_t  = torch.FloatTensor(sup_gene)
#     inf_morph_t = torch.FloatTensor(inf_morph)
#     inf_gene_t  = torch.FloatTensor(inf_gene)

#     N_sup = sup_morph.shape[0]
#     N_inf = inf_morph.shape[0]

#     # 2) Build adjacency for supervised portion
#     #    For demonstration, we do a full mesh among supervised nodes
#     srcs, dsts = [], []
#     for i in range(N_sup):
#         for j in range(N_sup):
#             if i != j:
#                 srcs.append(i)
#                 dsts.append(j)
#     edge_index_sup = torch.tensor([srcs, dsts], dtype=torch.long)

#     data_sup = Data(x=sup_morph_t, edge_index=edge_index_sup)

#     # 3) Build the AGDN
#     in_channels  = sup_morph.shape[1]
#     out_channels = sup_gene.shape[1]
#     model = AGDN(in_channels, hidden, out_channels, num_layers=num_layers, heads=heads, dropout=dropout)

#     # 4) Loss & optimizer
#     criterion = nn.MSELoss()
#     optimizer = Adam(model.parameters(), lr=lr)

#     # 5) Train on supervised
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data_sup.x, data_sup.edge_index)   # [N_sup, out_channels]
#         loss = criterion(out, sup_gene_t)
#         loss.backward()
#         optimizer.step()

#         if (epoch+1) % 10 == 0:
#             print(f"AGDN epoch={epoch+1}, MSE={loss.item():.4f}")

#     # 6) Build a combined graph that includes inference nodes,
#     #    so that we can propagate morphological info from sup->inf
#     #    We'll label the new nodes from [N_sup..N_sup+N_inf-1]
#     srcs_inf, dsts_inf = [], []
#     for i in range(N_inf):
#         inf_node_id = N_sup + i
#         # Connect to all supervised nodes (bidirectional):
#         for j in range(N_sup):
#             srcs_inf.append(inf_node_id)
#             dsts_inf.append(j)
#             srcs_inf.append(j)
#             dsts_inf.append(inf_node_id)
#     edge_index_inf = torch.tensor([srcs_inf, dsts_inf], dtype=torch.long)

#     big_x = torch.cat([sup_morph_t, inf_morph_t], dim=0)
#     big_edge = torch.cat([edge_index_sup, edge_index_inf], dim=1)
#     data_inf = Data(x=big_x, edge_index=big_edge)

#     # 7) Infer for the last portion
#     model.eval()
#     with torch.no_grad():
#         big_out = model(data_inf.x, data_inf.edge_index) # shape [N_sup+N_inf, out_channels]
#     inf_pred = big_out[N_sup:, :]  # shape [N_inf, out_channels]

#     return inf_pred.numpy(), inf_gene_t.numpy()

def agdn_regression(
    supervised_df,
    inference_df,
    morph_col='image_coordinates',
    label_col='year',
    hidden=64,
    num_layers=2,
    heads=1,
    dropout=0.1,
    epochs=500,
    lr=1e-3
):
    # 1) Extract embeddings and scalar labels
    sup_morph = np.stack(supervised_df[morph_col].values, axis=0)
    sup_years = supervised_df[label_col].values.reshape(-1, 1)
    inf_morph = np.stack(inference_df[morph_col].values, axis=0)
    inf_years = inference_df[label_col].values  # keep flat for evaluation

    sup_morph_t = torch.FloatTensor(sup_morph)
    sup_years_t = torch.FloatTensor(sup_years)
    inf_morph_t = torch.FloatTensor(inf_morph)

    N_sup = len(sup_morph)
    N_inf = len(inf_morph)

    # 2) Build full-mesh graph for supervised samples
    srcs, dsts = [], []
    for i in range(N_sup):
        for j in range(N_sup):
            if i != j:
                srcs.append(i)
                dsts.append(j)
    edge_index_sup = torch.tensor([srcs, dsts], dtype=torch.long)

    data_sup = Data(x=sup_morph_t, edge_index=edge_index_sup)

    # 3) Build AGDN model
    in_channels = sup_morph.shape[1]
    out_channels = 1  # scalar prediction
    model = AGDN(in_channels, hidden, out_channels, num_layers=num_layers, heads=heads, dropout=dropout)

    # 4) Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # 5) Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data_sup.x, data_sup.edge_index)  # [N_sup, 1]
        loss = criterion(out, sup_years_t)
        loss.backward()
        optimizer.step()

    # 6) Build new graph: inference + supervision nodes
    srcs_inf, dsts_inf = [], []
    for i in range(N_inf):
        inf_node = N_sup + i
        for j in range(N_sup):
            srcs_inf.append(inf_node)
            dsts_inf.append(j)
            srcs_inf.append(j)
            dsts_inf.append(inf_node)
    edge_index_inf = torch.tensor([srcs_inf, dsts_inf], dtype=torch.long)

    big_x = torch.cat([sup_morph_t, inf_morph_t], dim=0)
    big_edge = torch.cat([edge_index_sup, edge_index_inf], dim=1)
    data_inf = Data(x=big_x, edge_index=big_edge)

    # 7) Inference
    model.eval()
    with torch.no_grad():
        big_out = model(data_inf.x, data_inf.edge_index)  # [N_sup + N_inf, 1]
    inf_pred = big_out[N_sup:].squeeze().numpy()  # [N_inf]

    return inf_pred, inf_years

def cluster_and_predict(df, image_features, valid_indices, supervised_per_style, n_clusters):
    df = df.iloc[valid_indices].reset_index(drop=True) # selects only the rows of valid indices 
    image_features = image_features[:len(df)] 

    # n_supervised = max(1, int(supervised_fraction * n_total))
    # indices = np.arange(n_total)
    # np.random.shuffle(indices)

    # supervised_indices = indices[:n_supervised] # MAKE SURE INCLUDES ONE OF EACH CLASS 
    # test_indices = indices[n_supervised:]

    # Uniform per-style supervised sampling
    # supervised_indices = []

    # for style in df['style'].unique():
    #     style_indices = df.index[df['style'] == style].tolist()
    #     n_supervised_style = max(1, int(supervised_fraction * len(style_indices)))
    #     sampled = np.random.choice(style_indices, n_supervised_style, replace=False)
    #     supervised_indices.extend(sampled)

    # supervised_indices = np.array(supervised_indices)
    supervised_indices = []

    for style in df['style'].unique():
        style_indices = df.index[df['style'] == style].tolist()
        if len(style_indices) >= supervised_per_style:
            sampled = np.random.choice(style_indices, supervised_per_style, replace=False)
        else:
            sampled = style_indices  # if not enough, take all available
        supervised_indices.extend(sampled)

    supervised_indices = np.array(supervised_indices)

    # unsupervised_indices = np.setdiff1d(np.arange(len(df)), supervised_indices).sample()
    unsupervised_indices = np.setdiff1d(np.arange(len(df)), supervised_indices)
    test_indices = unsupervised_indices  # testing on the rest

    print(f"Sanity Check: Supervised samples per style:")
    for style in df['style'].unique():
        count = np.sum(df.iloc[supervised_indices]['style'] == style)
        print(f"  {style}: {count} supervised points")


    # Clustering on unsupervised data only
    # unsupervised_indices = np.setdiff1d(indices, supervised_indices)
    X_unsup = image_features[test_indices]
    y_unsup = df.iloc[test_indices]['year'].values.reshape(-1, 1)

    x_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_unsup) # only FITTING with the unsupervised data
    y_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(y_unsup)
    print("From cluster_and_predict: ", n_clusters)

    x_clusters = x_kmeans.predict(X_unsup)
    y_clusters = y_kmeans.predict(y_unsup) #df['year'].values.reshape(-1, 1))
    # x_clusters: which clusters the ith sample's image belongs to 
    # y_clusters: which clusters the ith sample's year belongs to 

    # Get supervised cluster assignments
    sup_image_features = image_features[supervised_indices]
    y_sup = df.iloc[supervised_indices]['year'].values.reshape(-1, 1)

    sup_x_cluster = x_kmeans.predict(sup_image_features)
    sup_y_cluster = y_kmeans.predict(y_sup)

    # Learn bridge using supervised set w/majority voting (bc hard voting)
    mapping = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        votes = np.zeros(n_clusters)
        for x_sup, y_sup in zip(sup_x_cluster, sup_y_cluster):
            if x_sup == i: 
                votes[y_sup] += 1
        mapping[i] = np.argmax(votes) # index i is the ith x cluster; the value of mapping is the y cluster it maps to 

    # Centroids
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        vals = [df['year'].iloc[j] for j in range(len(y_clusters)) if y_clusters[j] == i]
        centroids[i] = np.mean(vals) if vals else 0

    preds = np.array([centroids[mapping[x_clusters[i]]] for i in range(len(x_clusters))])
    true = df['year'].iloc[test_indices].values
    bridged_mae = mean_absolute_error(true, preds)

    # KNN baseline
    X_train = image_features[supervised_indices]
    y_train = df['year'].iloc[supervised_indices].values
    X_test = image_features[test_indices]
    y_test = df['year'].iloc[test_indices].values

    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_mae = mean_absolute_error(y_test, knn_preds)

    inference_df = df.drop(index=supervised_indices)
    # Additional Baselines
    mt_preds, mt_actuals = mean_teacher_regression(df.iloc[supervised_indices], inference_df)
    xgb_preds, xgb_actuals = xgboost_regression(df.iloc[supervised_indices], inference_df)
    lap_preds, lap_actuals = laprls_regression(df.iloc[supervised_indices], inference_df)
    tsvr_preds, tsvr_actuals = tsvr_regression(df.iloc[supervised_indices], inference_df)
    # tnnr_preds, tnnr_actuals = tnnr_regression(df.iloc[supervised_indices], inference_df)
    ucv_preds, ucv_actuals = ucvme_regression(df.iloc[supervised_indices], inference_df)
    rank_preds, rank_actuals = rankup_regression(df.iloc[supervised_indices], inference_df)

    mean_teacher_error = mean_absolute_error(mt_actuals, mt_preds)
    xgb_error          = mean_absolute_error(xgb_actuals, xgb_preds)
    lap_error          = mean_absolute_error(lap_actuals, lap_preds)
    tsvr_error         = mean_absolute_error(tsvr_actuals, tsvr_preds)
    # tnnr_error         = mean_absolute_error(tnnr_actuals, tnnr_preds)
    ucv_error          = mean_absolute_error(ucv_actuals, ucv_preds)
    rank_error         = mean_absolute_error(rank_actuals, rank_preds)

    print(f"Bridged Clustering Error: {bridged_mae}")
    print(f"KNN Error: {knn_mae}")
    print(f"Mean Teacher Error: {mean_teacher_error}")
    print(f"XGBoost Error: {xgb_error}")
    print(f"Laplacian RLS Error: {lap_error}")
    print(f"TSVR Error: {tsvr_error}")
    # print(f"TNNR Error: {tnnr_error}")
    print(f"UCVME Error: {ucv_error}")
    print(f"RankUp Error: {rank_error}")

    # --- GNN Bridging ---
    # adj = kneighbors_graph(image_features, n_neighbors=10, mode='connectivity', include_self=False)
    # edge_index, _ = dense_to_sparse(torch.tensor(adj.toarray(), dtype=torch.float32))

    # # Label preparation for GNN (use y_clusters and sup_y_cluster only where available)
    # full_y_cluster = np.full(len(df), -1)
    # full_y_cluster[test_indices] = y_clusters
    # full_y_cluster[supervised_indices] = sup_y_cluster
    # sup_mask = np.zeros(len(df), dtype=bool)
    # sup_mask[supervised_indices] = True

    # # GNN bridging predictions
    # gnn_preds = run_gnn_bridged(
    #     X=image_features,
    #     y_lab=full_y_cluster,
    #     sup_mask=sup_mask,
    #     n_clusters=n_clusters,
    #     edge_index=edge_index,
    #     edge_weights=None,
    #     centroids=centroids.reshape(-1, 1),
    #     tau=1.0,
    #     hidden_dim=64,
    #     lr=1e-3,
    #     epochs=100
    # )

    # gnn_pred_vals = gnn_preds[test_indices].flatten()
    # gnn_mae = mean_absolute_error(df['year'].iloc[test_indices].values, gnn_pred_vals)

    # --- GNN Setup ---
    adj = kneighbors_graph(image_features, n_neighbors=10, mode='connectivity', include_self=False)
    edge_index, _ = dense_to_sparse(torch.tensor(adj.toarray(), dtype=torch.float32))

    # Label prep
    full_y_cluster = np.full(len(df), -1)
    full_y_cluster[test_indices] = y_clusters
    full_y_cluster[supervised_indices] = sup_y_cluster
    sup_mask = np.zeros(len(df), dtype=bool)
    sup_mask[supervised_indices] = True

    # GNN Grid Search
    true_years = df['year'].iloc[test_indices].values
    best_config, gnn_mae, gnn_pred_vals, gnn_search_results = grid_search_gnn(
        X=image_features,
        y_lab=full_y_cluster,
        sup_mask=sup_mask,
        n_clusters=n_clusters,
        edge_index=edge_index,
        centroids=centroids,
        test_indices=test_indices,
        true_years=true_years,
        hidden_dims=[64, 128],
        taus=[0.5, 1.0, 2.0],
        lrs=[1e-4, 5e-4, 1e-3],
        epochs=100
    )

    return {
        'BKM': bridged_mae,
        'KNN': knn_mae,
        'Mean Teacher': mean_teacher_error,
        'XGBoost': xgb_error,
        'Laplacian RLS': lap_error,
        'TSVR': tsvr_error,
        # 'TNNR': tnnr_error,
        'UCVME': ucv_error,
        'RankUp': rank_error,
        'GNN': gnn_mae
    }

def run_all_trials(df, image_folder, n_clusters):
    method_names = [
        'BKM', 'KNN', 'Mean Teacher', 'XGBoost',
        'Laplacian RLS', 'TSVR', 'UCVME', 'RankUp', 'GNN'
    ]

    # Initialize results dictionary
    results = {
        s: {method: [] for method in method_names}
        for s in SUPERVISED_VALUES
    }

    image_features, valid_indices = encode_images(df, image_folder)
    df['image_coordinates'] = list(image_features)

    for s in SUPERVISED_VALUES:
        print(f"\nSupervised fraction: {s}")
        for trial in range(N_TRIALS):
            print(f"Trial {trial + 1}...")
            trial_results = cluster_and_predict(df.copy(), image_features, valid_indices, supervised_per_style=s, n_clusters=n_clusters)

            # Store each method's result
            for method in method_names:
                results[s][method].append(trial_results[method])

            # Print current trial results
            print(f"--- Trial {trial + 1} Results ---")
            for method in method_names:
                print(f"{method} MAE: {trial_results[method]:.2f}")

    # Final averaged results
    print("\n=== Average Results ===")
    for s in SUPERVISED_VALUES:
        print(f"\nSupervised value: {s}")
        for method in method_names:
            avg = np.mean(results[s][method])
            print(f"{method}: {avg:.2f}")

    return results

# if __name__ == '__main__':
#     for n_clusters in N_CLUSTERS: 
#         print(n_clusters)
#         metadata_csv = "/Users/ellietanimura/bridged_clustering/neurips/template_experiment/filtered_styles" + str(n_clusters) + ".csv"
#         image_folder = "/Users/ellietanimura/bridged_clustering/neurips/template_experiment/wikiart"

#         style_list = ['Early_Renaissance', 'Naive_Art_Primitivism', 'Abstract_Expressionism', 'Baroque', 'Ukiyo_e', 'Rococo']
#         print(style_list[:n_clusters])
#         df = pd.read_csv(metadata_csv)
#         df = df[df['style'].isin(style_list[:n_clusters])]
#         df = df.dropna(subset=['year'])
#         df['year'] = df['year'].astype(int)

#         results = run_all_trials(df, image_folder, n_clusters=n_clusters)

#         for s in SUPERVISED_VALUES:
#             print(f"Supervised {s:.2%} — BKM MAE: {np.mean(results[s]['BKM']):.2f}, KNN MAE: {np.mean(results[s]['KNN']):.2f}")

if __name__ == '__main__':
    output_path = "gnn_cluster_sweep_results.txt"

    with open(output_path, 'w') as f:
        for n_clusters in N_CLUSTERS: 
            f.write(f"\n========== Clusters: {n_clusters} ==========\n")
            print(n_clusters)

            metadata_csv = f"/Users/ellietanimura/bridged_clustering/neurips/template_experiment/filtered_styles{n_clusters}.csv"
            image_folder = "/Users/ellietanimura/bridged_clustering/neurips/template_experiment/wikiart"

            style_list = ['Early_Renaissance', 'Naive_Art_Primitivism', 'Abstract_Expressionism', 'Baroque', 'Ukiyo_e', 'Rococo']
            print(style_list[:n_clusters])
            df = pd.read_csv(metadata_csv)
            df = df[df['style'].isin(style_list[:n_clusters])]
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)

            results = run_all_trials(df, image_folder, n_clusters=n_clusters)

            for s in SUPERVISED_VALUES:
                bkm = np.mean(results[s]['BKM'])
                knn = np.mean(results[s]['KNN'])
                gnn = np.mean(results[s]['GNN'])
                line = f"Supervised {s:.2%} — BKM MAE: {bkm:.2f}, KNN MAE: {knn:.2f}, GNN MAE: {gnn:.2f}\n"
                print(line.strip())
                f.write(line)

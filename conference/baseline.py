### Script for Baselines

# Standard library
import itertools

# Core libraries
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
import torch_geometric.utils

# HuggingFace Transformers
from transformers import AutoModel, AutoTokenizer

# Scikit-learn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    pairwise_distances,
    r2_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# Constrained clustering
from k_means_constrained import KMeansConstrained

# Vision
import torchvision
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights

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

def mean_teacher_regression(supervised_samples, inference_samples, lr=0.001, w_max=0.1, alpha=0.99, ramp_len=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(supervised_samples['morph_coordinates'].sample().iloc[0])
    output_dim = len(supervised_samples['gene_coordinates'].sample().iloc[0])
    model = MeanTeacherModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    for epoch in range(100):
        train_mean_teacher(model, supervised_loader, unlabeled_loader, optimizer, device,alpha=alpha, w_max=w_max, ramp_len=ramp_len)

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
        epochs: int = 100
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
    epochs=100,
    batch_size=32,
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
    epochs=100,
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
    mc_T=5,                  # MC dropout samples
    lr=1e-3,
    epochs=100,
    w_unl=10.0,           # weight for unlabeled losses (wulb in Eq 11) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
    device=None
):
    T = mc_T  # number of MC dropout samples
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
    epochs=100,
    batch_size=128,
    alpha_arc=1.0,
    alpha_arc_ulb=1.0,
    alpha_rda=0.05,          # RDA weight
    temperature=0.7,                   # temperature for softmax
    tau=0.90,                # confidence threshold
    ema_m=0.999,
    device=None
):
    T = temperature  # temperature for softmax
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
    epochs=100,
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
# Unmatched Regression #
####################################

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from adapt.instance_based import KMM  # pip install adapt

def kernel_mean_matching_regression(
    image_df,
    gene_df,
    supervised_df,
    inference_df,
    alpha=1e-2,    # regularization for KRR
    kmm_B=1000,    # B parameter for KMM
    kmm_eps=1e-3,  # eps parameter for KMM
    sigma=None     # optional RBF width; if None it’s auto-computed
):
    """
    1) Compute KMM weights to match supervised → image-only X-marginal and supervised → gene-only Y-marginal
    2) Fit a weighted RBF-KRR on the supervised pairs
    3) Predict on inference_df and return (predictions, ground truth)
    """

    # --- 1) Extract raw arrays ---
    X_img   = np.vstack(image_df['morph_coordinates'].values)      # (N_img, d)
    Y_gene  = np.vstack(gene_df['gene_coordinates'].values)        # (N_gene, d)
    X_sup   = np.vstack(supervised_df['morph_coordinates'].values) # (N_sup, d)
    Y_sup   = np.vstack(supervised_df['gene_coordinates'].values)  # (N_sup, d)
    X_test  = np.vstack(inference_df['morph_coordinates'].values)  # (N_test, d)

    # --- 2) Fit scalers on combined supervised + unpaired sets ---
    scaler_X = StandardScaler().fit(np.vstack((X_sup, X_img)))
    scaler_Y = StandardScaler().fit(np.vstack((Y_sup, Y_gene)))

    # --- 3) Transform all sets into standardized space ---
    Xs_sup  = scaler_X.transform(X_sup)
    Xs_img  = scaler_X.transform(X_img)
    Xs_test = scaler_X.transform(X_test)
    Ys_sup  = scaler_Y.transform(Y_sup)
    Ys_gene = scaler_Y.transform(Y_gene)

    Xs_sup  = np.asarray(Xs_sup,  dtype=np.float64)
    Xs_img  = np.asarray(Xs_img,  dtype=np.float64)
    Xs_test = np.asarray(Xs_test, dtype=np.float64)
    Ys_sup  = np.asarray(Ys_sup,  dtype=np.float64)
    Ys_gene = np.asarray(Ys_gene, dtype=np.float64)

    # --- 4) Choose sigma if not provided ---
    if sigma is None:
        subset = Xs_sup[np.random.choice(len(Xs_sup), min(len(Xs_sup), 200), replace=False)]
        d2 = np.sum((subset[:, None, :] - subset[None, :, :])**2, axis=2).ravel()
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med)
    gamma = 1.0 / (2 * sigma**2)

    # --- 5) Prepare dummy y for KMM and compute weights ---
    N_sup = Xs_sup.shape[0]
    dummy_y = np.zeros(N_sup)

    kmm_x = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False)
    # supply named args so y is dummy_y and Xt is the image-only target
    kmm_x.fit(X=Xs_sup, y=dummy_y, Xt=Xs_img)
    w_x = kmm_x.weights_

    kmm_y = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False)
    kmm_y.fit(X=Ys_sup, y=dummy_y, Xt=Ys_gene)
    w_y = kmm_y.weights_

    # --- 6) Combine and renormalize weights ---
    w = w_x * w_y
    w *= (len(w) / np.sum(w))

    # --- 7) Fit weighted Kernel Ridge Regression per output dim ---
    preds = []
    for dim in range(Ys_sup.shape[1]):
        kr = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        kr.fit(Xs_sup, Ys_sup[:, dim], sample_weight=w)
        preds.append(kr.predict(Xs_test))
    Y_pred = np.vstack(preds).T

    # --- 8) Inverse-transform and return ---
    Y_pred = scaler_Y.inverse_transform(Y_pred)
    Y_true = np.vstack(inference_df['gene_coordinates'].values)
    return Y_pred, Y_true


def reversed_kernel_mean_matching_regression(
    gene_df,
    image_df,
    supervised_df,
    inference_df,
    alpha: float = 1e-2,
    kmm_B: int = 1000,
    kmm_eps: float = 1e-3,
    sigma: float | None = None,
    random_state: int = 0
):
    """
    Predict IMAGE (morph) features from GENE features.

    Steps:
    1) Stack and standardize gene (X) and image (Y) spaces over supervised + unpaired.
    2) Compute sigma via median distance on supervised X if not provided.
    3) Run KMM on X_sup → match gene_df distribution  → weights w_x
    4) Run KMM on Y_sup → match image_df distribution → weights w_y
    5) Combine w = w_x * w_y, normalize so ∑w = N_sup.
    6) Fit weighted RBF-KRR on (X_sup, Y_sup) → one model per Y-dim.
    7) Predict on inference_df['gene_coordinates'], un-standardize, return.
    """
    # 1) Extract raw arrays
    X_gene_unp = np.vstack(gene_df['gene_coordinates'].values)
    Y_img_unp  = np.vstack(image_df['morph_coordinates'].values)
    X_sup      = np.vstack(supervised_df['gene_coordinates'].values)
    Y_sup      = np.vstack(supervised_df['morph_coordinates'].values)
    X_test     = np.vstack(inference_df['gene_coordinates'].values)

    # 2) Standardize over supervised + unpaired
    scaler_X = StandardScaler().fit(np.vstack((X_sup, X_gene_unp)))
    scaler_Y = StandardScaler().fit(np.vstack((Y_sup, Y_img_unp)))
    Xs_sup      = scaler_X.transform(X_sup)
    Xs_gene_unp = scaler_X.transform(X_gene_unp)
    Ys_sup      = scaler_Y.transform(Y_sup)
    Ys_img_unp  = scaler_Y.transform(Y_img_unp)
    Xs_test     = scaler_X.transform(X_test)

    Xs_sup      = np.asarray(Xs_sup,      dtype=np.float64)
    Xs_gene_unp = np.asarray(Xs_gene_unp, dtype=np.float64)
    Ys_sup      = np.asarray(Ys_sup,      dtype=np.float64)
    Ys_img_unp  = np.asarray(Ys_img_unp,  dtype=np.float64)
    Xs_test     = np.asarray(Xs_test,     dtype=np.float64)

    # 3) Choose sigma if not provided
    if sigma is None:
        subset = Xs_sup[np.random.choice(len(Xs_sup), min(len(Xs_sup), 200), replace=False)]
        d2 = np.sum((subset[:, None, :] - subset[None, :, :])**2, axis=2).ravel()
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med)
    gamma = 1.0 / (2 * sigma**2)

    # 4) Prepare dummy y for KMM
    N_sup = Xs_sup.shape[0]
    dummy_y = np.zeros(N_sup)

    # 5) KMM to match supervised → unpaired distributions
    kmm_x = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False, random_state=random_state)
    kmm_x.fit(X=Xs_sup, y=dummy_y, Xt=Xs_gene_unp)
    w_x = kmm_x.weights_

    kmm_y = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False, random_state=random_state)
    kmm_y.fit(X=Ys_sup, y=dummy_y, Xt=Ys_img_unp)
    w_y = kmm_y.weights_

    # 6) Combine and normalize weights
    w = w_x * w_y
    w *= (len(w) / np.sum(w))

    # 7) Fit weighted Kernel Ridge Regression per output dimension
    preds = []
    for dim in range(Ys_sup.shape[1]):
        kr = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        kr.fit(Xs_sup, Ys_sup[:, dim], sample_weight=w)
        preds.append(kr.predict(Xs_test))
    Y_pred_s = np.vstack(preds).T

    # 8) Inverse-transform and return
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)
    Y_true = np.vstack(inference_df['morph_coordinates'].values)
    return Y_pred, Y_true


import numpy as np
from sklearn.cluster import KMeans

def gaussian_logpdf(X, mu, var, eps=1e-8):
    """
    Numerically stable log‐PDF of an isotropic Gaussian:
      X:   (n, d)
      mu:  (d,) or (n, d)
      var: scalar variance
    Returns
    -------
    log_pdf : (n,)
    """
    var = var + eps
    d = X.shape[1]
    diff = X - mu
    exponent = -0.5 * np.sum(diff**2, axis=1) / var
    log_norm = 0.5 * d * np.log(2 * np.pi * var)
    return exponent - log_norm

def em_regression(
    supervised_df,
    image_df,
    gene_df,
    inference_df,
    n_components=3,
    max_iter=100,
    tol=1e-4,
    eps=1e-8
):
    # 1) Stack
    X_sup = np.vstack(supervised_df['morph_coordinates'])
    Y_sup = np.vstack(supervised_df['gene_coordinates'])
    X_xo  = np.vstack(image_df['morph_coordinates']) if len(image_df) else np.empty((0, X_sup.shape[1]))
    Y_yo  = np.vstack(gene_df['gene_coordinates'])    if len(gene_df)  else np.empty((0, Y_sup.shape[1]))

    n_sup, d_x = X_sup.shape
    n_xo = X_xo.shape[0]
    n_yo = Y_yo.shape[0]
    d_y = Y_sup.shape[1]
    K   = n_components

    # 2) Init with KMeans on the supervised subset
    mu_x = KMeans(K, random_state=0).fit(X_sup).cluster_centers_
    mu_y = KMeans(K, random_state=0).fit(Y_sup).cluster_centers_
    var_x = np.full(K, np.mean(np.var(X_sup, axis=0)) + eps)
    var_y = np.full(K, np.mean(np.var(Y_sup, axis=0)) + eps)
    pi    = np.full(K, 1.0/K)

    # 3) EM
    for _ in range(max_iter):
        # --- E-step: build log-responsibilities ---
        log_r_sup = np.stack([
            np.log(pi[k] + eps)
            + gaussian_logpdf(X_sup, mu_x[k], var_x[k], eps)
            + gaussian_logpdf(Y_sup, mu_y[k], var_y[k], eps)
            for k in range(K)
        ], axis=1)
        log_r_sup -= log_r_sup.max(axis=1, keepdims=True)
        r_sup = np.exp(log_r_sup)
        r_sup /= (r_sup.sum(axis=1, keepdims=True) + eps)

        if n_xo>0:
            log_r_xo = np.stack([
                np.log(pi[k] + eps)
                + gaussian_logpdf(X_xo, mu_x[k], var_x[k], eps)
                for k in range(K)
            ], axis=1)
            log_r_xo -= log_r_xo.max(axis=1, keepdims=True)
            r_xo = np.exp(log_r_xo)
            r_xo /= (r_xo.sum(axis=1, keepdims=True) + eps)
        else:
            r_xo = np.zeros((0,K))

        if n_yo>0:
            log_r_yo = np.stack([
                np.log(pi[k] + eps)
                + gaussian_logpdf(Y_yo, mu_y[k], var_y[k], eps)
                for k in range(K)
            ], axis=1)
            log_r_yo -= log_r_yo.max(axis=1, keepdims=True)
            r_yo = np.exp(log_r_yo)
            r_yo /= (r_yo.sum(axis=1, keepdims=True) + eps)
        else:
            r_yo = np.zeros((0,K))

        # --- M-step ---
        Nk = r_sup.sum(axis=0) + r_xo.sum(axis=0) + r_yo.sum(axis=0)
        pi_new = Nk / (n_sup + n_xo + n_yo)

        mu_x_new = np.zeros_like(mu_x)
        mu_y_new = np.zeros_like(mu_y)
        var_x_new = np.zeros_like(var_x)
        var_y_new = np.zeros_like(var_y)

        for k in range(K):
            w_x = r_sup[:,k].sum() + r_xo[:,k].sum()
            if w_x>0:
                mu_x_new[k] = (
                    (r_sup[:,k,None]*X_sup).sum(0)
                  + (r_xo[:,k,None]*X_xo).sum(0)
                )/(w_x+eps)

            w_y = r_sup[:,k].sum() + r_yo[:,k].sum()
            if w_y>0:
                mu_y_new[k] = (
                    (r_sup[:,k,None]*Y_sup).sum(0)
                  + (r_yo[:,k,None]*Y_yo).sum(0)
                )/(w_y+eps)

            if w_x>0:
                dx_sup = X_sup - mu_x_new[k]
                dx_xo  = X_xo  - mu_x_new[k] if n_xo>0 else np.zeros((0,d_x))
                sx = (
                    (r_sup[:,k]*np.sum(dx_sup**2,axis=1)).sum()
                  + (r_xo[:,k]*np.sum(dx_xo**2,axis=1)).sum()
                )
                var_x_new[k] = sx/(d_x*(w_x+eps))+eps

            if w_y>0:
                dy_sup = Y_sup - mu_y_new[k]
                dy_yo  = Y_yo  - mu_y_new[k] if n_yo>0 else np.zeros((0,d_y))
                sy = (
                    (r_sup[:,k]*np.sum(dy_sup**2,axis=1)).sum()
                  + (r_yo[:,k]*np.sum(dy_yo**2,axis=1)).sum()
                )
                var_y_new[k] = sy/(d_y*(w_y+eps))+eps

        if (np.max(np.abs(pi_new-pi))<tol
        and np.max(np.abs(mu_x_new-mu_x))<tol
        and np.max(np.abs(mu_y_new-mu_y))<tol):
            pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new
            break

        pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new

    # 4) Inference
    X_test = np.vstack(inference_df['morph_coordinates'])
    Y_true = np.vstack(inference_df['gene_coordinates'])

    log_resp = np.stack([
        np.log(pi[k] + eps)
        + gaussian_logpdf(X_test, mu_x[k], var_x[k], eps)
        for k in range(K)
    ], axis=1)
    log_resp -= log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp /= (resp.sum(axis=1, keepdims=True) + eps)

    Y_pred = resp.dot(mu_y)
    return Y_pred, Y_true


def reversed_em_regression(
    gene_df,
    image_df,
    supervised_df,
    inference_df,
    n_components=3,
    max_iter=100,
    tol=1e-4,
    eps=1e-8
):
    # exactly the same but swap roles of X<->Y
    # X_sup = gene → Y_sup = image, etc.
    X_sup = np.vstack(supervised_df['gene_coordinates'])
    Y_sup = np.vstack(supervised_df['morph_coordinates'])
    X_xo  = np.vstack(gene_df['gene_coordinates'])    if len(gene_df)  else np.empty((0,X_sup.shape[1]))
    Y_yo  = np.vstack(image_df['morph_coordinates'])  if len(image_df) else np.empty((0,Y_sup.shape[1]))

    n_sup, d_x = X_sup.shape
    n_xo = X_xo.shape[0]
    n_yo = Y_yo.shape[0]
    d_y = Y_sup.shape[1]
    K   = n_components

    mu_x = KMeans(K, random_state=0).fit(X_sup).cluster_centers_
    mu_y = KMeans(K, random_state=0).fit(Y_sup).cluster_centers_
    var_x = np.full(K, np.mean(np.var(X_sup,axis=0))+eps)
    var_y = np.full(K, np.mean(np.var(Y_sup,axis=0))+eps)
    pi    = np.full(K, 1.0/K)

    # same EM loop as above...
    for _ in range(max_iter):
        log_r_sup = np.stack([
            np.log(pi[k]+eps)
            + gaussian_logpdf(X_sup, mu_x[k], var_x[k], eps)
            + gaussian_logpdf(Y_sup, mu_y[k], var_y[k], eps)
            for k in range(K)
        ],axis=1)
        log_r_sup -= log_r_sup.max(axis=1,keepdims=True)
        r_sup = np.exp(log_r_sup)
        r_sup /= (r_sup.sum(axis=1,keepdims=True)+eps)

        if n_xo>0:
            log_r_xo = np.stack([
                np.log(pi[k]+eps)
                + gaussian_logpdf(X_xo, mu_x[k], var_x[k], eps)
                for k in range(K)
            ],axis=1)
            log_r_xo -= log_r_xo.max(axis=1,keepdims=True)
            r_xo = np.exp(log_r_xo)
            r_xo /= (r_xo.sum(axis=1,keepdims=True)+eps)
        else:
            r_xo = np.zeros((0,K))

        if n_yo>0:
            log_r_yo = np.stack([
                np.log(pi[k]+eps)
                + gaussian_logpdf(Y_yo, mu_y[k], var_y[k], eps)
                for k in range(K)
            ],axis=1)
            log_r_yo -= log_r_yo.max(axis=1,keepdims=True)
            r_yo = np.exp(log_r_yo)
            r_yo /= (r_yo.sum(axis=1,keepdims=True)+eps)
        else:
            r_yo = np.zeros((0,K))

        Nk = r_sup.sum(axis=0) + r_xo.sum(axis=0) + r_yo.sum(axis=0)
        pi_new = Nk/(n_sup+n_xo+n_yo)

        mu_x_new = np.zeros_like(mu_x)
        mu_y_new = np.zeros_like(mu_y)
        var_x_new = np.zeros_like(var_x)
        var_y_new = np.zeros_like(var_y)

        for k in range(K):
            w_x = r_sup[:,k].sum() + r_xo[:,k].sum()
            if w_x>0:
                mu_x_new[k] = (
                    (r_sup[:,k,None]*X_sup).sum(0)
                  + (r_xo[:,k,None]*X_xo).sum(0)
                )/(w_x+eps)

            w_y = r_sup[:,k].sum() + r_yo[:,k].sum()
            if w_y>0:
                mu_y_new[k] = (
                    (r_sup[:,k,None]*Y_sup).sum(0)
                  + (r_yo[:,k,None]*Y_yo).sum(0)
                )/(w_y+eps)

            if w_x>0:
                dx_sup = X_sup - mu_x_new[k]
                dx_xo  = X_xo  - mu_x_new[k] if n_xo>0 else np.zeros((0,d_x))
                sx = (
                    (r_sup[:,k]*np.sum(dx_sup**2,axis=1)).sum()
                  + (r_xo[:,k]*np.sum(dx_xo**2,axis=1)).sum()
                )
                var_x_new[k] = sx/(d_x*(w_x+eps))+eps

            if w_y>0:
                dy_sup = Y_sup - mu_y_new[k]
                dy_yo  = Y_yo  - mu_y_new[k] if n_yo>0 else np.zeros((0,d_y))
                sy = (
                    (r_sup[:,k]*np.sum(dy_sup**2,axis=1)).sum()
                  + (r_yo[:,k]*np.sum(dy_yo**2,axis=1)).sum()
                )
                var_y_new[k] = sy/(d_y*(w_y+eps))+eps

        if (np.max(np.abs(pi_new-pi))<tol
        and np.max(np.abs(mu_x_new-mu_x))<tol
        and np.max(np.abs(mu_y_new-mu_y))<tol):
            pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new
            break

        pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new

    X_test = np.vstack(inference_df['gene_coordinates'])
    Y_true = np.vstack(inference_df['morph_coordinates'])

    log_resp = np.stack([
        np.log(pi[k]+eps)
        + gaussian_logpdf(X_test, mu_x[k], var_x[k], eps)
        for k in range(K)
    ],axis=1)
    log_resp -= log_resp.max(axis=1,keepdims=True)
    resp = np.exp(log_resp)
    resp /= (resp.sum(axis=1,keepdims=True)+eps)

    Y_pred = resp.dot(mu_y)
    return Y_pred, Y_true

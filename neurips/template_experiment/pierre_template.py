import os
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.random_projection import GaussianRandomProjection
from torchvision import models, transforms
import torch
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
OUT_FRAC        = 0.4    # fraction of "output-only" samples for Y-clustering
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

#############################
# Single‐trial Experiment
#############################
def run_experiment(df, model, tfm, recipes, ing2idx, vocab_size, X, sup_frac, out_frac):
    Y_true = np.vstack(df['ingredient_vec'].values)
    rp     = GaussianRandomProjection(n_components=Y_DIM_REDUCED, random_state=42)
    Y_proj = rp.fit_transform(Y_true)

    # X      = encode_images(df, IMAGE_FOLDER, model, tfm)

    # cluster X for stratification
    x_lab  = cluster_features(X, N_CLUSTERS)
    sup_mask, out_mask, inf_mask = stratified_split_masks(x_lab, sup_frac, out_frac)

    # cluster Y on sup+out
    y_lab = np.empty(len(Y_proj), int)
    y_lab[sup_mask|out_mask] = cluster_ingredients(Y_proj[sup_mask|out_mask], N_CLUSTERS)

    # learn mapping
    mapping   = learn_bridge(x_lab, y_lab, sup_mask, N_CLUSTERS)
    centroids = compute_centroids(Y_true, y_lab, N_CLUSTERS)

    # predict
    Yb_all    = predict_bridge(x_lab, mapping, centroids)
    Yk_all    = knn_baseline(X[sup_mask], Y_true[sup_mask], X[inf_mask])


    # evalute
    bkm_mae = mean_absolute_error(Y_true[inf_mask], Yb_all[inf_mask])
    knn_mae = mean_absolute_error(Y_true[inf_mask], Yk_all)
    return bkm_mae, knn_mae, sup_mask

#############################
# Multi‐trial Evaluation
#############################
def run_all_trials(df, model, tfm, recipes, ing2idx, vocab_size):
    summary = {}
    X      = encode_images(df, IMAGE_FOLDER, model, tfm) 
    for sup_frac in SUP_FRACS:
        key = f"sup={sup_frac:.2%}, out={OUT_FRAC:.2%}"
        b_list, k_list = [], []
        print(f"\n=== Supervised fraction: {sup_frac:.2%} (OUT={OUT_FRAC:.2%}) ===")
        for t in range(N_TRIALS):
            bkm, knn, sup_mask = run_experiment(
                df, model, tfm, recipes, ing2idx, vocab_size,
                X, sup_frac, OUT_FRAC
            )
            b_list.append(bkm); k_list.append(knn)
            print(f"--- Trial {t+1} Results ---")
            print(f"BKM MAE: {bkm:.4f}")
            print(f"KNN MAE: {knn:.4f}")
        avg_b = np.mean(b_list)
        avg_k = np.mean(k_list)
        print("=== Average Results ===")
        print(f"Supervised fraction: {sup_frac:.2%}")
        print(f"BKM: {avg_b:.4f}")
        print(f"KNN: {avg_k:.4f}")
        summary[key] = {'BKM_MAE':avg_b,'KNN_MAE':avg_k}
    return summary

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
    df = df[df['cuisine_type'].isin({'beef_tacos','pizza','ramen'})].reset_index(drop=True)

    model, tfm = get_image_encoder()
    summary = run_all_trials(df, model, tfm, recipes, ing2idx, D)
    print("\n=== Summary for all proportions ===")
    print(pd.DataFrame(summary).T)

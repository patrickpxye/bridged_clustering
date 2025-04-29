# import os
# import warnings
# warnings.filterwarnings("ignore")
# import numpy as np
# import pandas as pd
# from PIL import Image
# from sklearn.cluster import KMeans
# from sklearn.metrics import mean_absolute_error
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.random_projection import GaussianRandomProjection
# from torchvision import models, transforms
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import random

# # Configuration
# BASE_DIR = "input/ingredients_classifier"  # base of your folder
# IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
# INGREDIENTS_FILE = os.path.join(BASE_DIR, "ingredients.txt")
# RECIPES_FILE     = os.path.join(BASE_DIR, "recipes.txt")
# # VAL_IMAGES_FILE  = os.path.join(BASE_DIR, "val_images.txt")
# # VAL_LABELS_FILE  = os.path.join(BASE_DIR, "val_labels.txt")
# ALL_IMAGES_FILE = os.path.join(BASE_DIR, "all_images.txt")
# ALL_LABELS_FILE = os.path.join(BASE_DIR, "all_labels.txt")
# #############################
# # Parameters
# #############################
# N_CLUSTERS = 3              # Latent T space (3: Abstract Exp, Realism, Cubism)
# SUPERVISED_VALUES = [0.01, 0.05, 0.1]  # Percent of labeled (image, year) pairs
# N_TRIALS = 5
# IMAGE_SIZE = 224

# def load_paths(path):
 
#      with open(path, "r") as f:
#          return [line.strip() for line in f if line.strip()]

# # 1) Load master ingredient list (vocab)
# with open(INGREDIENTS_FILE) as f:
#     ingredients = [line.strip() for line in f.read().split(',') if line.strip()]
# ing2idx = {ing: i for i, ing in enumerate(ingredients)}

# # 2) Load recipes mapping (recipe_id -> list of ingredients)
# recipes = []
# with open(RECIPES_FILE) as f:
#     for line in f:
#         row = line.strip()
#         if not row:
#             recipes.append([])
#         else:
#             recipes.append([ing.strip() for ing in row.split(',') if ing.strip()])

# # 3) Image Encoder setup
# def get_image_encoder():
#     transform = transforms.Compose([
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     resnet = models.resnet50(pretrained=True)
#     resnet.fc = torch.nn.Identity()
#     resnet.eval()
#     return resnet, transform

# # 4) Encode images -> feature vectors
# def encode_images(df, image_root, model, transform):
#     features = []
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding images"):
#         img_path = row['img_path']  # e.g. 'apple_pie/42_...jpg'
#         full_path = os.path.join(image_root, img_path)
#         if not os.path.exists(full_path):
#             print(f"Missing: {full_path}")
#             features.append(np.zeros(2048, dtype=float))
#             continue
#         try:
#             img = Image.open(full_path).convert('RGB')
#             x = transform(img).unsqueeze(0)
#             with torch.no_grad():
#                 feat = model(x).squeeze().cpu().numpy()
#             features.append(feat)
#         except Exception as e:
#             print(f"Error reading {full_path}: {e}")
#             features.append(np.zeros(2048, dtype=float))
#     return np.vstack(features)

# # 5) Multi-hot builder
# def make_multihot(recipe_id):
#     vec = np.zeros(len(ingredients), dtype=int)
#     for ing in recipes[recipe_id]:
#         idx = ing2idx.get(ing)
#         if idx is not None:
#             vec[idx] = 1
#     return vec

# # Last) Main: load val split, encode, build vectors
# if __name__ == "__main__":
#     # Load validation split paths & labels
#     # Use load_paths instead of read_csv so commas in filenames don’t break us
#     img_list   = load_paths(ALL_IMAGES_FILE)
#     df_images  = pd.DataFrame({"img_path": img_list})
#     df_labels = pd.read_csv(ALL_LABELS_FILE, header=None, names=['recipe_id'])
#     df = pd.concat([df_images, df_labels], axis=1)

#     # Build ingredient multi-hot
#     df['ingredient_vec'] = df['recipe_id'].apply(make_multihot)
#     # Optionally extract cuisine type
#     df['cuisine_type'] = df['img_path'].str.split('/').str[0]

#     # -----------------------------
#     # 1) Stack into a (N × D) array
#     Y = np.vstack(df["ingredient_vec"].values)   # shape = (N_samples, D_orig)

#     # 2) Choose target dimension (D′) and fit a Gaussian RP
#     D_prime = 128
#     rp = GaussianRandomProjection(n_components=D_prime, random_state=42)
#     Y_proj = rp.fit_transform(Y)                 # shape = (N_samples, 128)

#     # 3) Store the projected vectors back if you like
#     df["ingredient_rp"] = list(Y_proj)

#     print(f"Original Y shape   : {Y.shape}")
#     print(f"Projected Y shape  : {Y_proj.shape}")

#     # -----------------------------

#     # **filter to only apple_pie, pizza, ramen**
#     keep = {"beef_tacos", "pizza", "ramen"}
#     df   = df[df["cuisine_type"].isin(keep)].reset_index(drop=True)

#     # Encode images
#     encoder, transform = get_image_encoder()
#     feats = encode_images(df, IMAGE_FOLDER, encoder, transform)

#     print("DataFrame head:")
#     print(df.head())
#     print("Feature matrix shape:", feats.shape)
#     print("Ingredient vector length:", df['ingredient_vec'].iloc[0].shape)
    
#     # Now `feats[i]` is the 2048-dim image feature for sample i
#     # and `df['ingredient_vec'].iloc[i]` is the multi-hot Y vector.
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
N_TRIALS        = 5
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
            if not row:
                recipes.append([])
            else:
                recipes.append([ing.strip() for ing in row.split(",") if ing.strip()])
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
            print("Missing:", full)
            feats.append(np.zeros(2048, float))
            continue
        img = Image.open(full).convert("RGB")
        x   = tfm(img).unsqueeze(0)
        with torch.no_grad():
            feats.append(model(x).squeeze().cpu().numpy())
    return np.vstack(feats)

#############################
# Label Encoder
#############################
def make_multihot(recipe_id, recipes, ing2idx, D):
    vec = np.zeros(D, int)
    for ing in recipes[recipe_id]:
        idx = ing2idx.get(ing)
        if idx is not None:
            vec[idx] = 1
    return vec

#############################
# Clustering + Bridging
#############################
def cluster_features(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X), km

def cluster_ingredients(Y_proj, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(Y_proj), km

def learn_bridge(x_lab, y_lab, sup_mask, n_clusters):
    mapping = np.zeros(n_clusters, int)
    for i in range(n_clusters):
        mask = (x_lab == i) & sup_mask
        if not mask.any():
            mapping[i] = 0
        else:
            counts = np.bincount(y_lab[mask], minlength=n_clusters)
            mapping[i] = counts.argmax()
    return mapping

def compute_centroids(Y, y_lab, n_clusters):
    centroids = np.zeros((n_clusters, Y.shape[1]), float)
    for j in range(n_clusters):
        members = (y_lab == j)
        if members.sum():
            centroids[j] = Y[members].mean(axis=0)
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
def run_experiment(df, model, tfm, recipes, ing2idx, vocab_size):
    # 1) encode Y
    Y_true = np.vstack(df["ingredient_vec"].values)              # (N, D)
    # 2) random projection for clustering
    rp     = GaussianRandomProjection(n_components=Y_DIM_REDUCED, random_state=42)
    Y_proj = rp.fit_transform(Y_true)                            # (N, D')

    # 3) encode X
    X = encode_images(df, IMAGE_FOLDER, model, tfm)             # (N, 2048)

    # 4) cluster X, cluster Y_proj
    x_lab, _ = cluster_features(X, N_CLUSTERS)
    y_lab, _ = cluster_ingredients(Y_proj, N_CLUSTERS)

    # 5) pick supervised subset
    N = X.shape[0]
    n_sup = max(1, int(frac * N))
    sup_idx = np.zeros(N, bool)
    sup_idx[np.random.choice(N, n_sup, replace=False)] = True

    # 6) learn bridge and centroids
    mapping   = learn_bridge(x_lab, y_lab, sup_idx, N_CLUSTERS)
    centroids = compute_centroids(Y_true, y_lab, N_CLUSTERS)

    # 7) predict
    Yb_cont = predict_bridge(x_lab, mapping, centroids)         # (N, D)
    Yk_cont = knn_baseline(X[sup_idx], Y_true[sup_idx], X)       # (N, D)

    # 8) evaluate MAE
    bkm_mae = mean_absolute_error(Y_true, Yb_cont)
    knn_mae = mean_absolute_error(Y_true, Yk_cont)

    return  bkm_mae, knn_mae, sup_idx

#############################
# Multi‐trial Evaluation
#############################
def run_all_trials(df, model, tfm, recipes, ing2idx, vocab_size):
    results = {f"{f:.2%}": {"BKM": [], "KNN": []} for f in SUP_FRACS}
    for f in SUP_FRACS:
        print("Supervised:", f)
        global frac
        frac = f
        for t in range(N_TRIALS):
            bkm_mae, knn_mae, sup_idx = run_experiment(df, model, tfm, recipes, ing2idx, vocab_size)
            
            # look up the cuisine types
            sup_cuisines = df.loc[sup_idx, 'cuisine_type'].tolist()

            # count how many per cuisine
            counts = Counter(sup_cuisines)
            # build a multi-line string
            counts_str = "\n".join(f"{cuisine}: {cnt} supervised points"
                                for cuisine, cnt in counts.items())

            print(
                f"Trial {t+1}:\n"
                f"{counts_str}\n"
                f"BKM MAE = {bkm_mae:.4f}, KNN MAE = {knn_mae:.4f}"
            )


            results[f"{f:.2%}"]["BKM"].append(bkm_mae)
            results[f"{f:.2%}"]["KNN"].append(knn_mae)

    # summary
    summary = {}
    for f_label, vals in results.items():
        summary[f_label] = {
            "BKM_MAE": np.mean(vals["BKM"]),
            "KNN_MAE": np.mean(vals["KNN"]),
        }
    return summary

#############################
# Main
#############################
if __name__ == "__main__":
    # load vocab & recipes
    vocab, ing2idx = load_ingredients()
    recipes        = load_recipes()
    D              = len(vocab)

    # load all images + labels
    imgs = load_paths(ALL_IMAGES_FILE)
    df   = pd.DataFrame({"img_path": imgs})
    df["recipe_id"] = pd.read_csv(ALL_LABELS_FILE, header=None).iloc[:,0]
    # build labels
    df["ingredient_vec"] = df["recipe_id"].apply(
        lambda rid: make_multihot(rid, recipes, ing2idx, D)
    )
    df["cuisine_type"] = df["img_path"].str.split("/").str[0]
    # optional filter
    df = df[df["cuisine_type"].isin({"beef_tacos","pizza","ramen"})].reset_index(drop=True)

    # get encoder
    model, tfm = get_image_encoder()

    # run evaluation
    summary = run_all_trials(df, model, tfm, recipes, ing2idx, D)
    print(pd.DataFrame(summary).T)

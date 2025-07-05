### Script 2: For Flickr30k dataset
import os
import re
import itertools
import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import CHRF

from k_means_constrained import KMeansConstrained

from baseline import (
    mean_teacher_regression,
    gcn_regression,
    fixmatch_regression,
    laprls_regression,
    tsvr_regression,
    tnnr_regression,
    ucvme_regression,
    rankup_regression
)


### Columns of this dataset:
# - 'image_emb': image embeddings
# - 'caption': text captions
# - 'caption_emb': text embeddings
# - 'img_id': unique image identifier

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load & initial TF–IDF + DBSCAN on captions to get “clusters” for sampling
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/flickr30k.parquet")
df = df.reset_index(drop=True)
df.drop(columns=['sentids', 'split','filename'], inplace=True)
df['_pair'] = df.apply(lambda r: list(zip(r['caption'], r['caption_embs'])), axis=1)
df = df.explode('_pair').reset_index(drop=True)
df[['caption', 'caption_emb']] = pd.DataFrame(df['_pair'].tolist(), index=df.index)
df = df.drop(columns=['caption_embs', '_pair'])
# df = df[:50000]

def tfidf_encode(captions, max_df=0.9, min_df=3, stop_words='english'):
    vect = TfidfVectorizer(max_df=max_df, min_df=min_df,
                           stop_words=stop_words)
    X = vect.fit_transform(captions)
    return X, vect

X_tfidf, vect = tfidf_encode(df['caption'])
db = DBSCAN(eps=0.6, min_samples=12, metric='euclidean')
df['cluster'] = db.fit_predict(X_tfidf)

df_valid     = df[df['cluster'] != -1].copy()
cluster_sz   = df_valid['cluster'].value_counts()
eligible     = cluster_sz[cluster_sz >= 25].index
print("Eligible clusters:")
print(cluster_sz.loc[eligible].sort_values(ascending=False))


df_pruned = df_valid[df_valid['cluster'].isin(eligible)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3) Rename embedding columns to match Script 1’s “x”, “yv”, “y”
# ─────────────────────────────────────────────────────────────────────────────

df_pruned['x']  = df_pruned['image_emb'].apply(lambda arr: np.array(arr))
df_pruned['yv'] = df_pruned['caption_emb'].apply(lambda arr: np.array(arr))
df_pruned['y']  = df_pruned['caption']

CAPTION_MAP = {}
for img_id, grp in df_pruned.groupby('img_id'):
    CAPTION_MAP[img_id] = {
        "embs" : np.stack(grp['caption_emb'].apply(np.array)),  # (5 , d)
        "texts": grp['caption'].tolist()                        # 5 strings
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) Copy in all of the experiment functions from Script 1 (no changes) 
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_caption(pred_vec: np.ndarray, img_id):
    """Return (best_emb, best_text) for this prediction."""
    entry  = CAPTION_MAP[img_id]
    embs   = entry["embs"]
    idx    = np.linalg.norm(embs - pred_vec, axis=1).argmin()
    return embs[idx], entry["texts"][idx]

def align_preds(pred_embs, img_ids):
    """Vectorised wrapper used by every model evaluation."""
    best_embs, best_texts = zip(*(_nearest_caption(p, i)
                                  for p, i in zip(pred_embs, img_ids)))
    return np.vstack(best_embs), list(best_texts)

def eval_model(pred_embs, pred_texts, img_ids):
    # pick the closest of the 5 GT captions
    act_embs, act_texts = align_preds(pred_embs, img_ids)

    mae, mse   = evaluate_regression_loss(pred_embs, act_embs)
    txt_metrics = evaluate_text_metrics(act_texts, pred_texts)
    return mae, mse, txt_metrics


def get_data(df, supervised_ratio, output_only_ratio, K=None, seed=None):
    """
    Split each 'cluster' (family/chunk) in df into supervised/inference/output-only
    according to the fixed ratios. Guarantees at least one supervised sample per cluster.
    Returns:
      sup_df, inf_df, out_df, X_for_clustering, Y_for_clustering
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    sup_list, inf_list, out_list = [], [], []

    # For each cluster value, shuffle & slice by ratios
    for cluster_val, group in df.groupby('cluster'):
        n = len(group)
        # compute counts
        n_sup = max(1, int(np.floor(supervised_ratio * n)))
        n_out = int(np.floor(output_only_ratio   * n))
        # avoid overshoot
        if n_sup + n_out >= n:
            n_sup = max(1, n - 2)
            n_out = 1
        n_inf = n - n_sup - n_out

        # shuffle once
        subseed = int(rng.integers(0, 2**32))
        perm = group.sample(frac=1, random_state=subseed).reset_index(drop=True)
        # slice
        sup = perm.iloc[       : n_sup          ]
        inf = perm.iloc[n_sup  : n_sup + n_inf  ]
        out = perm.iloc[n_sup + n_inf :           ]

        sup_list.append(sup)
        inf_list.append(inf)
        out_list.append(out)

    # concatenate all clusters back together
    sup_df = pd.concat(sup_list, ignore_index=True)
    inf_df = pd.concat(inf_list, ignore_index=True)
    out_df = pd.concat(out_list, ignore_index=True)

    # for clustering, inputs = inference + supervised; outputs = output-only + supervised
    X_for_clustering = pd.concat([inf_df, sup_df], ignore_index=True)
    Y_for_clustering = pd.concat([out_df, sup_df], ignore_index=True)

    return sup_df, inf_df, out_df, X_for_clustering, Y_for_clustering


def perform_clustering(Xc, Yc, K):
    Xm = np.vstack(Xc["x"].values)
    nX = Xm.shape[0]
    size_min_x = nX // K
    size_max_x = int(np.ceil(nX / K))
    x_kmc = KMeansConstrained(n_clusters=K, size_min=size_min_x,
                              size_max=size_max_x, random_state=42)
    x_cl = x_kmc.fit_predict(Xm)

    Ym = np.vstack(Yc["yv"].values)
    nY = Ym.shape[0]
    size_min_y = nY // K
    size_max_y = int(np.ceil(nY / K))
    y_kmc = KMeansConstrained(n_clusters=K, size_min=size_min_y,
                              size_max=size_max_y, random_state=42)
    y_cl = y_kmc.fit_predict(Ym)

    return x_cl, y_cl

def clustering_quality_metrics(Xc, x_cl, Yc, y_cl):
    Xm = np.vstack(Xc["x"].values)
    Ym = np.vstack(Yc["yv"].values)
    return {
        "input_silhouette":      silhouette_score(Xm, x_cl),
        "input_davies_bouldin":  davies_bouldin_score(Xm, x_cl),
        "output_silhouette":     silhouette_score(Ym, y_cl),
        "output_davies_bouldin": davies_bouldin_score(Ym, y_cl),
    }

def decisionVector(sample, x_column, y_column, dim=5):
    assoc = np.zeros((dim, dim), int)
    for i in range(dim):
        for j in range(dim):
            assoc[i,j] = np.sum((sample[x_column]==i)&(sample[y_column]==j))
    dec = np.zeros(dim, int)
    for i in range(dim):
        dec[i] = np.argmax(assoc[i])
    return dec

def build_decision_matrix(supervised_samples, x_clusters, y_clusters, K):
    """
    Build the decision matrix (association vector) using the supervised samples.
    """
    N_sup = len(supervised_samples)
    supervised_samples['x_cluster'] = x_clusters[-N_sup:]
    supervised_samples['y_cluster'] = y_clusters[-N_sup:]
    
    decision_matrix = decisionVector(supervised_samples, x_column='x_cluster', y_column='y_cluster', dim=K)
    return decision_matrix

def build_true_decision_vector(Xc, Yc, x_clusters, y_clusters, K):
    """
    Build an oracle decision vector for Script 1 by majority‐voting
    over the *entire* Xc/Yc (inference+supervised and output-only+supervised).

    Xc           : DataFrame (must include original df['cluster'])
    Yc           : DataFrame (must include original df['cluster'])
    x_clusters   : array-like of length len(Xc)
    y_clusters   : array-like of length len(Yc)
    K            : number of bridged clusters

    Returns
    -------
    true_vec : np.ndarray, shape (K,)
               For each bridged image-cluster i, the y-cluster whose majority
               z-cluster matches i’s majority z-cluster.
    """
    # attach the bridged assignments
    Xc2 = Xc.copy()
    Xc2['x_cluster'] = x_clusters
    Yc2 = Yc.copy()
    Yc2['y_cluster'] = y_clusters

    # 1) image_cluster → majority original z-cluster
    image_to_z = (
        Xc2
        .groupby('x_cluster')['cluster']
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    # 2) y_cluster → majority original z-cluster
    y_to_z = (
        Yc2
        .groupby('y_cluster')['cluster']
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    # 3) invert y_to_z → z_to_y
    z_to_y = { z: y for y, z in y_to_z.items() }

    # 4) assemble the vector
    true_vec = np.full(K, -1, dtype=int)
    for i in range(K):
        z = image_to_z.get(i)
        if z is not None:
            true_vec[i] = z_to_y.get(z, -1)

    return true_vec

def compute_y_centroids(Y_for_clustering, y_clusters, K):
    Y = Y_for_clustering.copy()
    Y['y_cluster'] = y_clusters
    centroids, text_prototypes = [], []
    for c in range(K):
        cluster_data = Y[Y['y_cluster'] == c]
        if len(cluster_data) > 0:
            yvs = np.stack(cluster_data['yv'].values)
            centroid = np.mean(yvs, axis=0)
            dists = np.linalg.norm(yvs - centroid, axis=1)
            idx = np.argmin(dists)
            prototype_text = cluster_data['y'].values[idx]
        else:
            centroid = np.zeros(Y['yv'].iloc[0].shape)
            prototype_text = ""
        centroids.append(centroid)
        text_prototypes.append(prototype_text)
    return np.array(centroids), text_prototypes

def perform_inference(inference_samples, image_clusters, decision_matrix, centroids, text_prototypes):
    inf = inference_samples.copy()
    inf['x_cluster'] = image_clusters[:len(inf)]
    inf['predicted_y_cluster'] = inf['x_cluster'].apply(lambda x: decision_matrix[x])
    inf['predicted_yv'] = inf['predicted_y_cluster'].apply(lambda j: centroids[j])
    inf['predicted_text'] = inf['predicted_y_cluster'].apply(lambda j: text_prototypes[j])
    return inf

def knn_regression(supervised_df, inference_df, n_neighbors=10):
    X_train = np.vstack(supervised_df['x'].values)
    y_train = np.vstack(supervised_df['yv'].values)
    texts_train = supervised_df['y'].values.tolist()
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    X_test = np.vstack(inference_df['x'].values)
    pred_emb = knn.predict(X_test)
    act_emb = np.vstack(inference_df['yv'].values)
    actual_texts = inference_df['y'].tolist()
    pred_texts = []
    for emb in pred_emb:
        dists = np.linalg.norm(y_train - emb, axis=1)
        idx = np.argmin(dists)
        pred_texts.append(texts_train[idx])
    return pred_emb, act_emb, pred_texts, actual_texts

def evaluate_regression_loss(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    return mae, mse

def evaluate_text_metrics(actual_texts, predicted_texts):
    """
    Compute BLEU, BERTScore, and chrF between actual and predicted texts.
    Ensure inputs are lists of Python strings.
    """
    # Convert pandas series or other iterables to list of str
    actual_texts = list(actual_texts)
    predicted_texts = list(predicted_texts)
    smoothie = SmoothingFunction().method1
    metrics = {}
    # BLEU
    bleu_scores = []
    for ref, hyp in zip(actual_texts, predicted_texts):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        bleu_scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
    metrics['BLEU'] = float(np.mean(bleu_scores))
    # chrF (corpus-level)
    refs = [[r] for r in actual_texts]
    _chrf_metric = CHRF()
    result = _chrf_metric.corpus_score(predicted_texts, refs)
    metrics['chrF'] = float(result.score / 100.0)
    return metrics

def _wrap_text_baseline(baseline_fn, sup_df, inf_df):
    """
    Call baseline_fn(supervised, inference) on embeddings, then
    recover actual_texts and predicted_texts via nearest‐neighbor lookup
    in sup_df['y'].
    """
    # 1) rename to Script 0’s expected columns
    sup = sup_df.rename(columns={'x':'morph_coordinates', 'yv':'gene_coordinates'})
    inf = inf_df.rename(columns={'x':'morph_coordinates', 'yv':'gene_coordinates'})

    # 2) run the numeric regressor

    mt_preds, mt_actuals = mean_teacher_regression(sup, inf, lr=0.001, w_max=0.1,alpha=0.95,ramp_len=50)
    gc_preds, gc_actuals = gcn_regression       (sup, inf,dropout=0.3, hidden=128,lr=0.001)
    fx_preds, fx_actuals = fixmatch_regression  (sup, inf,alpha_ema=0.99,batch_size=32,conf_threshold=0.05,lambda_u_max=1.0,lr=0.0001,rampup_length=30)
    lp_preds, lp_actuals = laprls_regression    (sup, inf,gamma=0.001,k=5,lam=0.1,sigma=2.0)
    ts_preds, ts_actuals = tsvr_regression      (sup, inf, C=1.0, epsilon=0.01, gamma='scale', self_training_frac=0.1)
    tn_preds, tn_actuals = tnnr_regression     (sup, inf, beta=1.0,lr=0.0001, rep_dim=128)
    uv_preds, uv_actuals = ucvme_regression    (sup, inf,lr=0.001,mc_T=5,w_unl=5)
    ru_preds, ru_actuals = rankup_regression   (sup, inf, alpha_rda=0.1, hidden_dim=512, lr=0.0001, tau=0.95, temperature=1.0)

    if baseline_fn.__name__ == 'mean_teacher_regression':
        preds, actuals = mt_preds, mt_actuals
    elif baseline_fn.__name__ == 'gcn_regression':
        preds, actuals = gc_preds, gc_actuals
    elif baseline_fn.__name__ == 'fixmatch_regression':
        preds, actuals = fx_preds, fx_actuals
    elif baseline_fn.__name__ == 'laprls_regression':
        preds, actuals = lp_preds, lp_actuals
    elif baseline_fn.__name__ == 'tsvr_regression':
        preds, actuals = ts_preds, ts_actuals
    elif baseline_fn.__name__ == 'tnnr_regression':
        preds, actuals = tn_preds, tn_actuals
    elif baseline_fn.__name__ == 'ucvme_regression':
        preds, actuals = uv_preds, uv_actuals
    elif baseline_fn.__name__ == 'rankup_regression':
        preds, actuals = ru_preds, ru_actuals
    else:
        raise ValueError(f"Unknown baseline function: {baseline_fn.__name__}")

    # 3) get actual texts
    actual_texts = inf_df['y'].tolist()

    # 4) for each predicted embedding, find nearest supervised text
    train_emb   = np.vstack(sup['gene_coordinates'])
    train_texts = sup_df['y'].tolist()

    pred_texts = []
    for e in preds:
        idx = np.argmin(np.linalg.norm(train_emb - e, axis=1))
        pred_texts.append(train_texts[idx])

    return preds, actuals, actual_texts, pred_texts

##### KMM Section

from baseline import kernel_mean_matching_regression, reversed_kernel_mean_matching_regression

from baseline import em_regression, reversed_em_regression


def run_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int = None,
):
    """
    Forward pipeline for Wikipedia‐style dataset:
      • Split into supervised, inference, output‐only
      • Size‐constrained KMeans on (inference+supervised) for x and (output-only+supervised) for y
      • Build bridged decision matrix and do Bridged inference
      • Evaluate BKM, KNN, MeanTeacher, FixMatch, LapRLS, TSVR, TNNR, UCVME, RankUp, GCN, KMM, EM
      • Returns dict with 'clustering', 'regression', and 'text' metrics
    """
    # 1) split
    sup_df, inf_df, out_df, Xc, Yc = get_data(df, supervised_ratio, output_only_ratio, K, seed)

    # 2) clustering
    x_cl, y_cl = perform_clustering(Xc, Yc, K)

    # 3) clustering quality
    ami_x = adjusted_mutual_info_score(Xc["cluster"], x_cl)
    ami_y = adjusted_mutual_info_score(Yc["cluster"], y_cl)
    decision = build_decision_matrix(sup_df, x_cl, y_cl, K)
    true_decision = build_true_decision_vector(Xc, Yc, x_cl, y_cl, K)
    accuracy = np.mean(decision == true_decision)

    # 4) bridged inference
    cents, texts = compute_y_centroids(Yc, y_cl, K)
    inf_res = perform_inference(inf_df, x_cl, decision, cents, texts)

    # 5) BKM baseline
    bkm_pred_emb = np.vstack(inf_res["predicted_yv"].values)
    bkm_act_emb  = np.vstack(inf_res["yv"].values)
    # bkm_mae, bkm_mse = evaluate_regression_loss(bkm_pred_emb, bkm_act_emb)
    # text_metrics_bkm = evaluate_text_metrics(inf_res["y"], inf_res["predicted_text"])

    # 6) KNN baseline
    knn_pred_emb, knn_act_emb, knn_pred_texts, knn_act_texts = knn_regression(sup_df, inf_df, knn_neighbors)
    # knn_mae, knn_mse = evaluate_regression_loss(knn_pred_emb, knn_act_emb)
    # text_metrics_knn = evaluate_text_metrics(knn_act_texts, knn_pred_texts)

    # 7) _wrap_text_baseline methods
    mt_pred, mt_act, mt_text_act, mt_text_pred = _wrap_text_baseline(mean_teacher_regression, sup_df, inf_df)
    fm_pred, fm_act, fm_text_act, fm_text_pred = _wrap_text_baseline(fixmatch_regression, sup_df, inf_df)
    lap_pred, lap_act, lap_text_act, lap_text_pred = _wrap_text_baseline(laprls_regression, sup_df, inf_df)
    tsvr_pred, tsvr_act, tsvr_text_act, tsvr_text_pred = _wrap_text_baseline(tsvr_regression, sup_df, inf_df)
    tnnr_pred, tnnr_act, tnnr_text_act, tnnr_text_pred = _wrap_text_baseline(tnnr_regression, sup_df, inf_df)
    ucv_pred, ucv_act, ucv_text_act, ucv_text_pred = _wrap_text_baseline(ucvme_regression, sup_df, inf_df)
    rank_pred, rank_act, rank_text_act, rank_text_pred = _wrap_text_baseline(rankup_regression, sup_df, inf_df)
    gcn_pred, gcn_act, gcn_text_act, gcn_text_pred = _wrap_text_baseline(gcn_regression, sup_df, inf_df)

    # 8) KMM forward on full marginals
    X_kmm = pd.concat([inf_df, sup_df], ignore_index=True).rename(columns={'x':'morph_coordinates'})
    Y_kmm = pd.concat([out_df, sup_df], ignore_index=True).rename(columns={'yv':'gene_coordinates'})
    sup_kmm = sup_df.rename(columns={'x':'morph_coordinates','yv':'gene_coordinates'})
    inf_kmm = inf_df.rename(columns={'x':'morph_coordinates','yv':'gene_coordinates'})
    kmm_pred_emb, kmm_act_emb = kernel_mean_matching_regression(
        image_df      = X_kmm,
        gene_df       = Y_kmm,
        supervised_df = sup_kmm,
        inference_df  = inf_kmm,
        alpha           = 0.1,
        kmm_B           = 100,
        kmm_eps         = 0.001,
        sigma           = 1.0
    )
    train_embs, train_texts = np.vstack(sup_df['yv']), sup_df['y'].tolist()
    kmm_pred_texts = [train_texts[np.argmin(np.linalg.norm(train_embs - e,axis=1))] for e in kmm_pred_emb]

    # 9) EM forward on full marginals
    em_pred_emb, em_act_emb = em_regression(
        supervised_df = sup_kmm,
        image_df      = X_kmm,
        gene_df       = Y_kmm,
        inference_df  = inf_kmm,
        n_components  = K,
        eps             = 0.001,
        max_iter        = 100,
        tol             = 0.0001
    )
    em_pred_texts = [train_texts[np.argmin(np.linalg.norm(train_embs - e,axis=1))] for e in em_pred_emb]
    em_text_metrics = evaluate_text_metrics(inf_df['y'], em_pred_texts)

    # ids of the inference rows (same order as every pred array)
    inf_ids = inf_df['img_id'].values

    # ── Bridged/BKM ────────────────────────────────────────────────────────────
    bkm_mae, bkm_mse, text_metrics_bkm = eval_model(
        bkm_pred_emb,
        inf_res["predicted_text"].tolist(),
        inf_ids
    )

    # ── KNN ────────────────────────────────────────────────────────────────────
    knn_mae, knn_mse, text_metrics_knn = eval_model(
        knn_pred_emb,
        knn_pred_texts,
        inf_ids
    )

    # ── baselines via _wrap_text_baseline() ────────────────────────────────────
    mt_mae, mt_mse, mt_text_metrics   = eval_model(mt_pred,   mt_text_pred,   inf_ids)
    fm_mae, fm_mse, fm_text_metrics   = eval_model(fm_pred,   fm_text_pred,   inf_ids)
    lap_mae,lap_mse,lap_text_metrics  = eval_model(lap_pred,  lap_text_pred,  inf_ids)
    tsvr_mae,tsvr_mse,tsvr_text_metrics = eval_model(tsvr_pred,tsvr_text_pred,inf_ids)
    tnnr_mae,tnnr_mse,tnnr_text_metrics = eval_model(tnnr_pred,tnnr_text_pred,inf_ids)
    ucv_mae, ucv_mse, ucv_text_metrics = eval_model(ucv_pred, ucv_text_pred,  inf_ids)
    rank_mae,rank_mse,rank_text_metrics = eval_model(rank_pred,rank_text_pred,inf_ids)
    gcn_mae, gcn_mse, gcn_text_metrics = eval_model(gcn_pred, gcn_text_pred,  inf_ids)

    # ── KMM & EM (same idea) ───────────────────────────────────────────────────
    kmm_mae, kmm_mse, kmm_text_metrics = eval_model(
        kmm_pred_emb, kmm_pred_texts, inf_ids
    )
    em_mae, em_mse, em_text_metrics   = eval_model(
        em_pred_emb,  em_pred_texts,  inf_ids
    )


    # 10) package metrics
    metrics = {
        'clustering': {
            'AMI_X': ami_x,
            'AMI_Y': ami_y,
            'Bridging Accuracy': accuracy
        },
        'regression': {
            'BKM':        {'MAE': bkm_mae,  'MSE': bkm_mse},
            'KNN':        {'MAE': knn_mae,  'MSE': knn_mse},
            'MeanTeacher':{'MAE': mt_mae,   'MSE': mt_mse},
            'FixMatch':   {'MAE': fm_mae,   'MSE': fm_mse},
            'LapRLS':     {'MAE': lap_mae,  'MSE': lap_mse},
            'TSVR':       {'MAE': tsvr_mae,'MSE': tsvr_mse},
            'TNNR':       {'MAE': tnnr_mae,'MSE': tnnr_mse},
            'UCVME':      {'MAE': ucv_mae, 'MSE': ucv_mse},
            'RankUp':     {'MAE': rank_mae,'MSE': rank_mse},
            'GCN':        {'MAE': gcn_mae, 'MSE': gcn_mse},
            'KMM':        {'MAE': kmm_mae, 'MSE': kmm_mse},
            'EM':         {'MAE': em_mae,  'MSE': em_mse},
        },
        'text': {
            'BKM':         text_metrics_bkm,
            'KNN':         text_metrics_knn,
            'MeanTeacher': mt_text_metrics,
            'FixMatch':    fm_text_metrics,
            'LapRLS':      lap_text_metrics,
            'TSVR':        tsvr_text_metrics,
            'TNNR':        tnnr_text_metrics,
            'UCVME':       ucv_text_metrics,
            'RankUp':      rank_text_metrics,
            'GCN':         gcn_text_metrics,
            'KMM':         kmm_text_metrics,
            'EM':          em_text_metrics,
        }
    }
    # print the bleu scores
    bleu_scores = {k: v['BLEU'] for k, v in metrics['text'].items()}
    print("BLEU scores:")
    for model, score in bleu_scores.items():
        print(f"{model}: {score:.4f}")
    return metrics


def run_reversed_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int = None,
):
    """
    Mirror of run_experiment:
      • Supervised, inference, input-only (images) via get_data()
      • Cluster on text (yv) then on image (x)
      • Build decision vector from text→image clusters
      • Predict image embeddings from text
      • Evaluate Bridged + KNN + MeanTeacher + GCN + FixMatch + LapRLS + TSVR + TNNR + UCVME + RankUp + KMM + EM

    Returns a dict with:
      - 'clustering': {'AMI_text', 'AMI_image', 'Bridging Accuracy'}
      - 'regression': { ... all MAE/MSE for each model ... }
    """
    from collections import Counter

    # 1) split
    sup_df, inf_df, input_only_df, Xc, Yc = get_data(
        df, supervised_ratio, output_only_ratio, K, seed
    )

    # 2) build clustering sets
    Xc_rev = pd.concat([inf_df, sup_df], ignore_index=True)        # text‐only + supervised
    Yc_rev = pd.concat([input_only_df, sup_df], ignore_index=True) # image‐only + supervised

    # 3) cluster on text (yv)
    T = np.vstack(Xc_rev["yv"].values)
    size_min_t = len(T) // K
    size_max_t = int(np.ceil(len(T) / K))
    km_text = KMeansConstrained(n_clusters=K, size_min=size_min_t, size_max=size_max_t, random_state=42).fit(T)
    text_clusters = km_text.labels_

    # 4) cluster on image (x)
    I = np.vstack(Yc_rev["x"].values)
    size_min_i = len(I) // K
    size_max_i = int(np.ceil(len(I) / K))
    km_img = KMeansConstrained(n_clusters=K, size_min=size_min_i, size_max=size_max_i, random_state=42).fit(I)
    img_clusters = km_img.labels_

    # 5) cluster quality
    ami_text  = adjusted_mutual_info_score(Xc_rev["cluster"], text_clusters)
    ami_image = adjusted_mutual_info_score(Yc_rev["cluster"], img_clusters)

    # 6) build text→image decision vector
    sup_block = sup_df.copy()
    sup_block["text_cluster"] = text_clusters[-len(sup_df):]
    sup_block["img_cluster"]  = img_clusters[-len(sup_df):]
    decision_vec = decisionVector(sup_block, "text_cluster", "img_cluster", dim=K)

    # 7) oracle for reversed mapping
    true_vec  = build_true_decision_vector(Xc_rev, Yc_rev, text_clusters, img_clusters, K)
    oracle_rev = np.full(K, -1, dtype=int)
    for img_c, txt_c in enumerate(true_vec):
        if txt_c >= 0:
            oracle_rev[txt_c] = img_c
    decision_acc = (decision_vec == oracle_rev).mean()

    # 8) compute image‐cluster centroids
    Yc_rev = Yc_rev.copy()
    Yc_rev["img_cluster"] = img_clusters
    Yc_rev["x"]           = np.vstack(Yc_rev["x"].values).tolist()
    img_cents = []
    for c in range(K):
        pts = (
            np.stack(Yc_rev[Yc_rev["img_cluster"] == c]["x"].values)
            if (Yc_rev["img_cluster"] == c).any()
            else np.zeros(I.shape[1])
        )
        img_cents.append(pts.mean(axis=0))
    img_cents = np.vstack(img_cents)

    # 9) bridged‐reversed inference
    inf = inf_df.copy()
    inf["text_cluster"]        = text_clusters[: len(inf)]
    inf["predicted_img_cluster"] = inf["text_cluster"].map(lambda t: decision_vec[t])
    inf["pred_x"]              = inf["predicted_img_cluster"].map(lambda c: img_cents[c])
    bridged_preds  = np.vstack(inf["pred_x"].values)
    bridged_actual = np.vstack(inf["x"].values)

    # 10) prepare marginals for KMM & EM
    gene_df_rev  = Xc_rev.rename(columns={'yv': 'gene_coordinates'}).copy()  # text→ gene_coordinates
    image_df_rev = Yc_rev.rename(columns={'x':  'morph_coordinates'}).copy() # image→ morph_coordinates

    sup_rev_reg = sup_df.rename(columns={'yv': 'gene_coordinates',
                                         'x':  'morph_coordinates'}).copy()
    inf_rev_reg = inf_df.rename(columns={'yv': 'gene_coordinates',
                                         'x':  'morph_coordinates'}).copy()

    # ── reversed KMM ──
    kmm_rev_pred_emb, kmm_rev_act_emb = reversed_kernel_mean_matching_regression(
        gene_df       = gene_df_rev,
        image_df      = image_df_rev,
        supervised_df = sup_rev_reg,
        inference_df  = inf_rev_reg,
        alpha           = 0.1,
        kmm_B           = 100,
        kmm_eps         = 0.001,
        sigma           = 1.0
    )

    # ── reversed EM ──
    em_rev_pred_emb, em_rev_act_emb = reversed_em_regression(
        gene_df       = gene_df_rev,
        image_df      = image_df_rev,
        supervised_df = sup_rev_reg,
        inference_df  = inf_rev_reg,
        n_components  = K,
        eps             = 0.001,
        max_iter        = 100,
        tol             = 0.0001
    )

    # 11) prepare sup/inf for other baselines (text→image)
    sup_rev = sup_df.rename(columns={'yv': 'morph_coordinates',
                                     'x':  'gene_coordinates'}).copy()
    inf_rev = inf_df.rename(columns={'yv': 'morph_coordinates',
                                     'x':  'gene_coordinates'}).copy()

    # KNN
    knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
    knn.fit(np.vstack(sup_rev["morph_coordinates"]), np.vstack(sup_rev["gene_coordinates"]))
    knn_preds = knn.predict(np.vstack(inf_rev["morph_coordinates"]))
    y_te       = np.vstack(inf_rev["gene_coordinates"])

    # other baselines
    mt_preds, mt_actuals = mean_teacher_regression(sup_rev, inf_rev, lr=0.001, w_max=0.5,alpha=0.95,ramp_len=1)
    gc_preds, gc_actuals = gcn_regression       (sup_rev, inf_rev,dropout=0.0, hidden=32,lr=0.003)
    fx_preds, fx_actuals = fixmatch_regression  (sup_rev, inf_rev,alpha_ema=0.999,batch_size=32,conf_threshold=0.05,lambda_u_max=1.0,lr=0.0003,rampup_length=30)
    lp_preds, lp_actuals = laprls_regression    (sup_rev, inf_rev,gamma=1,k=10,lam=0.1,sigma=0.5)
    ts_preds, ts_actuals = tsvr_regression      (sup_rev, inf_rev, C=1.0, epsilon=0.01, gamma='scale', self_training_frac=0.5)
    tn_preds, tn_actuals = tnnr_regression     (sup_rev, inf_rev, beta=0.1,lr=0.0003, rep_dim=128)
    uv_preds, uv_actuals = ucvme_regression    (sup_rev, inf_rev,lr=0.0003,mc_T=5,w_unl=1)
    ru_preds, ru_actuals = rankup_regression   (sup_rev, inf_rev, alpha_rda=0.01, hidden_dim=512, lr=0.001, tau=0.95, temperature=0.5)

    # 12) collect errors
    def eval_(p, a):
        return mean_absolute_error(a, p), mean_squared_error(a, p)

    errors = {}
    mses   = {}

    errors["BKM"],         mses["BKM"]         = eval_(bridged_preds,  bridged_actual)
    errors["KNN"],         mses["KNN"]         = eval_(knn_preds,      y_te)
    errors["MeanTeacher"], mses["MeanTeacher"] = eval_(mt_preds,       mt_actuals)
    errors["GCN"],         mses["GCN"]         = eval_(gc_preds,       gc_actuals)
    errors["FixMatch"],    mses["FixMatch"]    = eval_(fx_preds,       fx_actuals)
    errors["LapRLS"],      mses["LapRLS"]      = eval_(lp_preds,       lp_actuals)
    errors["TSVR"],        mses["TSVR"]        = eval_(ts_preds,       ts_actuals)
    errors["TNNR"],        mses["TNNR"]        = eval_(tn_preds,       tn_actuals)
    errors["UCVME"],       mses["UCVME"]       = eval_(uv_preds,       uv_actuals)
    errors["RankUp"],      mses["RankUp"]      = eval_(ru_preds,       ru_actuals)
    errors["KMM"],         mses["KMM"]         = eval_(kmm_rev_pred_emb, kmm_rev_act_emb)
    errors["EM"],          mses["EM"]          = eval_(em_rev_pred_emb,  em_rev_act_emb)

    #print errors per model
    print("Errors per model:")
    for model, error in errors.items():
        print(f"{model}: MAE={error:.4f}, MSE={mses[model]:.4f}")

    return {
        "clustering": {
            "AMI_X":           ami_text,
            "AMI_Y":          ami_image,
            "Bridging Accuracy":  decision_acc,
        },
        "regression": {
            "BKM":        {"MAE": errors["BKM"],   "MSE": mses["BKM"]},
            "KNN":        {"MAE": errors["KNN"],   "MSE": mses["KNN"]},
            "MeanTeacher":{"MAE": errors["MeanTeacher"], "MSE": mses["MeanTeacher"]},
            "GCN":        {"MAE": errors["GCN"],   "MSE": mses["GCN"]},
            "FixMatch":   {"MAE": errors["FixMatch"],  "MSE": mses["FixMatch"]},
            "LapRLS":     {"MAE": errors["LapRLS"],   "MSE": mses["LapRLS"]},
            "TSVR":       {"MAE": errors["TSVR"],   "MSE": mses["TSVR"]},
            "TNNR":       {"MAE": errors["TNNR"],   "MSE": mses["TNNR"]},
            "UCVME":      {"MAE": errors["UCVME"],  "MSE": mses["UCVME"]},
            "RankUp":     {"MAE": errors["RankUp"], "MSE": mses["RankUp"]},
            "KMM":        {"MAE": errors["KMM"],   "MSE": mses["KMM"]},
            "EM":         {"MAE": errors["EM"],    "MSE": mses["EM"]},
        },
    }



if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(
        description="Run experiments (forward or reversed)"
    )
    parser.add_argument(
        '--reversed',
        action='store_true',
        help='If set, run run_reversed_experiments instead of run_experiments'
    )
    args = parser.parse_args()
    runner = run_reversed_experiment if args.reversed else run_experiment
    experiment_key = "flick_reversed" if args.reversed else "flick"

    # Experiment grid
    K_values   = [3, 4, 5, 6, 7]
    sup_values = [1, 2, 3, 4]            # sup_per_cluster
    out_only   = 0.2
    cluster_sz = 25
    seeds = list(range(30))  # 10 random seeds

    pruned_counts    = df_pruned['cluster'].value_counts()
    eligible_pruned  = pruned_counts[pruned_counts >= cluster_sz].index
    eligible_pruned = eligible_pruned[3:]

    print(f"Eligible clusters indeces and sizes: {eligible_pruned.tolist()}")

    # Which models to record (must match keys in run_experiment’s metrics)
    models = [
      'BKM', 'KNN',
      'MeanTeacher', 'GCN', 'FixMatch',
      'LapRLS', 'TSVR', 'TNNR', 'UCVME',
      'RankUp', 'KMM', 'EM'
    ]
    nK      = len(K_values)
    nSup    = len(sup_values)
    nModels = len(models)
    nTrials = len(seeds)

    # Make results folder
    os.makedirs("results", exist_ok=True)

    # Preallocate arrays
    ami_x     = np.empty((nK, nSup, nTrials))
    ami_y     = np.empty((nK, nSup, nTrials))
    accuracy  = np.empty((nK, nSup, nTrials))

    mae       = np.empty((nK, nSup, nModels, nTrials))
    mse       = np.empty((nK, nSup, nModels, nTrials))
    bleu      = np.empty((nK, nSup, nModels, nTrials))
    chrf      = np.empty((nK, nSup, nModels, nTrials))

    # Main loops
    for i, K in enumerate(K_values):
        for j, sup_per in enumerate(sup_values):
            for t, s in enumerate(seeds):
                seed = s + i * nK + j * nK * nSup  # unique seed for each K, sup_per, and trial
                # ── sample K clusters from pruned set ──
                rng = np.random.default_rng(seed)

                # choose clusters
                chosen = rng.choice(eligible_pruned, size=K, replace=False)

                # sample each cluster independently
                samples = []
                for c in chosen:
                    sub = df_pruned[df_pruned['cluster'] == c]
                    subseed = int(rng.integers(0, 2**32))
                    samples.append(sub.sample(cluster_sz, random_state=subseed))
                sample = pd.concat(samples, ignore_index=True)

                # ── 2) run holistic experiment ──
                metrics = runner(
                    sample,
                    supervised_ratio   = sup_per/cluster_sz,
                    output_only_ratio  = out_only,
                    K                   = K,
                    knn_neighbors      = sup_per,
                    seed                = seed
                )

                # store clustering AMI
                ami_x[i, j, t] = metrics['clustering']['AMI_X']
                ami_y[i, j, t] = metrics['clustering']['AMI_Y']
                accuracy[i, j, t] = metrics['clustering']['Bridging Accuracy']

                # store regression & text metrics for each model
                for m_idx, m in enumerate(models):
                    reg = metrics['regression'][m]
                    txt = metrics['text'][m] if 'text' in metrics else {}
                    mae [i, j, m_idx, t] = reg['MAE']
                    mse [i, j, m_idx, t] = reg['MSE']
                    bleu[i, j, m_idx, t] = txt['BLEU'] if 'text' in metrics else 0.0
                    chrf[i, j, m_idx, t] = txt['chrF'] if 'text' in metrics else 0.0

            print(f"Finished K={K}, sup={sup_per}")

    os.makedirs(f'results/{experiment_key}', exist_ok=True)
    # ── save everything ──
    np.save(f"results/{experiment_key}/ami_x.npy", ami_x)
    np.save(f"results/{experiment_key}/ami_y.npy", ami_y)
    np.save(f"results/{experiment_key}/accuracy.npy", accuracy)
    np.save(f"results/{experiment_key}/mae.npy", mae)
    np.save(f"results/{experiment_key}/mse.npy", mse)
    np.save(f"results/{experiment_key}/bleu.npy", bleu)
    np.save(f"results/{experiment_key}/chrf.npy", chrf)

    print("All done! Results are in the `results/` folder.")

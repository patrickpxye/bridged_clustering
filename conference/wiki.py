### Script 1: For Wikimedia dataset

import itertools
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.optimize import linear_sum_assignment
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sacrebleu.metrics import CHRF
from k_means_constrained import KMeansConstrained
from tqdm import tqdm
from scipy.stats import gaussian_kde
import os
from collections import Counter
from bioscan_copy import (
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
# 'x' - image embedding (numpy array)
# 'y' - text description (string)
# 'yv' - text embedding (numpy array)
# 'z' - page description (string, for generating clusters only)
# 'zv' - page embedding (numpy array, for generating clusters only)


df = pd.read_csv("wiki_df.csv")
df['x'] = df['x'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()  # Convert string back to numpy array
df['yv'] = df['yv'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()  # Convert string back to numpy array
df['zv'] = df['zv'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()  # Convert string back to numpy array

X = np.vstack(df['zv'].values)  # shape: (n_samples, embedding_dim)
db = DBSCAN(eps=0.35, min_samples=5, metric='cosine').fit(X)
labels = db.labels_
df = df.assign(cluster=labels)
df_valid = df[df['cluster'] != -1].copy()
cluster_sizes = df_valid['cluster'].value_counts()
eligible = cluster_sizes[cluster_sizes >= 12].index

pruned_clusters = []
for cl in eligible:
    sub = df_valid[df_valid['cluster'] == cl]
    # build list of word lists: split by ',' then by whitespace
    word_lists = sub['z'].str.split(',').apply(
        lambda shards: [w for shard in shards for w in shard.strip().split()]
    )
    # count all words across the cluster
    word_counts = Counter(w for words in word_lists for w in words)
    most_common_word, count = word_counts.most_common(1)[0]
    print(f"Most common word in cluster {cl}: {most_common_word} (count: {count})")
    
    # keep only those rows whose z contains this word
    mask = word_lists.apply(lambda words: most_common_word in words)
    pruned = sub[mask]
    if len(pruned) >= 12:
        pruned_clusters.append(pruned)

df_pruned = pd.concat(pruned_clusters, ignore_index=True)


def get_data(df, supervised_ratio, output_only_ratio, K=3, seed=None):


    """
    Split df into supervised, inference, and output-only subsets.
    - Supervised points: equally sampled (w/o replacement) from K contiguous chunks.
    - Inference and output-only: sampled at random from the remainder.
    Ensures no overlap and that supervised_ratio + inference_ratio + output_only_ratio == 1.0.
    """
    N = len(df)
    inference_ratio = 1.0 - supervised_ratio - output_only_ratio
    rng = np.random.default_rng(seed)

    # total supervised count
    n_sup = int(supervised_ratio * N)
    if n_sup < K:
        raise ValueError(f"Need at least K={K} supervised points, but got n_sup={n_sup}")

    # how many per chunk (distribute the remainder across the first `extra` chunks)
    base = n_sup // K
    extra = n_sup % K

    # build supervised index list
    supervised_idx = []
    # compute chunk boundaries
    chunk_sizes = [N // K + (1 if i < (N % K) else 0) for i in range(K)]
    starts = np.cumsum([0] + chunk_sizes[:-1])
    ends   = np.cumsum(chunk_sizes)

    for i, (start, end) in enumerate(zip(starts, ends)):
        chunk_inds = np.arange(start, end)
        m = base + (1 if i < extra else 0)
        chosen = rng.choice(chunk_inds, size=m, replace=False)
        supervised_idx.append(chosen)
    supervised_idx = np.concatenate(supervised_idx)

    # remaining indices for inference + output-only
    all_idx = np.arange(N)
    remaining = np.setdiff1d(all_idx, supervised_idx, assume_unique=True)

    # inference
    n_inf = int(inference_ratio * N)
    inf_idx = rng.choice(remaining, size=n_inf, replace=False)

    # output-only = whatever is left
    out_idx = np.setdiff1d(remaining, inf_idx, assume_unique=True)

    # build DataFrames
    supervised_df  = df.iloc[supervised_idx].reset_index(drop=True)
    inference_df   = df.iloc[inf_idx].reset_index(drop=True)
    output_only_df = df.iloc[out_idx].reset_index(drop=True)

    # for clustering inputs
    X_for_clustering = pd.concat([inference_df, supervised_df], ignore_index=True)
    Y_for_clustering = pd.concat([output_only_df, supervised_df], ignore_index=True)

    # summary
    # print(f"Total samples:         {N}")
    # print(f"  Supervised set:      {len(supervised_df)}")
    # print(f"  Inference set:       {len(inference_df)}")
    # print(f"  Output-only set:     {len(output_only_df)}")

    return supervised_df, inference_df, output_only_df, X_for_clustering, Y_for_clustering

def perform_clustering(X_for_clustering, Y_for_clustering, K):
    """
    Perform size‐constrained KMeans clustering on image and gene samples.
    Returns the cluster assignments for X and Y.
    """
    # --- X clustering ---
    X_matrix = np.vstack(X_for_clustering["x"].values)
    n_samples_x = X_matrix.shape[0]
    # enforce roughly equal cluster sizes
    size_min_x = n_samples_x // K
    size_max_x = int(np.ceil(n_samples_x / K))
    x_kmc = KMeansConstrained(
        n_clusters=K,
        size_min=size_min_x,
        size_max=size_max_x,
        random_state=42
    ).fit(X_matrix)
    x_clusters = x_kmc.labels_

    # --- Y clustering ---
    Y_matrix = np.vstack(Y_for_clustering["yv"].values)
    n_samples_y = Y_matrix.shape[0]
    size_min_y = n_samples_y // K
    size_max_y = int(np.ceil(n_samples_y / K))
    y_kmc = KMeansConstrained(
        n_clusters=K,
        size_min=size_min_y,
        size_max=size_max_y,
        random_state=42
    ).fit(Y_matrix)
    y_clusters = y_kmc.labels_

    return x_clusters, y_clusters

def clustering_quality_metrics(X_for_clustering, x_clusters, Y_for_clustering, y_clusters):
    """
    Compute Silhouette Coefficient and Davies–Bouldin Index
    for both input-space and output-space cluster assignments.
    """
    # Prepare feature matrices
    X_matrix = np.vstack(X_for_clustering["x"].values)
    Y_matrix = np.vstack(Y_for_clustering["yv"].values)
    
    # Input-space clustering metrics
    sil_x = silhouette_score(X_matrix, x_clusters)
    db_x  = davies_bouldin_score(X_matrix, x_clusters)
    
    # Output-space clustering metrics
    sil_y = silhouette_score(Y_matrix, y_clusters)
    db_y  = davies_bouldin_score(Y_matrix, y_clusters)
    
    return {
        "input_silhouette": sil_x,
        "input_davies_bouldin": db_x,
        "output_silhouette": sil_y,
        "output_davies_bouldin": db_y
    }

def decisionVector(sample, x_column, y_column, dim=5):

    # Check if the specified columns exist in the DataFrame
    if x_column not in sample.columns:
        raise KeyError(f"Column '{x_column}' not found in the DataFrame.")
    if y_column not in sample.columns:
        raise KeyError(f"Column '{y_column}' not found in the DataFrame.")

    # Create association matrix
    association_matrix = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            association_matrix[i, j] = np.sum((sample[x_column] == i) & (sample[y_column] == j))
    
    # Initialize decision array (this could be improved based on specific logic for decision making)
    decision = np.zeros(dim, dtype=int)
    
    # Logic to compute the decision vector based on association_matrix (you can modify this logic)
    # For now, just assigning maximum values
    for i in range(dim):
        decision[i] = np.argmax(association_matrix[i, :])  # You can customize this

    return decision

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
    preds, actuals = baseline_fn(sup, inf)

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

def run_experiment(df, supervised_ratio=0.05, output_only_ratio=0.5,
                   K=100, knn_neighbors=10, seed=None):

    sup_df, inf_df, out_df, Xc, Yc = get_data(df, supervised_ratio, output_only_ratio, K, seed)
    x_cl, y_cl = perform_clustering(Xc, Yc, K)

    ami_x = adjusted_mutual_info_score(Xc['cluster'], x_cl)
    ami_y = adjusted_mutual_info_score(Yc['cluster'], y_cl)
    decision = build_decision_matrix(sup_df, x_cl, y_cl, K)
    true_decision = build_true_decision_vector(Xc, Yc, x_cl, y_cl, K)
    accuracy = np.mean(decision == true_decision)

    cents, texts = compute_y_centroids(Yc, y_cl, K)
    inf_res = perform_inference(inf_df, x_cl, decision, cents, texts)

    # BKM baseline metrics
    bkm_pred_emb = np.vstack(inf_res['predicted_yv'].values)
    bkm_act_emb = np.vstack(inf_res['yv'].values)
    bkm_mae, bkm_mse = evaluate_regression_loss(bkm_pred_emb, bkm_act_emb)
    text_metrics_bkm = evaluate_text_metrics(inf_res['y'], inf_res['predicted_text'])
    # KNN baseline metrics
    knn_pred_emb, knn_act_emb, knn_pred_texts, knn_act_texts = knn_regression(sup_df, inf_df, knn_neighbors)
    knn_mae, knn_mse = evaluate_regression_loss(knn_pred_emb, knn_act_emb)
    text_metrics_knn = evaluate_text_metrics(knn_act_texts, knn_pred_texts)
    # Mean Teacher baseline metrics
    mt_pred, mt_act, mt_text_act, mt_text_pred = _wrap_text_baseline(mean_teacher_regression, sup_df, inf_df)
    mt_mae, mt_mse = evaluate_regression_loss(mt_pred, mt_act)
    mt_text_metrics = evaluate_text_metrics(mt_text_act, mt_text_pred)
    # FixMatch baseline metrics
    fm_pred, fm_act, fm_text_act, fm_text_pred = _wrap_text_baseline(fixmatch_regression, sup_df, inf_df)
    fm_mae, fm_mse = evaluate_regression_loss(fm_pred, fm_act)
    fm_text_metrics = evaluate_text_metrics(fm_text_act, fm_text_pred)
    # LapRLS baseline metrics
    laprls_pred, laprls_act, laprls_text_act, laprls_text_pred = _wrap_text_baseline(laprls_regression, sup_df, inf_df)
    laprls_mae, laprls_mse = evaluate_regression_loss(laprls_pred, laprls_act)
    laprls_text_metrics = evaluate_text_metrics(laprls_text_act, laprls_text_pred)
    # TSVR baseline metrics
    tsvr_pred, tsvr_act, tsvr_text_act, tsvr_text_pred = _wrap_text_baseline(tsvr_regression, sup_df, inf_df)
    tsvr_mae, tsvr_mse = evaluate_regression_loss(tsvr_pred, tsvr_act)
    tsvr_text_metrics = evaluate_text_metrics(tsvr_text_act, tsvr_text_pred)
    # TNNR baseline metrics
    tnnr_pred, tnnr_act, tnnr_text_act, tnnr_text_pred = _wrap_text_baseline(tnnr_regression, sup_df, inf_df)
    tnnr_mae, tnnr_mse = evaluate_regression_loss(tnnr_pred, tnnr_act)
    tnnr_text_metrics = evaluate_text_metrics(tnnr_text_act, tnnr_text_pred)
    # UCVME baseline metrics
    ucvme_pred, ucvme_act, ucvme_text_act, ucvme_text_pred = _wrap_text_baseline(ucvme_regression, sup_df, inf_df)
    ucvme_mae, ucvme_mse = evaluate_regression_loss(ucvme_pred, ucvme_act)
    ucvme_text_metrics = evaluate_text_metrics(ucvme_text_act, ucvme_text_pred)
    # RankUp baseline metrics
    rankup_pred, rankup_act, rankup_text_act, rankup_text_pred = _wrap_text_baseline(rankup_regression, sup_df, inf_df)
    rankup_mae, rankup_mse = evaluate_regression_loss(rankup_pred, rankup_act)
    rankup_text_metrics = evaluate_text_metrics(rankup_text_act, rankup_text_pred)
    # GCN baseline metrics
    gcn_pred, gcn_act, gcn_text_act, gcn_text_pred = _wrap_text_baseline(gcn_regression, sup_df, inf_df)
    gcn_mae, gcn_mse = evaluate_regression_loss(gcn_pred, gcn_act)
    gcn_text_metrics = evaluate_text_metrics(gcn_text_act, gcn_text_pred)

    metrics = {
        'clustering': {"AMI_X": ami_x, "AMI_Y": ami_y, "Bridging Accuracy": accuracy},
        'regression': {'BKM': {'MAE': bkm_mae, 'MSE': bkm_mse}, 'KNN': {'MAE': knn_mae, 'MSE': knn_mse}, 'MeanTeacher': {'MAE': mt_mae, 'MSE': mt_mse},
                      'FixMatch': {'MAE': fm_mae, 'MSE': fm_mse}, 'LapRLS': {'MAE': laprls_mae, 'MSE': laprls_mse},
                      'TSVR': {'MAE': tsvr_mae, 'MSE': tsvr_mse}, 'TNNR': {'MAE': tnnr_mae, 'MSE': tnnr_mse},
                      'UCVME': {'MAE': ucvme_mae, 'MSE': ucvme_mse}, 'RankUp': {'MAE': rankup_mae, 'MSE': rankup_mse},
                      'GCN': {'MAE': gcn_mae, 'MSE': gcn_mse}},
        'text': {'BKM': text_metrics_bkm, 'KNN': text_metrics_knn, 'MeanTeacher': mt_text_metrics,
                 'FixMatch': fm_text_metrics, 'LapRLS': laprls_text_metrics, 'TSVR': tsvr_text_metrics,
                 'TNNR': tnnr_text_metrics, 'UCVME': ucvme_text_metrics, 'RankUp': rankup_text_metrics,
                 'GCN': gcn_text_metrics},
    }
    return metrics



if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd

    # Experiment grid
    K_values   = [3, 4, 5]
    sup_values = [1, 2, 3, 4]            # sup_per_cluster
    out_only   = 0.5
    cluster_sz = 12
    seeds      = list(range(100))

    pruned_counts    = df_pruned['cluster'].value_counts()
    eligible_pruned  = pruned_counts[pruned_counts >= cluster_sz].index

    # Which models to record (must match keys in run_experiment’s metrics)
    models = [
      'BKM', 'KNN',
      'MeanTeacher', 'GCN', 'FixMatch',
      'LapRLS', 'TSVR', 'TNNR', 'UCVME',
      'RankUp'
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
                # ── sample K clusters from pruned set ──
                chosen = np.random.choice(eligible_pruned, size=K, replace=False)
                sample = pd.concat([
                    df_pruned[df_pruned['cluster']==c]
                              .sample(cluster_sz, random_state=s)
                    for c in chosen
                ]).reset_index(drop=True)

                # ── 2) run holistic experiment ──
                metrics = run_experiment(
                    sample,
                    supervised_ratio   = sup_per/cluster_sz,
                    output_only_ratio  = out_only,
                    K                   = K,
                    knn_neighbors      = sup_per,
                    seed                = s
                )

                # store clustering AMI
                ami_x[i, j, t] = metrics['clustering']['AMI_X']
                ami_y[i, j, t] = metrics['clustering']['AMI_Y']
                accuracy[i, j, t] = metrics['clustering']['Bridging Accuracy']

                # store regression & text metrics for each model
                for m_idx, m in enumerate(models):
                    reg = metrics['regression'][m]
                    txt = metrics['text'][m]
                    mae [i, j, m_idx, t] = reg['MAE']
                    mse [i, j, m_idx, t] = reg['MSE']
                    bleu[i, j, m_idx, t] = txt['BLEU']
                    chrf[i, j, m_idx, t] = txt['chrF']

            print(f"Finished K={K}, sup={sup_per}")

    # ── save everything ──
    np.save("results/ami_x.npy",    ami_x)
    np.save("results/ami_y.npy",    ami_y)
    np.save("results/accuracy.npy", accuracy)
    np.save("results/mae.npy",      mae)
    np.save("results/mse.npy",      mse)
    np.save("results/bleu.npy",     bleu)
    np.save("results/chrf.npy",     chrf)

    print("All done! Results are in the `results/` folder.")

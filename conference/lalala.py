
def kmm_image_to_text(supervised_df, inference_df, lam=1e-2, sigma=None, random_state=0):
    """
    Kernel Mean Matching baseline for image→text:
      - supervised_df has columns:
          'x'   : image embedding np.array
          'yv'  : text embedding np.array
          'y'   : text string
      - inference_df same schema, but 'yv' and 'y' are the ground-truth targets.
    Returns
    -------
    pred_emb : np.ndarray, shape (N_inf, d)
        predicted text embeddings
    true_emb : np.ndarray, shape (N_inf, d)
        ground-truth text embeddings
    """
    # prepare DataFrames for the original KMM function
    img_df  = supervised_df.rename(columns={'x':'morph_coordinates'})
    gene_df = supervised_df.rename(columns={'yv':'gene_coordinates'})
    supervised_df = supervised_df.rename(columns={
        'x':'morph_coordinates',
        'yv':'gene_coordinates',
    })
    inf_df2 = inference_df.rename(columns={
        'x':'morph_coordinates',
        'yv':'gene_coordinates'
    })
    # call your original
    Y_pred, Y_true = kernel_mean_matching_regression(
        image_df   = img_df,
        gene_df    = gene_df,
        supervised_df= supervised_df,
        inference_df = inf_df2,
    )
    return Y_pred, Y_true

def kmm_text_to_image(supervised_df, inference_df, lam=1e-2, sigma=None, random_state=0):
    """
    Reversed KMM baseline for text→image:
      - supervised_df has columns 'x','yv','y'
      - inference_df same schema
    Returns
    -------
    pred_emb : np.ndarray, shape (N_inf, d)
        predicted image embeddings
    true_emb : np.ndarray, shape (N_inf, d)
        ground-truth image embeddings
    """
    gene_df = supervised_df.rename(columns={'yv':'gene_coordinates'})
    img_df  = supervised_df.rename(columns={'x' :'morph_coordinates'})
    supervised_df = supervised_df.rename(columns={
        'yv':'gene_coordinates',
        'x' :'morph_coordinates'
    })
    inf_df2 = inference_df.rename(columns={
        'yv':'gene_coordinates',
        'x' :'morph_coordinates'
    })
    Y_pred, Y_true = reversed_kernel_mean_matching_regression(
        gene_df      = gene_df,
        image_df     = img_df,
        supervised_df= supervised_df,
        inference_df = inf_df2,
    )
    return Y_pred, Y_true
def em_image_to_text(
    supervised_df,
    inference_df,
    n_components=3,
    max_iter=100,
    tol=1e-4,
    eps=1e-8
):
    """
    EM baseline for image→text:
      - supervised_df has columns:
          'x'   : image embedding np.array
          'yv'  : text embedding np.array
          'y'   : text string
      - inference_df same schema, but 'yv' and 'y' are the ground-truth targets.
    Returns
    -------
    pred_emb : np.ndarray, shape (N_inf, d)
        predicted text embeddings
    true_emb : np.ndarray, shape (N_inf, d)
        ground-truth text embeddings
    """
    # prepare DataFrames for the EM function
    img_df  = supervised_df.rename(columns={'x': 'morph_coordinates'})
    gene_df = supervised_df.rename(columns={'yv': 'gene_coordinates'})
    sup_df  = supervised_df.rename(columns={
        'x': 'morph_coordinates',
        'yv': 'gene_coordinates'
    })
    inf_df2 = inference_df.rename(columns={
        'x': 'morph_coordinates',
        'yv': 'gene_coordinates'
    })

    # call the EM regression
    Y_pred, Y_true = em_regression(
        supervised_df = sup_df,
        image_df      = img_df,
        gene_df       = gene_df,
        inference_df  = inf_df2,
        n_components  = n_components,
        max_iter      = max_iter,
        tol           = tol,
        eps           = eps
    )
    return Y_pred, Y_true


def em_text_to_image(
    supervised_df,
    inference_df,
    n_components=3,
    max_iter=100,
    tol=1e-4,
    eps=1e-8
):
    """
    Reversed EM baseline for text→image:
      - supervised_df has columns:
          'x'   : image embedding np.array
          'yv'  : text embedding np.array
          'y'   : text string
      - inference_df same schema, but 'x' and 'y' are the ground-truth targets.
    Returns
    -------
    pred_emb : np.ndarray, shape (N_inf, d)
        predicted image embeddings
    true_emb : np.ndarray, shape (N_inf, d)
        ground-truth image embeddings
    """
    gene_df = supervised_df.rename(columns={'yv': 'gene_coordinates'})
    img_df  = supervised_df.rename(columns={'x' : 'morph_coordinates'})
    sup_df  = supervised_df.rename(columns={
        'x': 'morph_coordinates',
        'yv': 'gene_coordinates'
    })
    inf_df2 = inference_df.rename(columns={
        'x': 'morph_coordinates',
        'yv': 'gene_coordinates'
    })

    Y_pred, Y_true = reversed_em_regression(
        gene_df       = gene_df,
        image_df      = img_df,
        supervised_df = sup_df,
        inference_df  = inf_df2,
        n_components  = n_components,
        max_iter      = max_iter,
        tol           = tol,
        eps           = eps
    )
    return Y_pred, Y_true

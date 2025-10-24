import torch
from sklearn.mixture import GaussianMixture
import numpy as np

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
def signguard(groups_gradients, device = 'cpu', seed = 47):
    """
    Based on the description in https://arxiv.org/abs/2109.05872
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    seed: seed for randomness
    """


    param_list_indexed = [(i, params) for i, params in groups_gradients.items()]
    param_list = [params for i, params in param_list_indexed]
    index = [i for i, params in param_list_indexed]
    n = len(param_list)

    num_params = param_list[0].size(0)
    selection_fraction = 1.0 #original 0.1

    # lower and upper bound L,R for gradient norm
    L = 0.1 # original 0.1
    R = 3 # original 3.
    S1 = []
    S2 = []

    # compute l2-norm
    l2_norm = torch.stack([torch.norm(g.flatten(), p=2.0) for g in param_list])

    # compute element wise sign
    num_selection = int(num_params * selection_fraction)
    perm = torch.randperm(num_params)
    idx = perm[:num_selection]
    sign_grads = [torch.sign(g[idx]) for g in param_list]

    # norm-threshold filtering
    M = torch.median(l2_norm)
    for i in range(n):
        if L <= l2_norm[i] / M and l2_norm[i] / M <= R:
            S1.append(i)

    # compute sign statistics
    sign_pos = torch.stack([grad.eq(1.0).float().mean() for grad in sign_grads])
    sign_zero = torch.stack([grad.eq(0.0).float().mean() for grad in sign_grads])
    sign_neg = torch.stack([grad.eq(-1.0).float().mean() for grad in sign_grads])

    # sign-based clustering
    sign_feat = torch.stack([sign_pos, sign_zero, sign_neg], dim=1).detach().cpu().numpy()
    cluster = KMeans(n_clusters=2, max_iter=10, random_state=seed)
    labels = cluster.fit_predict(sign_feat)

    labels_tensor = torch.from_numpy(labels).to(device)
    count = torch.bincount(labels_tensor)
    largest_cluster = torch.argmax(count)

    for i, value in enumerate(labels_tensor):
        if value == largest_cluster:
            S2.append(i)

    # compute intersection of S1 and S2
    S = [i for i in S1 if i in S2]


    isAnomalous = {
        index[i]: (0 if i in S else 1)
        for i in range(n)
    }

    

    return isAnomalous


def signguard_anomaly_scores(groups_gradients: dict,
                             device = 'cpu',
                             seed: int = 42,
                             R: float = 3.0,
                             selection_fraction: float = 1,
                             alpha: float = 0.0):
    """
    Compute SignGuard-like anomaly scores for a dict of gradients.

    Args
    ----
    groups_gradients : dict[int, Tensor | Sequence[Tensor]]
        Mapping from group_id -> gradient. Each gradient can be:
          - a single 1-D/flat tensor, or
          - a sequence of tensors (e.g., per-layer), which will be flattened & concatenated.
    device : torch.device or str
    seed : int
    R : float
        Scale for norm-deviation clipping.
    selection_fraction : float in (0,1]
        Fraction of coordinates to sample for sign features.
    alpha : float in [0,1]
        Blend between norm deviation (alpha) and sign-GMM score (1-alpha).

    Returns
    -------
    dict[int, torch.Tensor] : group_id -> scalar tensor score on `device`
    """
    if not isinstance(groups_gradients, dict) or len(groups_gradients) == 0:
        return {}

    # --- fix RNG for coordinate subsampling
    torch.manual_seed(seed)

    group_ids = list(groups_gradients.keys())

    # --- flatten each group's gradient to a single vector on device
    
    param_list = []
    for gid in group_ids:
        g = groups_gradients[gid]
        if isinstance(g, (list, tuple)):
            flat = torch.cat([x.to(device).reshape(-1) for x in g])
        else:
            flat = g.to(device).reshape(-1)
        param_list.append(flat)

    n = len(param_list)


    # --- [2] L2 norms & median-based deviation
    l2_norm = torch.stack([p.norm(p=2) for p in param_list])  # (n,)
    M = l2_norm.median()
    norm_ratio = l2_norm / M
    score_norm = torch.clamp((norm_ratio - 1.0).abs() / R, 0.0, 1.0)  # (n,)

    # --- [3] coordinate subsampling for sign features
    num_params = param_list[0].numel()
    sf = float(selection_fraction)
    sf = max(1e-9, min(1.0, sf))
    num_sel = max(1, int(num_params * sf))
    idx = torch.randperm(num_params, device=device)[:num_sel]

    # --- [4] sign fractions
    sign_grads = torch.stack([torch.sign(p[idx]) for p in param_list])  # (n, num_sel)
    sign_pos  = (sign_grads ==  1).float().mean(dim=1)  # (n,)
    sign_zero = (sign_grads ==  0).float().mean(dim=1)  # (n,)
    sign_neg  = (sign_grads == -1).float().mean(dim=1)  # (n,)

    # --- [5] GMM on CPU to score "outlierness" via minority-prob
    feat = torch.stack([sign_pos, sign_zero, sign_neg], dim=1).cpu().numpy()  # (n,3)
    try:
        gmm = GaussianMixture(n_components=2, random_state=seed).fit(feat)
        probs = gmm.predict_proba(feat)  # (n,2)
        labels = gmm.predict(feat)
        counts = np.bincount(labels, minlength=2)
        major = int(np.argmax(counts))
        score_sign = 1.0 - torch.from_numpy(probs[:, major]).to(device)  # (n,)
    except Exception:
        # Fallback if GMM fails (e.g., degenerate features): use zeros
        raise Exception("GMM fitting failed")

    # --- [6] blend
    anomaly_scores = alpha * score_norm + (1 - alpha) * score_sign  # (n,)

    return {gid: anomaly_scores[i] for i, gid in enumerate(group_ids)}


def observation_function(gradients, bayesian_params):
    """
    A placeholder observation function that processes gradients.
    This function should be replaced with the actual logic for observing gradients.
    
    Args:
        gradients (Dict): Dictionary of grouped gradients, group_id: gradient 
    
    Returns:
        list: Processed scores based on the gradients.
    """

    method = bayesian_params.get("observation_method", "signguard")
    

    if method == "signguard":
        # Implement the signguard observation logic
        raise NotImplementedError(f"Observation method '{method}' is Redacted. Use binarySignguard instead.")
    elif method == "binarySignguard":
        return signguard(gradients)
    else:
        # Implement the other observation logic
        raise NotImplementedError(f"Observation method '{method}' is not implemented.")

def expectation_function(config):
    """
    A placeholder expectation function that computes expectations based on the configuration.
    
    Args:
        config (tuple): Configuration of faulty/not faulty nodes.
    
    Returns:
        float: Computed expectation for the given configuration.
    """
    # Example logic: return a dummy expectation value
    return sum(config)  # Replace with actual logic if needed

def likelihood_function(observed_score, expectation):
    """
    A placeholder likelihood function that computes the likelihood of an observed score given an expectation.
    
    Args:
        observed_score (float): The observed score from the observation function.
        expectation (float): The expectation computed from the expectation function.
    
    Returns:
        float: Computed likelihood.
    """
    # Example logic: return a dummy likelihood value
    return np.exp(-abs(observed_score - expectation))  # Replace with actual logic if needed
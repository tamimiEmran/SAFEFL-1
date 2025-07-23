import pandas as pd

# 1) load your data
df = pd.read_csv("final.csv")
# generate a mapping for client_id to is_user_malicious
# take only grouping_agg = "aggregate" 
df = df[df['grouping_agg'] == "average"]
# remove any scoring_func that starts wit "flame_"
df = df[~df['scoring_func'].str.startswith("flame_")]

def generate_malicious_mapping(df):
    """
    Generate a mapping from client_id to is_user_malicious.
    Assumes df has columns 'client_id' and 'is_user_malicious'.
    """
    return dict(zip(df['client_id'], df['is_user_malicious']))

id_to_malicious = generate_malicious_mapping(df)

# drop the is_user_malicious column
df = df.drop(columns=["is_user_malicious"])

# 2) define your grouping‐keys and aggregations
keys = ["round_id","group_id","grouping_agg","byz_type","scoring_func", "group_size"]

agg_dict = {
    "client_id":         list,      # becomes your users_in_group_at_round
    "n_total_clients":   "first",
    "n_total_mal":       "first",  # if you want to keep the first value of n_total_mal
    "n_total_groups":    "first",
    "group_id_size":     "first",  
    "group_id_mal_size":  "first",  # if you want to keep the first value of group_id_mal_size
    "score":            "mean"       # if you’d also like all the individual scores
}
# 3) do the groupby-agg
grouped = (
    df
    .groupby(keys, as_index=False)
    .agg(agg_dict)
)


import numpy as np
import pandas as pd

# --- Assumption Strategies ---
class AssumptionStrategy:
    """
    Base class for generating per-user probability assumptions from client IDs.
    """
    def apply(self, client_list: list) -> list[float]:
        raise NotImplementedError("Must implement apply method")

class MappingAssumption(AssumptionStrategy):
    """
    Assign probabilities based on a user-provided mapping dict.
    """
    def __init__(self, mapping: dict, default: float = 0.0):
        self.mapping = mapping
        self.default = default
    def apply(self, client_list: list) -> list[float]:
        return [self.mapping.get(cid, self.default) for cid in client_list]

class UniformAssumption(AssumptionStrategy):
    """
    Assign every user the same fixed probability.
    """
    def __init__(self, value: float = 0.5):
        self.value = value
    def apply(self, client_list: list) -> list[float]:
        return [self.value] * len(client_list)

class NoisyAssumption(AssumptionStrategy):
    """
    Add Gaussian noise around a base assumption strategy.
    """
    def __init__(self, base: AssumptionStrategy, sigma: float = 0.1):
        self.base = base
        self.sigma = sigma
    def apply(self, client_list: list) -> list[float]:
        base_p = np.array(self.base.apply(client_list))
        noisy = base_p + np.random.normal(scale=self.sigma, size=base_p.shape)
        return list(np.clip(noisy, 0.0, 1.0))

# --- Expectation Strategies ---

class ExpectationStrategy:
    """
    Base class for summarizing a list of probabilities into a group anomaly score.
    """
    def apply(self, p_list: list[float]) -> float:
        raise NotImplementedError("Must implement apply method")

class ExpectedCountExpectation(ExpectationStrategy):
    """
    Original expectation (expected number of malicious users):
    \[
    \mathbb{E}[f(\text{Group})]
    = \sum_{i \in \text{Group}} P(M_i = 1)
    \]
    """
    def apply(self, p_list: list[float]) -> float:
        return float(np.sum(p_list))

class NoMaliciousProbabilityExpectation(ExpectationStrategy):
    """
    Probability the group has no malicious users:
    \[
    P(\text{no malicious})
    = \prod_{i \in \text{Group}} \bigl(1 - P(M_i = 1)\bigr)
    \]
    """
    def apply(self, p_list: list[float]) -> float:
        return float(np.prod([1.0 - p for p in p_list]))

class GroupMaliciousProbabilityExpectation(ExpectationStrategy):
    """
    Revised expectation (probability group is malicious):
    \[
    P(\text{Group is malicious})
    = 1 - P(\text{no malicious})
    = 1 - \prod_{i \in \text{Group}} \bigl(1 - P(M_i = 1)\bigr)
    \]
    """
    def apply(self, p_list: list[float]) -> float:
        return 1.0 - float(np.prod([1.0 - p for p in p_list]))

class MeanExpectation(ExpectationStrategy):
    """
    Mean of all probabilities in the group.
    """
    def apply(self, p_list: list[float]) -> float:
        return float(np.mean(p_list)) if p_list else 0.0

class MaxExpectation(ExpectationStrategy):
    """
    Maximum probability in the group.
    """
    def apply(self, p_list: list[float]) -> float:
        return float(np.max(p_list)) if p_list else 0.0


# --- Experiment Runner ---
class ExperimentRunner:
    """
    Runs all combinations of assumption and expectation strategies on grouped client data,
    preserving group-level metadata columns.

    Accepts DataFrame in one of three forms:
      1) Pre-aggregated, with 'client_list' column and metadata
      2) Pre-aggregated, with 'client_id' column containing lists and metadata
      3) Raw, with individual 'client_id' rows plus metadata

    Metadata columns required:
      'grouping_agg','byz_type','scoring_func',
      'group_size','group_id_size','group_id_mal_size','score'

    Returns a DataFrame with columns:
      ['assumption','expectation','round_id','group_id',
       metadata..., 'expected_score']
    """

    METADATA_COLS = [
        'grouping_agg','byz_type','scoring_func',
        'group_size','group_id_size','group_id_mal_size','score'
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        assumptions: dict[str, AssumptionStrategy],
        expectations: dict[str, ExpectationStrategy]
    ):
        self.assumptions = assumptions
        self.expectations = expectations
        cols = set(df.columns)
        base_req = {'round_id','group_id'} | set(self.METADATA_COLS)

        # Case 1: exact 'client_list' column
        if 'client_list' in cols:
            missing = base_req.union({'client_list'}) - cols
            if missing:
                raise ValueError(f"Missing columns for pre-agg stage: {missing}")
            df_proc = df.copy()
            df_proc['client_list'] = df_proc['client_list']

        # Case 2: 'client_id' contains lists of IDs
        elif 'client_id' in cols and isinstance(df['client_id'].iat[0], list):
            missing = base_req.union({'client_id'}) - cols
            if missing:
                raise ValueError(f"Missing columns for pre-agg with list-col: {missing}")
            df_proc = df.copy()
            df_proc['client_list'] = df_proc['client_id']

        # Case 3: raw rows needing grouping
        else:
            if not ({'client_id'} | base_req).issubset(cols):
                raise ValueError(f"Raw data must contain columns: {base_req.union({'client_id'})}")
            # group into lists
            df_proc = (
                df
                .groupby(['round_id','group_id'] + self.METADATA_COLS)
                ['client_id']
                .apply(list)
                .reset_index(name='client_list')
            )

        self.df = df_proc

    def run(self) -> pd.DataFrame:
        records = []
        for a_name, a_strat in self.assumptions.items():
            for e_name, e_strat in self.expectations.items():
                for _, row in self.df.iterrows():
                    client_list = row['client_list']
                    # ensure flat list of IDs
                    if any(isinstance(x, list) for x in client_list):
                        # flatten one level
                        client_list = [cid for sub in client_list for cid in sub]
                    p_list = a_strat.apply(client_list)
                    expected = e_strat.apply(p_list)
                    rec = {
                        'assumption': a_name,
                        'expectation': e_name,
                        'round_id': row['round_id'],
                        'group_id': row['group_id'],
                        'expected_score': expected
                    }
                    # include metadata
                    for col in self.METADATA_COLS:
                        rec[col] = row[col]
                    records.append(rec)

        return pd.DataFrame(records)

# --- Scenario Generators ---
def generate_boolean_assumption_scenarios(
    id_to_malicious: dict[int, bool],
    uniform_value: float = 0.5
) -> dict[str, AssumptionStrategy]:
    """
    Given a mapping of client_id -> is_malicious (bool),
    generate three scenarios:
      - uniform_<value>: everyone gets uniform_value
      - very_wrong: malicious -> low (1-uniform), normal -> high
      - very_correct: malicious -> high, normal -> low
    """
    mapping_wrong = {}
    mapping_correct = {}
    for cid, mal in id_to_malicious.items():
        if mal:
            mapping_wrong[cid] = 1.0 - uniform_value
            mapping_correct[cid] = uniform_value
        else:
            mapping_wrong[cid] = uniform_value
            mapping_correct[cid] = 1.0 - uniform_value

    inv_value = round(1 - uniform_value, 1)  # Round to avoid floating point errors
    return {
        f'uniform_{uniform_value}': UniformAssumption(uniform_value),
        f'very_wrong_{uniform_value}': MappingAssumption(mapping_wrong, default=uniform_value),
        f'very_correct_{inv_value}': MappingAssumption(mapping_correct, default=uniform_value)
    }

# --- Example usage ---
# df has one row per group, and df['client_id'] may already be a list of IDs
mutiple_assumptions = {}
for value in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    mutiple_assumptions.update(generate_boolean_assumption_scenarios(id_to_malicious, uniform_value=value))

expectations = {'mean': MeanExpectation(), 'max': MaxExpectation(), "expected_count": ExpectedCountExpectation(),
               "no_malicious_prob": GroupMaliciousProbabilityExpectation()}
runner = ExperimentRunner(df, mutiple_assumptions, expectations)
results = runner.run()
print(results)

# drop byz_type
results = results.drop(columns=["byz_type"], errors='ignore')
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# --- Mutual Information Estimators ---

def histogram_mi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Estimate MI by discretizing (histogram-based)."""
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    nzs = p_xy > 0
    denom = p_x[:, None] * p_y[None, :]
    return float(np.sum(p_xy[nzs] * np.log(p_xy[nzs] / denom[nzs])))

def entropy_from_counts(counts: np.ndarray) -> float:
    """Compute entropy H(P) for a probability vector P derived from counts."""
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))

def normalized_mi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Normalized MI = I(X;Y) / sqrt(H(X)*H(Y))."""
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / c_xy.sum()
    h_x = entropy_from_counts(p_xy.sum(axis=1))
    h_y = entropy_from_counts(p_xy.sum(axis=0))
    mi = histogram_mi(x, y, bins=bins)
    return mi / np.sqrt(h_x * h_y) if (h_x > 0 and h_y > 0) else 0.0

def knn_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """k-NN based MI estimator using scikit-learn."""
    return float(mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False, n_neighbors=k)[0])

# --- Compute MI for subsets ---

def compute_mutual_info_subsets(
    df: pd.DataFrame,
    bins: int = 100,
    k: int = 10
) -> pd.DataFrame:
    """
    Compute three MI measures between 'expected_score' and 'score'
    for each subset defined by ['scoring_func','grouping_agg','expectation','assumption'].
    
    Returns a DataFrame with columns:
      ['scoring_func','grouping_agg','expectation','assumption',
       'histogram_mi','normalized_mi','knn_mi']
    """
    grouping_cols = ['scoring_func', 'grouping_agg', 'expectation', 'assumption']
    records = []
    for keys, grp in df.groupby(grouping_cols):
        x = grp['expected_score'].values
        y = grp['score'].values
        records.append({
            'scoring_func': keys[0],
            'grouping_agg': keys[1],
            'expectation': keys[2],
            'assumption': keys[3],
            'histogram_mi': histogram_mi(x, y, bins=bins),
            'normalized_mi': normalized_mi(x, y, bins=bins),
            'knn_mi': knn_mi(x, y, k=k)
        })
    return pd.DataFrame(records)
#%%
# To use:
mi_results = compute_mutual_info_subsets(results, bins=1000, k=200)

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_faceted_heatmaps(
    mi_df, 
    metric='histogram_mi', 
    max_cols=5, 
    aggfunc='mean',
    figsize_multiplier=5,  # Increased figure size multiplier
    dpi=400  # Added higher DPI for better resolution
):
    """
    Create faceted heatmaps of assumption × expectation MI values for each scoring_func,
    aggregating across grouping_agg by the specified aggfunc (e.g. 'mean' or 'median').
    
    Args:
        mi_df (DataFrame): must include ['scoring_func','assumption','expectation',metric].
        metric (str): one of ['histogram_mi','normalized_mi','knn_mi'].
        max_cols (int): max heatmaps per row.
        aggfunc (str): aggregation function for duplicates, e.g. 'mean' or 'median'.
        figsize_multiplier (int): multiplier for the figure size.
        dpi (int): dots per inch for the figure resolution.
    """
    scoring_funcs = sorted(mi_df['scoring_func'].unique())
    assumptions = sorted(mi_df['assumption'].unique())
    expectations = sorted(mi_df['expectation'].unique())
    
    n = len(scoring_funcs)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, 
                            figsize=(ncols * figsize_multiplier, nrows * figsize_multiplier), 
                            squeeze=False,
                            dpi=dpi)
    
    for idx, sf in enumerate(scoring_funcs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        
        sub = mi_df[mi_df['scoring_func'] == sf]
        # aggregate across grouping_agg (and any other remaining dims)
        agg = sub.groupby(['assumption','expectation'])[metric].agg(aggfunc)
        pivot = agg.unstack(level='expectation').reindex(index=assumptions, columns=expectations)
        
        im = ax.imshow(pivot, aspect='auto')
        ax.set_xticks(range(len(expectations)))
        ax.set_xticklabels(expectations, rotation=90, fontsize=10)
        ax.set_yticks(range(len(assumptions)))
        ax.set_yticklabels(assumptions, fontsize=10)
        ax.set_title(sf, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[idx])
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Filter out 'max' and 'mean' from expectations
filtered_results = mi_results[~mi_results['expectation'].isin(['max', 'mean'])]
plot_faceted_heatmaps(filtered_results, metric='normalized_mi', max_cols=4, aggfunc='mean')
# %%

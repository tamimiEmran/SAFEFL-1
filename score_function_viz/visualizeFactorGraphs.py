"""
sample of observation_scores.csv:
round_id,group_id,numberOfMal,score,avgMalScore,avgNormScore,minMalScore,maxNormScore,idxOfMaxNormScore,idxOfMinNormScore,dataset,attack_type,n_byzantine,bias,n_workers
0,0,0,0.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,1,1,1.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,2,0,0.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,3,0,0.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,4,1,1.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,5,2,1.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,6,0,0.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
0,7,1,1.0,0.5021818,0.4982222,0.5,0.50909084,11,0,MNIST,scaling_attack,50,0.5,500
"""

import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

import matplotlib.pyplot as plt

def plot_scores(csv_path = r'observation_scores.csv', save_path=None):
    # read CSV
    df = pd.read_csv(csv_path)
    # group by round_id (in case there are multiple rows per round)
    grouped = df.groupby('round_id', as_index=False)[['avgMalScore', 'avgNormScore']].mean()

    # plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(grouped['round_id'], grouped['avgMalScore'], color='red', label='avgMalScore')
    plt.plot(grouped['round_id'], grouped['avgNormScore'], color='green', label='avgNormScore')

    plt.xlabel('round')
    plt.ylabel('score')
    plt.title('Average Malicious vs Normal Scores by Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

# example usage
plot_scores(r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\observation_scores.csv')
# or to save:
# plot_scores('observation_scores.csv', save_path='scores_by_round.png')

def plot_min_max_by_idx(csv_path=r'observation_scores.csv', save_path=None):
    df2 = pd.read_csv(csv_path)

    # helper to select a representative idx per round (mode or first)
    def mode_or_first(s):
        m = s.mode()
        return int(m.iat[0]) if not m.empty else int(s.iat[0])

    # pick score column names present in the file
    max_col = 'maxNormScore' if 'maxNormScore' in df2.columns else ('max_score' if 'max_score' in df2.columns else None)
    min_col = 'minMalScore' if 'minMalScore' in df2.columns else ('min_score' if 'min_score' in df2.columns else None)
    if max_col is None or min_col is None:
        raise RuntimeError("Expected score columns 'maxNormScore' and 'minMalScore' (or fallback names) in CSV")

    # aggregate per round: mean for scores, mode (or first) for idx columns
    grouped2 = df2.groupby('round_id', as_index=False).agg({
        max_col: 'mean',
        min_col: 'mean',
        'idxOfMaxNormScore': mode_or_first,
        'idxOfMinNormScore': mode_or_first,
    })

    rounds = grouped2['round_id']
    idx_max = grouped2['idxOfMaxNormScore'].astype(int)
    idx_min = grouped2['idxOfMinNormScore'].astype(int)
    scores_max = grouped2[max_col]
    scores_min = grouped2[min_col]

    # unique idx set and color mapping (one unique color per idx)
    unique_idxs = sorted(set(idx_max.tolist() + idx_min.tolist()))
    cmap = plt.get_cmap('tab20', max(1, len(unique_idxs)))
    color_map = {idx: cmap(i % cmap.N) for i, idx in enumerate(unique_idxs)}

    plt.figure(figsize=(10, 5))
    plt.scatter(rounds, scores_max, c=[color_map[i] for i in idx_max], marker='o', s=40, label='max user (idx)')
    plt.scatter(rounds, scores_min, c=[color_map[i] for i in idx_min], marker='x', s=40, label='min user (idx)')

    # create a legend showing that X is for minimal user and O is for max user
    legend_elements = [
        Line2D([0], [0], marker='o', label='max norm user (idx)', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='x', label='min mal user (idx)', markerfacecolor='gray', markersize=8)
    ]
    plt.legend(handles=legend_elements)
    plt.xlabel('round')
    plt.ylabel('score')
    plt.title('Min and Max User Scores colored by user idx')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

# example usage:
plot_min_max_by_idx(r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\observation_scores.csv')


def plot_binary_confusion(csv_path=r'observation_scores.csv', normalize=False, save_path=None):
    """
    Plot a confusion matrix for a binary classification where the true label is:
      positive (1) if numberOfMal > 0, else negative (0).
    The predicted label is taken directly from a binary score column (0.0/1.0).
    """
    df = pd.read_csv(csv_path)

    if 'numberOfMal' not in df.columns:
        raise RuntimeError("CSV must contain a 'numberOfMal' column to derive true labels")

    # true labels: 1 if any malicious present, else 0
    y_true_full = (df['numberOfMal'] > 0).astype(int).to_numpy()

    chosen = 'score'

    scores_full = df[chosen].astype(float).to_numpy()

    # drop rows with NaN scores so confusion is computed on valid pairs only
    valid_mask = ~np.isnan(scores_full)
    if not valid_mask.any():
        raise RuntimeError("No valid (non-NaN) scores found in chosen column")
    y_true = y_true_full[valid_mask]
    scores = scores_full[valid_mask]

    # ensure scores are binary floats (0.0/1.0) and convert to int labels
    unique_vals = np.unique(scores)
    rounded = np.round(unique_vals).astype(int)
    if set(rounded).issubset({0, 1}) and np.all(np.isin(unique_vals, rounded)):
        y_pred = np.round(scores).astype(int)
    else:
        raise RuntimeError(f"Chosen score column '{chosen}' does not contain binary 0/1 values (found: {unique_vals}).")

    # compute confusion matrix
    labels = [0, 1]
    if normalize:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        values_format = '.2f'
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        values_format = 'd'

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, values_format=values_format, colorbar=False)
    ax.set_title(f"Confusion matrix ({chosen} -> pred)")
    plt.tight_layout()

    # also print simple classification report to stdout
    try:
        print(classification_report(y_true, y_pred, target_names=['negative', 'positive']))
    except Exception:
        pass

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

# example usages:
plot_binary_confusion(r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\observation_scores.csv')
plot_binary_confusion(r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\observation_scores.csv', normalize=True)


def plot_tpr_tnr_by_num_mal(csv_path=r'observation_scores.csv', save_path=None):
    """
    Compute and plot TPR, TNR, FPR and FNR aggregated by the 'numberOfMal' value.
    - y_true = (numberOfMal > 0)
    - y_pred = rounded 'score' column (expects values that round to 0 or 1)
    Produces a grouped bar chart with one group per distinct numberOfMal.
    """
    df = pd.read_csv(csv_path)

    if 'numberOfMal' not in df.columns:
        raise RuntimeError("CSV must contain a 'numberOfMal' column")
    if 'score' not in df.columns:
        raise RuntimeError("CSV must contain a 'score' column to derive predictions")

    # drop rows with NaN in score or numberOfMal
    df = df.dropna(subset=['score', 'numberOfMal']).copy()
    if df.empty:
        raise RuntimeError("No valid rows after dropping NaNs in 'score'/'numberOfMal'")

    y_true = (df['numberOfMal'] > 0).astype(int).to_numpy()
    scores = df['score'].astype(float).to_numpy()
    # ensure binary after rounding
    rounded_unique = np.unique(np.round(scores))
    if not set(np.round(rounded_unique).astype(int)).issubset({0, 1}):
        raise RuntimeError(f"'score' column does not round to binary values (found: {rounded_unique})")
    y_pred = np.round(scores).astype(int)

    df['_y_true'] = y_true
    df['_y_pred'] = y_pred

    groups = df.groupby('numberOfMal', sort=True)
    num_mal_vals = []
    tprs = []
    tnrs = []
    fprs = []
    fnrs = []
    counts = []

    for num_mal, g in groups:
        tp = int(((g['_y_true'] == 1) & (g['_y_pred'] == 1)).sum())
        fn = int(((g['_y_true'] == 1) & (g['_y_pred'] == 0)).sum())
        tn = int(((g['_y_true'] == 0) & (g['_y_pred'] == 0)).sum())
        fp = int(((g['_y_true'] == 0) & (g['_y_pred'] == 1)).sum())

        # TPR = TP / (TP + FN) ; TNR = TN / (TN + FP)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        # FPR = FP / (FP + TN) ; FNR = FN / (FN + TP)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        num_mal_vals.append(int(num_mal))
        tprs.append(tpr)
        tnrs.append(tnr)
        fprs.append(fpr)
        fnrs.append(fnr)
        counts.append(len(g))

    if not num_mal_vals:
        raise RuntimeError("No groups found to plot")

    # plotting grouped bars for TPR, TNR, FPR, FNR
    x = np.arange(len(num_mal_vals))
    width = 0.2

    plt.figure(figsize=(12, 6))
    bars_tpr = plt.bar(x - 1.5*width, tprs, width, label='TPR (recall pos)', color='tab:green')
    bars_tnr = plt.bar(x - 0.5*width, tnrs, width, label='TNR (recall neg)', color='tab:blue')
    bars_fpr = plt.bar(x + 0.5*width, fprs, width, label='FPR', color='tab:orange')
    bars_fnr = plt.bar(x + 1.5*width, fnrs, width, label='FNR', color='tab:red')

    # annotate helper
    def annotate_bars(bars):
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                txt = "n/a"
                va = 'center'
                y = 0.02
            else:
                txt = f"{h:.2f}"
                va = 'bottom'
                y = h + 0.01
            plt.text(b.get_x() + b.get_width() / 2, y, txt, ha='center', va=va, fontsize=9)

    annotate_bars(bars_tpr)
    annotate_bars(bars_tnr)
    annotate_bars(bars_fpr)
    annotate_bars(bars_fnr)

    plt.xticks(x, [str(v) for v in num_mal_vals])
    plt.xlabel('numberOfMal')
    plt.ylabel('Rate')
    plt.ylim(-0.02, 1.05)
    plt.title('TPR, TNR, FPR and FNR grouped by numberOfMal (aggregated over rows)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # optional subtitle with counts per group
    counts_str = ", ".join(f"{v}:{c}" for v, c in zip(num_mal_vals, counts))
    plt.gcf().text(0.99, 0.01, f"counts per group: {counts_str}", ha='right', fontsize=8, color='gray')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


# example usage:
plot_tpr_tnr_by_num_mal(r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\observation_scores.csv')
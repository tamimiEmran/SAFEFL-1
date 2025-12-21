import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
BASELINE_FILE =  r"M:\PythonTests\newSafeFL\SAFEFL\finalResults\21stDec.json"
FACTORGRAPH_FILE = r"M:\PythonTests\newSafeFL\SAFEFL\finalResults\factorGraphs_21stDec.json"

METHOD_MAP = {
    'fedavg': 'FedAvg',
    'krum': 'Krum',
    'shieldfl': 'ShieldFL',
    'signguard': 'SignGuard',
    'factorgraph': 'FactorGraphs'
}

ATTACK_MAP = {
    'label_flipping_attack': 'Label Flipping',
    'scaling_attack': 'Backdoor (Scaling)',
    'no': 'No Attack'
}

def load_data(files):
    all_rows = []
    for file_path in files:
        if not os.path.exists(file_path): continue
        with open(file_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            meta = entry['meta_data']
            res = entry['results']
            if not res.get('accuracy'): continue

            # Filters: Label Flipping Only (Active Attack)
            raw_attack = meta.get('byz_type', 'no')
            nbyz = meta.get('nbyz', 0)
            
            # We focus on the active attack scenarios for this chart
            if raw_attack != 'label_flipping_attack': continue 
            if nbyz == 6: continue # Skip unused default

            # Attributes
            raw_agg = meta.get('aggregation', 'fedavg')
            group_size = meta.get('group_size', 0)
            base_name = METHOD_MAP.get(raw_agg, raw_agg)
            group_label = "Grouped" if group_size > 0 else "Non-Grouped"
            
            # Metrics
            acc_hist = [x for x in res['accuracy'] if x is not None]
            final_acc = np.mean(acc_hist[-10:]) * 100 if acc_hist else 0

            all_rows.append({
                'dataset': meta.get('dataset', 'Unknown'),
                'bias': meta.get('bias', 0.0),
                'method': base_name,
                'grouping': group_label,
                'final_acc': final_acc
            })
            
    return pd.DataFrame(all_rows)

# ==========================================
# 2. GENERATE 4 SUBFIGURES (2x2 Grid)
# ==========================================
def plot_4_subfigures(df):
    """
    Creates a 2x2 grid of plots:
    Row 1: MNIST (Bias 0.0, Bias 0.5)
    Row 2: FEMNIST (Bias 0.0, Bias 0.5)
    """
    if df.empty:
        print("No data found for Label Flipping attack.")
        return

    # Create a column for the subtitle "Dataset (Bias=X)" to simplify plotting
    df['scenario'] = df.apply(lambda x: f"{x['dataset']} ($\\beta={x['bias']}$)", axis=1)
    
    # Sort scenarios to ensure logical order: MNIST 0.0, MNIST 0.5, FEMNIST 0.0, FEMNIST 0.5
    scenarios = sorted(df['scenario'].unique(), key=lambda x: (x.split()[0], x))
    
    # Setup Figure
    g = sns.catplot(
        data=df,
        x='method', 
        y='final_acc', 
        hue='grouping',
        col='scenario',     # Creates the subfigures
        col_wrap=2,         # Forces a 2x2 grid (wraps after 2 columns)
        kind='bar',
        height=4, 
        aspect=1.5,
        palette="viridis",
        errorbar=None,
        sharey=True         # Share Y axis to make comparison fair
    )
    
    # Customize Titles and Labels
    g.fig.suptitle('Impact of Grouping under Label Flipping Attack', fontsize=16, y=1.02)
    g.set_axis_labels("", "Accuracy (%)")
    g.set_titles("{col_name}", size=14)
    
    # Add annotations (numbers on bars)
    for ax in g.axes.flat:
        # Add gridlines for easier reading
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=9, color='black', 
                            xytext=(0, 3), textcoords='offset points')

    # Save
    filename = 'fig_grouping_impact_4panel.pdf'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    df = load_data([BASELINE_FILE, FACTORGRAPH_FILE])
    plot_4_subfigures(df)
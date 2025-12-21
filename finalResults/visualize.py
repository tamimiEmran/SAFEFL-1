
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASELINE_FILE =  r"M:\PythonTests\newSafeFL\SAFEFL\finalResults\21stDec.json"
FACTORGRAPH_FILE = r"M:\PythonTests\newSafeFL\SAFEFL\finalResults\factorGraphs_21stDec.json"
# Clean Names for Paper
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

            # Filters
            raw_attack = meta.get('byz_type', 'no')
            nbyz = meta.get('nbyz', 0)
            if raw_attack != 'no' and nbyz == 6: continue # Skip unused default

            # Attributes
            raw_agg = meta.get('aggregation', 'fedavg')
            group_size = meta.get('group_size', 0)
            base_name = METHOD_MAP.get(raw_agg, raw_agg)
            group_label = "Grouped" if group_size > 0 else "Non-Grouped"
            
            # Metrics
            acc_hist = [x for x in res['accuracy'] if x is not None]
            final_acc = np.mean(acc_hist[-10:]) * 100 if acc_hist else 0
            bsr_hist = [x for x in res.get('backdoor_success', []) if x is not None]
            final_bsr = np.mean(bsr_hist[-10:]) * 100 if bsr_hist else 0

            all_rows.append({
                'dataset': meta.get('dataset', 'Unknown'),
                'bias': meta.get('bias', 0.0),
                'method': base_name,
                'grouping': group_label,
                'attack': ATTACK_MAP.get(raw_attack, raw_attack),
                'raw_attack': raw_attack,
                'final_acc': final_acc,
                'final_bsr': final_bsr
            })
            
    return pd.DataFrame(all_rows)

# ==========================================
# FIGURE 1: EFFECT OF GROUPING (FACETED)
# ==========================================
def plot_grouping_impact(df):
    """
    Splits datasets into columns (Facets) to declutter the view.
    Comparison: Grouped vs Non-Grouped under Label Flipping.
    """
    subset = df[df['raw_attack'] == 'no']
    if subset.empty: return

    # Catplot automatically creates side-by-side subplots
    g = sns.catplot(
        data=subset, 
        x='method', 
        y='final_acc', 
        hue='grouping', 
        col='dataset',       # <--- This creates the Facets (Split by Dataset)
        kind='bar', 
        height=5, 
        aspect=1.2,
        palette="viridis",
        errorbar=None
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Impact of Grouping on Model Accuracy (No Attack)', fontsize=16)
    g.set_axis_labels("", "Accuracy (%)")
    g.set_titles("{col_name}")
    
    # Add value labels
    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5), 
                            textcoords='offset points')

    plt.savefig('fig_grouping_impact.pdf')
    print("Saved 'fig_grouping_impact.pdf'")

# ==========================================
# FIGURE 2: EFFECT OF BIAS (GROUPED ONLY)
# ==========================================
def plot_bias_robustness(df):
    """
    Compares Grouped vs Non-Grouped and IID (0.0) vs Non-IID (0.5).
    Uses a Scatter Plot to show the distribution and impact.
    """
    # Filter for specific attacks (Label Flipping & No Attack)
    subset = df[df['raw_attack'].isin(['label_flipping_attack', 'no'])].copy()
    if subset.empty: return

    # Convert bias to string for discrete coloring
    subset['bias'] = subset['bias'].astype(str)

    # Create a faceted scatter plot (relplot)
    # Rows: Attack Type
    # Cols: Dataset
    # X: Method
    # Y: Accuracy
    # Hue: Bias (Color shows the bias impact)
    # Style: Grouping (Shape shows Grouped vs Non-Grouped)
    g = sns.relplot(
        data=subset, 
        x='method', 
        y='final_acc', 
        hue='bias', 
        style='grouping',
        col='dataset',
        row='attack',
        kind='scatter', 
        s=200,             # Larger markers
        height=4, 
        aspect=1.5,
        palette="viridis", 
        alpha=0.8
    )
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Impact of Bias and Grouping on Accuracy', fontsize=16)
    g.set_axis_labels("Aggregation Method", "Accuracy (%)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    
    # Add grid for easier reading
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('fig_bias_robustness.pdf')
    print("Saved 'fig_bias_robustness.pdf'")

# FIGURE 3: BACKDOOR TRADEOFF (SCATTER)
# ==========================================
def plot_backdoor_tradeoff(df):
    """
    Scatter plot: Accuracy (X) vs Backdoor Success (Y).
    Best performance is Bottom-Right corner.
    Creates a 2x2 grid of subplots for each (dataset, bias) combination.
    """
    subset = df[df['raw_attack'] == 'scaling_attack']
    if subset.empty: return

    # Use relplot to create a grid of scatter plots
    g = sns.relplot(
        data=subset,
        x='final_acc',
        y='final_bsr',
        hue='method',
        style='grouping',
        col='bias',
        row='dataset',
        s=150,
        palette='deep',
        height=4,
        aspect=1.2
    )

    g.fig.subplots_adjust(top=0.9, right=0.85) # Adjust space for legend
    g.fig.suptitle('Backdoor Defense Trade-off: Accuracy vs. Attack Success', fontsize=16)

    g.set_axis_labels("Main Task Accuracy (%)", "Backdoor Success Rate (%)")
    g.set_titles(row_template="{row_name}", col_template="Bias = {col_name}")
    
    # Move legend outside
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.85, 0.6))


    # Add "Ideal Zone" annotation to each subplot
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.annotate('Ideal Zone', 
                     xy=(98, 2), xytext=(75, 25),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1))

    plt.savefig('fig_backdoor_tradeoff.pdf')
    print("Saved 'fig_backdoor_tradeoff.pdf'")


# ==========================================
# TABLE GENERATOR (CLEAN LATEX)
# ==========================================
def print_clean_latex(df):
    """
    Prints a table specifically highlighting the 'Gap' (Delta).
    """
    subset = df[df['raw_attack'] == 'label_flipping_attack']
    if subset.empty: return

    print("\n% --- LaTeX Table: Impact of Grouping (Delta Analysis) ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Impact of Grouping on Robustness (Label Flipping)}")
    print("\\begin{tabular}{l|cc|cc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{2}{c|}{\\textbf{MNIST (Acc \\%)}} & \\multicolumn{2}{c}{\\textbf{FEMNIST (Acc \\%)}} \\\\")
    print(" & Non-Grouped & Grouped ($\\Delta$) & Non-Grouped & Grouped ($\\Delta$) \\\\")
    print("\\midrule")

    for method in sorted(subset['method'].unique()):
        row_str = f"{method}"
        
        for ds in ['MNIST', 'FEMNIST']:
            non_g = subset[(subset['method']==method) & (subset['dataset']==ds) & (subset['grouping']=='Non-Grouped')]['final_acc'].mean()
            grouped = subset[(subset['method']==method) & (subset['dataset']==ds) & (subset['grouping']=='Grouped')]['final_acc'].mean()
            
            if pd.isna(non_g) or pd.isna(grouped):
                row_str += " & - & -"
            else:
                delta = grouped - non_g
                # Color the delta green if positive
                row_str += f" & {non_g:.1f} & \\textbf{{{grouped:.1f}}} (\\textcolor{{green}}{{+{delta:.1f}}})"
        
        row_str += " \\\\"
        print(row_str)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    df = load_data([BASELINE_FILE, FACTORGRAPH_FILE])
    
    if not df.empty:
        # Generate the visual highlights
        plot_grouping_impact(df)
        plot_bias_robustness(df)
        plot_backdoor_tradeoff(df)
        
        # Print the clean table
        print_clean_latex(df)
    else:
        print("No data found.")
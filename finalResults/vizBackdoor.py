import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
BASELINE_FILE = r"M:\PythonTests\newSafeFL\SAFEFL\finalResults\21stDec.json"
FACTORGRAPH_FILE = r"M:\PythonTests\newSafeFL\SAFEFL\finalResults\factorGraphs_21stDec.json"

METHOD_MAP = {
    'fedavg': 'FedAvg',
    'krum': 'Krum',
    'shieldfl': 'ShieldFL',
    'signguard': 'SignGuard',
    'factorgraph': 'FactorGraphs'
}

# Markers: Circle for Grouped, X for Non-Grouped to match your reference
MARKERS_MAP = {"Grouped": "o", "Non-Grouped": "X"}

def load_tradeoff_data(files):
    """
    Loads data specifically for Backdoor Trade-off analysis.
    Filters for 'scaling_attack' and extracts Accuracy & Backdoor Success Rate (BSR).
    """
    all_rows = []
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            meta = entry['meta_data']
            res = entry['results']
            
            # 1. STRICT FILTER: Only analyze Backdoor (scaling) attacks
            attack_type = meta.get('byz_type', 'no')
            if attack_type != 'scaling_attack': 
                continue

            # 2. Metadata
            dataset = meta.get('dataset', 'Unknown')
            bias = meta.get('bias', 0.0)
            
            # Create a combined label for the subfigure titles
            # e.g., "MNIST (Beta=0.0)"
            scenario_label = f"{dataset} ($\\beta={bias}$)"
            
            raw_agg = meta.get('aggregation', 'fedavg')
            method_name = METHOD_MAP.get(raw_agg, raw_agg)
            group_size = meta.get('group_size', 0)
            group_label = "Grouped" if group_size > 0 else "Non-Grouped"

            # 3. Metrics (Main Accuracy) - Last 10 average
            acc_hist = res.get('accuracy', [])
            if not acc_hist: continue
            final_acc = np.mean(acc_hist[-10:]) * 100

            # 4. Metrics (Backdoor Success Rate) - Last 10 average
            # Checks for both common key variations
            bsr_hist = res.get('backdoor_accuracy') or res.get('backdoor_success_rate') or []
            if not bsr_hist:
                # If BSR is missing in a backdoor attack run, skip to avoid bad data
                continue 
            final_bsr = np.mean(bsr_hist[-10:]) * 100

            all_rows.append({
                'Scenario': scenario_label,
                'Dataset': dataset,
                'Bias': bias,
                'Method': method_name,
                'Grouping': group_label,
                'Accuracy': final_acc,
                'BSR': final_bsr
            })

    df = pd.DataFrame(all_rows)
    
    # Sort for logical plotting order: MNIST first, then FEMNIST
    if not df.empty:
         df.sort_values(by=['Dataset', 'Bias'], inplace=True)
         
    return df

# ==========================================
# 2. PLOTTING FUNCTION (4 SUBFIGURES)
# ==========================================
def plot_4_panel_tradeoff(df):
    """
    Generates a 2x2 grid of scatter plots showing Acc vs BSR.
    One subplot for each Dataset x Bias combination.
    """
    if df.empty: 
        print("No valid data found for Scaling Attack trade-off analysis.")
        return

    # Use relplot for cleaner subplot grid creation with scatter plots
    g = sns.relplot(
        data=df,
        x="Accuracy", 
        y="BSR",
        hue="Method", 
        style="Grouping",     # Shapes distinguish Grouping
        markers=MARKERS_MAP,  # Force specific shapes (Circle vs X)
        col="Scenario",       # SPLIT by Scenario (Dataset x Bias)
        col_wrap=2,           # Wrap to create 2x2 grid
        kind="scatter",
        s=200,                # Large marker size like reference
        alpha=0.85,
        edgecolor='black',    # Add border to markers
        palette='deep',
        height=4.5, 
        aspect=1.1,
        # Lock axes 0-105 so all 4 plots are directly comparable
        facet_kws={'sharex': True, 'sharey': True, 'xlim': (-5, 105), 'ylim': (-5, 105)}
    )

    # -----------------------------
    # Styling and Annotations
    # -----------------------------
    g.fig.suptitle('Backdoor Defense Trade-off (Scaling Attack)', 
                   fontsize=18, y=1.03, fontweight='bold')
    
    g.set_axis_labels("Main Task Accuracy (%)", "Backdoor Success Rate (%)")
    g.set_titles("{col_name}", size=14, fontweight='bold')

    # Add "Ideal Zone" arrow to *every* subplot
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Arrow pointing to bottom-right
        ax.annotate('Ideal', 
                     xy=(98, 2),            # Arrow tip (High Acc, Low BSR)
                     xytext=(82, 25),       # Text location
                     arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Improve Legend
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)

    # Save
    filename = 'fig_backdoor_tradeoff_4panel.pdf'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load specific tradeoff data
    df = load_tradeoff_data([BASELINE_FILE, FACTORGRAPH_FILE])
    
    # 2. Check and Plot
    if not df.empty:
        print(f"Loaded {len(df)} scaling attack experiments.")
        print("Scenarios:", df['Scenario'].unique())
        plot_4_panel_tradeoff(df)
    else:
        print("DataFrame is empty. Check JSON paths and ensure 'scaling_attack' data exists.")
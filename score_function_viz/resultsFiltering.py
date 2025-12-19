"""
File Structure: allResults.json
Overview
Format: JSON array containing 110 experiment objects
Total Lines: 33,672 lines
Purpose: Stores results from federated learning (FL) experiments testing various aggregation rules against Byzantine attacks
Root Structure
1. meta_data Object — Experiment Configuration
Field	Type	Description	Observed Values
exp_id	int	Experiment identifier	0
net	string	Neural network model	"mobilenet_v3_small"
server_pc	int	Server participation count	100
dataset	string	Dataset used	"FEMNIST", "MNIST"
bias	float	Data distribution bias	0.0
p	float	Client participation rate	0.1
niter	int	Number of training iterations	500, 1000
nworkers	int	Total number of workers/clients	200
batch_size	int	Training batch size	32
lr	float	Learning rate	0.1
gpu	int	GPU device ID	0, 1, 4, etc.
seed	int	Random seed	1
nruns	int	Number of runs	1
test_every	int	Evaluation frequency (every N rounds)	10
isGrouped	bool	Whether clients are grouped	true
Aggregation Parameters
Field	Type	Description	Observed Values
aggregation	string	Aggregation rule used	"fedavg", "krum", "shieldfl", "signguard"
flod_threshold	float	FLOD threshold	0.5
flame_epsilon	int	FLAME epsilon	3000
flame_delta	float	FLAME delta	0.001
dnc_niters	int	DnC iterations	5
dnc_c	int	DnC c parameter	1
dnc_b	int	DnC b parameter	2000
Factor Graph / Bayesian Parameters
Field	Type	Description	Observed Values
factorGraphs_num_iters	int	Factor graph iterations	100
factorGraphs_temperature	float	Temperature parameter	0.1
factorGraphs_initial_threshold	float	Initial threshold	0.5
factorGraphs_observation_method	string	Observation method	"binarySignguard"
factorGraphs_likelihood_sigma	int	Likelihood sigma	2
factorGraphs_true_negative_rate	float	TNR	0.7
factorGraphs_true_positive_rate	float	TPR	0.7
factorGraphs_shuffling_strategy	string	Shuffling strategy	"random"
factorGraphs_highProbThreshold	float	High probability threshold	0.9
factorGraphs_prob_sort_temp	float	Probability sort temperature	0.1
Attack Parameters
Field	Type	Description	Observed Values
nbyz	int	Number of Byzantine/malicious workers	6, 10
byz_type	string	Attack type	"no", "label_flipping_attack", "scaling_attack"
group_size	int	Size of worker groups	0, 10
MPC-SPDZ Parameters (Secure Computation)
Field	Type	Description
mpspdz	bool	MPC-SPDZ enabled
port	int	Port number
chunk_size	int	Chunk size
protocol	string	Protocol type
players	int	Number of players
threads	int	Thread count
parallels	int	Parallel count
always_compile	bool	Always compile flag
metaArgs_id	string	Unique hash of config
2. results Object — Experiment Results
Field	Type	Description
rounds	int[]	Array of round numbers when evaluation occurred (e.g., [9, 19, 29, ...])
accuracy	float[]	Model accuracy at each evaluation round (0.0 to 1.0)
backdoor_success	float[] or null[]	Backdoor attack success rate at each round — null if no backdoor attack
stats	object	Summary statistics
stats Object
Field	Type	Description
best_accuracy	float	Highest accuracy achieved
last_accuracy	float	Final accuracy at last round
backdoor_success_at_best_accuracy	float or null	Backdoor success when best accuracy was reached
last_backdoor_success	float or null	Backdoor success at final round
Key Dimensions for Visualization
Primary Independent Variables
aggregation — 4 values: fedavg, krum, shieldfl, signguard
byz_type — 3 values: no, label_flipping_attack, scaling_attack
nbyz — Number of attackers: 6 or 10
group_size — Grouping configuration: 0 (no grouping) or 10
dataset — FEMNIST or MNIST
Primary Dependent Variables (Metrics)
accuracy — Time series showing model accuracy progression
backdoor_success — Time series showing attack success (when applicable)
best_accuracy — Peak performance achieved
last_accuracy — Final convergence point
Visualization Recommendations
Visualization Type	Data to Use
Line charts	rounds vs accuracy per aggregation rule
Heatmaps	best_accuracy by aggregation × byz_type
Grouped bar charts	Compare last_accuracy across aggregation methods
Dual-axis plots	accuracy and backdoor_success over rounds
Box plots	Distribution of best_accuracy per aggregation
Faceted grids	Accuracy curves faceted by dataset, nbyz, group_size
"""

# get the best performing (last accuracy) aggregation rule for each dataset, bias, nbyz, group_size, byz_type
import json
import pandas as pd

with open(r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\allResults.json', 'r') as f:
    data = json.load(f)

# create a dataframe with the following columns: dataset, bias, nbyz, group_size, byz_type, aggregation, best_accuracy, last_accuracy, backdoor_success_at_best_accuracy, last_backdoor_success
rows = []
for experiment in data:
    meta = experiment['meta_data']
    stats = experiment['results']['stats']
    rows.append({
        'dataset': meta['dataset'],
        'bias': meta['bias'],
        'nbyz': meta['nbyz'],
        'group_size': meta['group_size'],
        'byz_type': meta['byz_type'],
        'aggregation': meta['aggregation'],
        'best_accuracy': stats['best_accuracy'],
        'last_accuracy': stats['last_accuracy'],
        'backdoor_success_at_best_accuracy': stats['backdoor_success_at_best_accuracy'],
        'last_backdoor_success': stats['last_backdoor_success']
    })

df = pd.DataFrame(rows)

# Group by dataset, bias, nbyz, group_size, byz_type and find best performing aggregation rule (highest last_accuracy)
group_cols = ['dataset', 'bias', 'nbyz', 'group_size', 'byz_type']
best_per_group = df.loc[df.groupby(group_cols)['last_accuracy'].idxmax()]

print("Best performing aggregation rule for each configuration:")
print("=" * 80)
for _, row in best_per_group.iterrows():
    print(f"\nDataset: {row['dataset']}, Bias: {row['bias']}, NByz: {row['nbyz']}, "
          f"Group Size: {row['group_size']}, Attack: {row['byz_type']}")
    print(f"  Best Aggregation: {row['aggregation']}")
    print(f"  Last Accuracy: {row['last_accuracy']:.4f}")
    print(f"  Best Accuracy: {row['best_accuracy']:.4f}")

# Save to CSV

# Also create a pivot table for easier viewing
print("\n\nPivot Table - Best Aggregation by Dataset and Attack Type:")
print("=" * 80)
pivot = best_per_group.pivot_table(
    index=['dataset', 'nbyz', 'group_size'],
    columns='byz_type',
    values='aggregation',
    aggfunc='first'
)
print(pivot)


def compare_accuracy_no_attack_vs_label_flipping(df, dataset=None, nbyz=None, group_size=None, save_path=None):
    """
    Visually compare the last accuracy of all aggregation methods 
    between label_flipping_attack and no attacks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: aggregation, byz_type, last_accuracy, dataset, nbyz, group_size
    dataset : str, optional
        Filter by specific dataset (e.g., 'FEMNIST', 'MNIST')
    nbyz : int, optional
        Filter by number of Byzantine workers
    group_size : int, optional
        Filter by group size
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter for only 'no' and 'label_flipping_attack' attack types
    attack_types = ['no', 'label_flipping_attack']
    filtered_df = df[df['byz_type'].isin(attack_types)].copy()
    
    # Apply optional filters
    if dataset is not None:
        filtered_df = filtered_df[filtered_df['dataset'] == dataset]
    if nbyz is not None:
        filtered_df = filtered_df[filtered_df['nbyz'] == nbyz]
    if group_size is not None:
        filtered_df = filtered_df[filtered_df['group_size'] == group_size]
    
    if filtered_df.empty:
        print("No data matches the specified filters.")
        return None
    
    # Get unique aggregation methods
    aggregations = sorted(filtered_df['aggregation'].unique())
    
    # Create pivot table for comparison
    pivot_data = filtered_df.pivot_table(
        index='aggregation',
        columns='byz_type',
        values='last_accuracy',
        aggfunc='mean'
    ).reindex(aggregations)
    
    # Set up the figure with a dark theme for modern look
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar positioning
    x = np.arange(len(aggregations))
    width = 0.35
    
    # Colors - using a vibrant palette
    colors = {
        'no': '#2ecc71',  # Green for no attack
        'label_flipping_attack': '#e74c3c'  # Red for attack
    }
    
    # Create bars
    bars_no = ax.bar(
        x - width/2, 
        pivot_data.get('no', [0] * len(aggregations)), 
        width, 
        label='No Attack', 
        color=colors['no'],
        edgecolor='white',
        linewidth=1.2
    )
    bars_attack = ax.bar(
        x + width/2, 
        pivot_data.get('label_flipping_attack', [0] * len(aggregations)), 
        width, 
        label='Label Flipping Attack', 
        color=colors['label_flipping_attack'],
        edgecolor='white',
        linewidth=1.2
    )
    
    # Add value labels on bars
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold',
                            color='#2c3e50')
    
    add_bar_labels(bars_no)
    add_bar_labels(bars_attack)
    
    # Styling
    ax.set_xlabel('Aggregation Method', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Last Accuracy', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Build title with filter info
    title_parts = ['Last Accuracy Comparison: No Attack vs Label Flipping Attack']
    filter_info = []
    if dataset:
        filter_info.append(f'Dataset: {dataset}')
    if nbyz is not None:
        filter_info.append(f'NByz: {nbyz}')
    if group_size is not None:
        filter_info.append(f'Group Size: {group_size}')
    if filter_info:
        title_parts.append(f"({', '.join(filter_info)})")
    
    ax.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels([agg.upper() for agg in aggregations], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add accuracy drop annotation
    for i, agg in enumerate(aggregations):
        no_acc = pivot_data.loc[agg, 'no'] if 'no' in pivot_data.columns and pd.notna(pivot_data.loc[agg, 'no']) else None
        attack_acc = pivot_data.loc[agg, 'label_flipping_attack'] if 'label_flipping_attack' in pivot_data.columns and pd.notna(pivot_data.loc[agg, 'label_flipping_attack']) else None
        
        if no_acc is not None and attack_acc is not None:
            drop = no_acc - attack_acc
            drop_pct = (drop / no_acc) * 100 if no_acc > 0 else 0
            color = '#c0392b' if drop > 0 else '#27ae60'
            ax.annotate(f'{drop_pct:+.1f}%',
                        xy=(i, max(no_acc, attack_acc) + 0.06),
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def compare_accuracy_all_combinations(df, save_dir=None, show_plots=True):
    """
    Generate comparison figures for all combinations of nbyz, bias, dataset, and group_size.
    
    Note: When byz_type='no' (no attack), the nbyz value is a default placeholder
    and is ignored. The 'no attack' baseline is matched by (dataset, bias, group_size) only,
    then compared against each (dataset, bias, nbyz, group_size) attack configuration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: aggregation, byz_type, last_accuracy, dataset, nbyz, bias, group_size
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    show_plots : bool, default True
        Whether to display the plots. Set to False when only saving.
    
    Returns:
    --------
    list
        List of (params_dict, figure) tuples for all generated figures
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product
    import os
    
    # Separate 'no attack' and 'label_flipping_attack' data
    no_attack_df = df[df['byz_type'] == 'no'].copy()
    attack_df = df[df['byz_type'] == 'label_flipping_attack'].copy()
    
    # Get unique values for each parameter from attack data (nbyz matters here)
    datasets = sorted(attack_df['dataset'].unique())
    nbyz_values = sorted(attack_df['nbyz'].unique())
    bias_values = sorted(attack_df['bias'].unique())
    group_size_values = sorted(attack_df['group_size'].unique())
    
    print(f"Generating figures for all parameter combinations:")
    print(f"  Datasets: {datasets}")
    print(f"  NByz values (from attack data): {nbyz_values}")
    print(f"  Bias values: {bias_values}")
    print(f"  Group size values: {group_size_values}")
    print(f"  Total combinations: {len(datasets) * len(nbyz_values) * len(bias_values) * len(group_size_values)}")
    print("=" * 60)
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    figures = []
    
    # Colors - using a vibrant palette
    colors = {
        'no': '#2ecc71',  # Green for no attack
        'label_flipping_attack': '#e74c3c'  # Red for attack
    }
    
    for dataset, nbyz, bias, group_size in product(datasets, nbyz_values, bias_values, group_size_values):
        # Get 'no attack' baseline for this (dataset, bias, group_size) - ignore nbyz for no attack
        no_attack_data = no_attack_df[
            (no_attack_df['dataset'] == dataset) &
            (no_attack_df['bias'] == bias) &
            (no_attack_df['group_size'] == group_size)
        ]
        
        # Get attack data for this (dataset, nbyz, bias, group_size)
        attack_data = attack_df[
            (attack_df['dataset'] == dataset) &
            (attack_df['nbyz'] == nbyz) &
            (attack_df['bias'] == bias) &
            (attack_df['group_size'] == group_size)
        ]
        
        if no_attack_data.empty:
            print(f"  Skipping: Dataset={dataset}, NByz={nbyz}, Bias={bias}, GroupSize={group_size} (no baseline 'no attack' data)")
            continue
        
        if attack_data.empty:
            print(f"  Skipping: Dataset={dataset}, NByz={nbyz}, Bias={bias}, GroupSize={group_size} (no attack data)")
            continue
        
        print(f"  Generating: Dataset={dataset}, NByz={nbyz}, Bias={bias}, GroupSize={group_size}")
        
        # Combine the data for this comparison
        combo_df = pd.concat([no_attack_data, attack_data], ignore_index=True)
        
        # Get unique aggregation methods
        aggregations = sorted(combo_df['aggregation'].unique())
        
        # Create pivot table for comparison
        pivot_data = combo_df.pivot_table(
            index='aggregation',
            columns='byz_type',
            values='last_accuracy',
            aggfunc='mean'
        ).reindex(aggregations)
        
        # Set up the figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Bar positioning
        x = np.arange(len(aggregations))
        width = 0.35
        
        # Get values, handling missing columns
        no_values = pivot_data['no'].values if 'no' in pivot_data.columns else [0] * len(aggregations)
        attack_values = pivot_data['label_flipping_attack'].values if 'label_flipping_attack' in pivot_data.columns else [0] * len(aggregations)
        
        # Replace NaN with 0 for plotting
        no_values = np.nan_to_num(no_values, nan=0)
        attack_values = np.nan_to_num(attack_values, nan=0)
        
        # Create bars
        bars_no = ax.bar(
            x - width/2, 
            no_values, 
            width, 
            label='No Attack', 
            color=colors['no'],
            edgecolor='white',
            linewidth=1.2
        )
        bars_attack = ax.bar(
            x + width/2, 
            attack_values, 
            width, 
            label='Label Flipping Attack', 
            color=colors['label_flipping_attack'],
            edgecolor='white',
            linewidth=1.2
        )
        
        # Add value labels on bars
        def add_bar_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=9, fontweight='bold',
                                color='#2c3e50')
        
        add_bar_labels(bars_no)
        add_bar_labels(bars_attack)
        
        # Styling
        ax.set_xlabel('Aggregation Method', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Last Accuracy', fontsize=12, fontweight='bold', color='#2c3e50')
        
        # Title with parameter info
        title = f'Last Accuracy: No Attack vs Label Flipping Attack\nDataset: {dataset} | NByz: {nbyz} | Bias: {bias} | GroupSize: {group_size}'
        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels([agg.upper() for agg in aggregations], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # Add horizontal grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add accuracy drop annotation
        for i, agg in enumerate(aggregations):
            no_acc = pivot_data.loc[agg, 'no'] if 'no' in pivot_data.columns and pd.notna(pivot_data.loc[agg, 'no']) else None
            attack_acc = pivot_data.loc[agg, 'label_flipping_attack'] if 'label_flipping_attack' in pivot_data.columns and pd.notna(pivot_data.loc[agg, 'label_flipping_attack']) else None
            
            if no_acc is not None and attack_acc is not None and no_acc > 0:
                drop = no_acc - attack_acc
                drop_pct = (drop / no_acc) * 100
                color = '#c0392b' if drop > 0 else '#27ae60'
                ax.annotate(f'{drop_pct:+.1f}%',
                            xy=(i, max(no_acc, attack_acc) + 0.06),
                            ha='center', va='bottom',
                            fontsize=8, fontweight='bold',
                            color=color,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if directory specified
        if save_dir:
            filename = f"accuracy_comparison_{dataset}_nbyz{nbyz}_bias{bias}_groupsize{group_size}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"    Saved: {filepath}")
        
        # Store figure info
        params = {'dataset': dataset, 'nbyz': nbyz, 'bias': bias, 'group_size': group_size}
        figures.append((params, fig))
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print("=" * 60)
    print(f"Generated {len(figures)} figures total.")
    
    return figures


def compare_accuracy_scaling_attack_all_combinations(df, save_dir=None, show_plots=True):
    """
    Generate comparison figures for scaling attack across all combinations of nbyz, bias, dataset, and group_size.
    Each figure shows two subplots: (1) Accuracy comparison, (2) Backdoor success rate.
    
    Note: When byz_type='no' (no attack), the nbyz value is a default placeholder
    and is ignored. The 'no attack' baseline is matched by (dataset, bias, group_size) only,
    then compared against each (dataset, bias, nbyz, group_size) attack configuration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: aggregation, byz_type, last_accuracy, last_backdoor_success, 
        dataset, nbyz, bias, group_size
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    show_plots : bool, default True
        Whether to display the plots. Set to False when only saving.
    
    Returns:
    --------
    list
        List of (params_dict, figure) tuples for all generated figures
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product
    import os
    
    # Separate 'no attack' and 'scaling_attack' data
    no_attack_df = df[df['byz_type'] == 'no'].copy()
    attack_df = df[df['byz_type'] == 'scaling_attack'].copy()
    
    if attack_df.empty:
        print("No scaling_attack data found in the dataset.")
        return []
    
    # Get unique values for each parameter from attack data (nbyz matters here)
    datasets = sorted(attack_df['dataset'].unique())
    nbyz_values = sorted(attack_df['nbyz'].unique())
    bias_values = sorted(attack_df['bias'].unique())
    group_size_values = sorted(attack_df['group_size'].unique())
    
    print(f"Generating figures for SCALING ATTACK - all parameter combinations:")
    print(f"  Datasets: {datasets}")
    print(f"  NByz values (from attack data): {nbyz_values}")
    print(f"  Bias values: {bias_values}")
    print(f"  Group size values: {group_size_values}")
    print(f"  Total combinations: {len(datasets) * len(nbyz_values) * len(bias_values) * len(group_size_values)}")
    print("=" * 60)
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    figures = []
    
    # Colors - using a vibrant palette
    colors = {
        'no': '#2ecc71',  # Green for no attack
        'scaling_attack': '#9b59b6'  # Purple for scaling attack
    }
    
    for dataset, nbyz, bias, group_size in product(datasets, nbyz_values, bias_values, group_size_values):
        # Get 'no attack' baseline for this (dataset, bias, group_size) - ignore nbyz for no attack
        no_attack_data = no_attack_df[
            (no_attack_df['dataset'] == dataset) &
            (no_attack_df['bias'] == bias) &
            (no_attack_df['group_size'] == group_size)
        ]
        
        # Get attack data for this (dataset, nbyz, bias, group_size)
        attack_data = attack_df[
            (attack_df['dataset'] == dataset) &
            (attack_df['nbyz'] == nbyz) &
            (attack_df['bias'] == bias) &
            (attack_df['group_size'] == group_size)
        ]
        
        if no_attack_data.empty:
            print(f"  Skipping: Dataset={dataset}, NByz={nbyz}, Bias={bias}, GroupSize={group_size} (no baseline 'no attack' data)")
            continue
        
        if attack_data.empty:
            print(f"  Skipping: Dataset={dataset}, NByz={nbyz}, Bias={bias}, GroupSize={group_size} (no attack data)")
            continue
        
        print(f"  Generating: Dataset={dataset}, NByz={nbyz}, Bias={bias}, GroupSize={group_size}")
        
        # Combine the data for this comparison
        combo_df = pd.concat([no_attack_data, attack_data], ignore_index=True)
        
        # Get unique aggregation methods
        aggregations = sorted(combo_df['aggregation'].unique())
        
        # Create pivot tables for comparison
        pivot_accuracy = combo_df.pivot_table(
            index='aggregation',
            columns='byz_type',
            values='last_accuracy',
            aggfunc='mean'
        ).reindex(aggregations)
        
        pivot_backdoor = combo_df.pivot_table(
            index='aggregation',
            columns='byz_type',
            values='last_backdoor_success',
            aggfunc='mean'
        ).reindex(aggregations)
        
        # Set up the figure with 2 subplots
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Bar positioning
        x = np.arange(len(aggregations))
        width = 0.35
        
        # ============== SUBPLOT 1: ACCURACY ==============
        # Get values, handling missing columns
        no_acc_values = pivot_accuracy['no'].values if 'no' in pivot_accuracy.columns else [0] * len(aggregations)
        attack_acc_values = pivot_accuracy['scaling_attack'].values if 'scaling_attack' in pivot_accuracy.columns else [0] * len(aggregations)
        
        # Replace NaN with 0 for plotting
        no_acc_values = np.nan_to_num(no_acc_values, nan=0)
        attack_acc_values = np.nan_to_num(attack_acc_values, nan=0)
        
        # Create bars for accuracy
        bars_no_acc = ax1.bar(
            x - width/2, 
            no_acc_values, 
            width, 
            label='No Attack', 
            color=colors['no'],
            edgecolor='white',
            linewidth=1.2
        )
        bars_attack_acc = ax1.bar(
            x + width/2, 
            attack_acc_values, 
            width, 
            label='Scaling Attack', 
            color=colors['scaling_attack'],
            edgecolor='white',
            linewidth=1.2
        )
        
        # Add value labels on accuracy bars
        def add_bar_labels(ax, bars, fontsize=8):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=fontsize, fontweight='bold',
                                color='#2c3e50')
        
        add_bar_labels(ax1, bars_no_acc)
        add_bar_labels(ax1, bars_attack_acc)
        
        # Styling for accuracy subplot
        ax1.set_xlabel('Aggregation Method', fontsize=11, fontweight='bold', color='#2c3e50')
        ax1.set_ylabel('Last Accuracy', fontsize=11, fontweight='bold', color='#2c3e50')
        ax1.set_title('Model Accuracy', fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels([agg.upper() for agg in aggregations], fontsize=10, fontweight='bold', rotation=15, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Add accuracy drop annotation
        for i, agg in enumerate(aggregations):
            no_acc = pivot_accuracy.loc[agg, 'no'] if 'no' in pivot_accuracy.columns and pd.notna(pivot_accuracy.loc[agg, 'no']) else None
            attack_acc = pivot_accuracy.loc[agg, 'scaling_attack'] if 'scaling_attack' in pivot_accuracy.columns and pd.notna(pivot_accuracy.loc[agg, 'scaling_attack']) else None
            
            if no_acc is not None and attack_acc is not None and no_acc > 0:
                drop = no_acc - attack_acc
                drop_pct = (drop / no_acc) * 100
                color = '#c0392b' if drop > 0 else '#27ae60'
                ax1.annotate(f'{drop_pct:+.1f}%',
                            xy=(i, max(no_acc, attack_acc) + 0.05),
                            ha='center', va='bottom',
                            fontsize=7, fontweight='bold',
                            color=color,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))
        
        # ============== SUBPLOT 2: BACKDOOR SUCCESS ==============
        # Get backdoor success values (only for scaling attack, no attack has null)
        backdoor_values = pivot_backdoor['scaling_attack'].values if 'scaling_attack' in pivot_backdoor.columns else [0] * len(aggregations)
        backdoor_values = np.nan_to_num(backdoor_values, nan=0)
        
        # Create bars for backdoor success - single bar per aggregation
        bar_colors = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.2 else '#27ae60' for v in backdoor_values]
        bars_backdoor = ax2.bar(
            x, 
            backdoor_values, 
            width * 1.5, 
            label='Backdoor Success Rate',
            color=bar_colors,
            edgecolor='white',
            linewidth=1.2
        )
        
        # Add value labels on backdoor bars
        for bar in bars_backdoor:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color='#2c3e50')
        
        # Styling for backdoor subplot
        ax2.set_xlabel('Aggregation Method', fontsize=11, fontweight='bold', color='#2c3e50')
        ax2.set_ylabel('Backdoor Success Rate', fontsize=11, fontweight='bold', color='#2c3e50')
        ax2.set_title('Backdoor Attack Success (Lower is Better)', fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels([agg.upper() for agg in aggregations], fontsize=10, fontweight='bold', rotation=15, ha='right')
        ax2.set_ylim(0, 1.0)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)
        
        # Add color legend for backdoor
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', edgecolor='white', label='Low (<20%)'),
            Patch(facecolor='#f39c12', edgecolor='white', label='Medium (20-50%)'),
            Patch(facecolor='#e74c3c', edgecolor='white', label='High (>50%)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, title='Risk Level')
        
        # Main title for entire figure
        fig.suptitle(f'Scaling Attack Analysis\nDataset: {dataset} | NByz: {nbyz} | Bias: {bias} | GroupSize: {group_size}',
                     fontsize=14, fontweight='bold', color='#2c3e50', y=1.02)
        
        plt.tight_layout()
        
        # Save figure if directory specified
        if save_dir:
            filename = f"scaling_attack_{dataset}_nbyz{nbyz}_bias{bias}_groupsize{group_size}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"    Saved: {filepath}")
        
        # Store figure info
        params = {'dataset': dataset, 'nbyz': nbyz, 'bias': bias, 'group_size': group_size}
        figures.append((params, fig))
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print("=" * 60)
    print(f"Generated {len(figures)} figures total.")
    
    return figures


# Example usage - uncomment to run:
if __name__ == "__main__":
    # Option 1: Generate figures for label flipping attack (all parameter combinations)
    print("\n\nGenerating comparison visualizations for LABEL FLIPPING attack...")
    compare_accuracy_all_combinations(df, save_dir='comparison_figures', show_plots=True)
    
    # Option 2: Generate figures for scaling attack with accuracy + backdoor success
    print("\n\nGenerating comparison visualizations for SCALING attack...")
    compare_accuracy_scaling_attack_all_combinations(df, save_dir='scaling_attack_figures', show_plots=True)
    
    # Option 3: Generate a single comparison figure with specific filters
    # compare_accuracy_no_attack_vs_label_flipping(df, dataset='FEMNIST', nbyz=10)
    
    # Option 4: Save all figures without displaying
    # compare_accuracy_all_combinations(df, save_dir='comparison_figures', show_plots=False)
    # compare_accuracy_scaling_attack_all_combinations(df, save_dir='scaling_attack_figures', show_plots=False)
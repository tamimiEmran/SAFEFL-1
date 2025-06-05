import subprocess
import time
import os
import csv
import json

# Ensure results directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/hierarchical", exist_ok=True)
os.makedirs("resultsHFL/hierarchical", exist_ok=True)
"""
["no", "trim_attack", "krum_attack",
                            "scaling_attack", "fltrust_attack", "label_flipping_attack", "min_max_attack", "min_sum_attack"]

"""


# Define protocols and their parameters
protocols = {
    "heirichalFL": ["--aggregation", "heirichalFL", "--n_groups", "10", "--assumed_mal_prct", "0.6"],
    "fedavg": ["--aggregation", "fedavg"],
    "flame": ["--aggregation", "flame", "--flame_epsilon", "3000", "--flame_delta", "0.001"],
    "divide_and_conquer": ["--aggregation", "divide_and_conquer", "--dnc_niters", "5", "--dnc_c", "1", "--dnc_b", "2000"],
    "fltrust": ["--aggregation", "fltrust", "--server_pc", "100"]
}

protocols = {
    "heirichalFL": ["--aggregation", "heirichalFL", "--n_groups", "10", "--assumed_mal_prct", "0.3"]
}

# Parameters to vary (modify these lists to run different experiments)
nbyz_values = [10, 5,0]
byz_types = ["no", "scaling_attack", "label_flipping_attack"]
bias_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7][::-1] # max is 0.7

# Additional parameters specific to heirichalFL
"""
n_groups_values = [5, 10]
mal_prct_values = [0.2]
"""
# Fixed parameters
dataset = "MNIST"
niter = 200
test_every = 50
nruns = 1
seed = 42
nworkers = 50
batch_size = 8
p = 0.5
experiment_id = 23 # the full final experiments for the aggregation rule comparisons is 10. dont change this comment.


# Common parameters for all experiments
common_params = [
    "--dataset", dataset,
    "--niter", str(niter),
    "--test_every", str(test_every),
    "--nruns", str(nruns),
    "--seed", str(seed),
    "--nworkers", str(nworkers),
    "--batch_size", str(batch_size),
    "--p", str(p),
    "--exp_id", str(experiment_id),
    "--gpu", "1",  # Use GPU 1 for training
]

# Function to save aggregation parameters to CSV
def save_agg_parameters(experiment_id):
    """Save the aggregation rule parameters to a CSV file for reference in the dashboard"""
    csv_file = f"results/hierarchical/agg_parameters_{experiment_id}.csv"

    # Prepare data for CSV
    rows = []
    for agg_name, params in protocols.items():
        # Convert the parameter list to a dictionary for easier reading
        param_dict = {}
        i = 0
        while i < len(params):
            if params[i].startswith('--'):
                param_name = params[i][2:]  # Remove -- prefix
                if i + 1 < len(params) and not params[i + 1].startswith('--'):
                    param_value = params[i + 1]
                    i += 2
                else:
                    param_value = True  # Flag-only parameter
                    i += 1
                param_dict[param_name] = param_value
            else:
                i += 1

        # Add to rows
        rows.append({
            'experiment_id': experiment_id,
            'aggregation_name': agg_name,
            'parameters': json.dumps(param_dict)
        })

    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment_id', 'aggregation_name', 'parameters'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved aggregation parameters to {csv_file}")

# Track results
successful = []
failed = []


# Function to run a single experiment
def run_experiment(cmd, description, protocol_name=None, params=None):
    """
    Run a single experiment with the given command and description.

    Args:
        cmd: Command to run
        description: Description of the experiment
        protocol_name: Name of the protocol being run
        params: Additional parameters specific to this experiment run
    """
    print(f"Running experiment: {description}")
    print(f"\n{'='*80}")

    start_time = time.time()
    result = subprocess.run(cmd)
    end_time = time.time()

    success = result.returncode == 0
    status = "completed successfully" if success else f"failed with return code {result.returncode}"
    print(f"Experiment {status} in {end_time - start_time:.2f} seconds")

    if success:
        successful.append(description)
    else:
        failed.append(description)

# Save aggregation parameters to CSV for reference
save_agg_parameters(experiment_id)

# Main execution
print("Starting experiments...")

# Loop through all protocols
for protocol_name, protocol_params in protocols.items():
    if False:
        # Special case for heirichalFL - vary more parameters
        for nbyz in nbyz_values:
            for byz_type in byz_types:
                for bias in bias_values:
                    for n_groups in n_groups_values:
                        for mal_prct in mal_prct_values:
                            # Build command
                            cmd = ["python3", "main.py"] + common_params + protocol_params + [
                                "--nbyz", str(nbyz),
                                "--byz_type", byz_type,
                                "--bias", str(bias),
                                "--n_groups", str(n_groups),
                                "--assumed_mal_prct", str(mal_prct)
                            ]

                            description = f"{protocol_name}: nbyz={nbyz}, attack={byz_type}, bias={bias}, n_groups={n_groups}, mal_prct={mal_prct}"
                            run_experiment(cmd, description, protocol_name=protocol_name)
    else:
        # For other protocols
        for nbyz in nbyz_values:
            for byz_type in byz_types:
                if (byz_type == "no" and nbyz > 0) or (byz_type != "no" and nbyz == 0):
                    continue
                for bias in bias_values:
                    # Build command
                    cmd = ["python3", "main.py"] + common_params + protocol_params + [
                        "--nbyz", str(nbyz),
                        "--byz_type", byz_type,
                        "--bias", str(bias)
                    ]

                    description = f"{protocol_name}: nbyz={nbyz}, attack={byz_type}, bias={bias}"
                    run_experiment(cmd, description, protocol_name=protocol_name)

# Print summary
print("\n" + "="*80)
print(f"EXPERIMENT SUMMARY")
print(f"Successful experiments: {len(successful)}")
print(f"Failed experiments: {len(failed)}")
print("="*80)

if successful:
    print("\nSuccessful experiments:")
    for exp in successful:
        print(f"  - {exp}")

if failed:
    print("\nFailed experiments:")
    for exp in failed:
        print(f"  - {exp}")
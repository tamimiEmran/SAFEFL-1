import subprocess
import time
import os
import itertools


# Basic experiment settings (fixed parameters)
dataset = "MNIST"
niter = 200
test_every = 200
nruns = 1
seed = 42
nworkers = 50
batch_size = 8
p = 0.1


# Attack parameters to vary
nbyz_values = [5]  # Number of Byzantine workers
byz_types = ["scaling_attack", "label_flipping_attack"]  # Attack types
bias_values = [0.0, 0.25, 0.5]  # Bias values

# HFL specific parameters to vary
n_groups_values = [10]  # Number of groups
mal_prct_values = [0.4]  # Assumed malicious percentage

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
    
    "--aggregation", "heirichalFL"  # Fixed for hierarchical FL
]

# Function to run a single experiment
def run_experiment(cmd, description):
    print(f"\n{'='*80}")
    print(f"Running experiment: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd)
    end_time = time.time()
    
    success = result.returncode == 0
    status = "completed successfully" if success else f"failed with return code {result.returncode}"
    print(f"Experiment {status} in {end_time - start_time:.2f} seconds")
    
    return (description, success)

# Main execution
def main():
    # Track results
    successful = []
    failed = []
    
    print("Starting hierarchical FL experiments...")
    
    # Loop through all parameter combinations for HFL
    expStart = 0
    for nbyz in nbyz_values:
        for byz_type in byz_types:
            # Skip invalid combinations (no attack but with Byzantine workers or attack with no Byzantine workers)
            if (byz_type == "no" and nbyz > 0) or (byz_type != "no" and nbyz == 0):
                continue
                
            for bias in bias_values:
                for n_groups in n_groups_values:
                    for mal_prct in mal_prct_values:
                        expStart += 1
                        
                        # Build command
                        cmd = ["python", "main.py"] + common_params + [
                            "--nbyz", str(nbyz),
                            "--byz_type", byz_type,
                            "--bias", str(bias),
                            "--n_groups", str(n_groups),
                            "--assumed_mal_prct", str(mal_prct),
                            "--exp_id", str(expStart)
                        ]
                        
                        description = f"HeirichalFL: nbyz={nbyz}, attack={byz_type}, bias={bias}, n_groups={n_groups}, mal_prct={mal_prct}"
                        
                        result, success = run_experiment(cmd, description)
                        
                        if success:
                            successful.append(result)
                        else:
                            failed.append(result)
    
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

if __name__ == "__main__":
    main()
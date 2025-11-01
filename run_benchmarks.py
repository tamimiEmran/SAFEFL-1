import subprocess
import sys
import shlex

#!/usr/bin-env python3
# /m:/PythonTests/newSafeFL/SAFEFL/run_benchmarks.py
#
# --- Corrected and Refactored Benchmark Script ---

# customize these lists as needed
#datasets = ["FEMNIST", 'MNIST', 'CIFAR10']
datasets = ['FEMNIST','MNIST'] # for debugging purposes, only run MNIST
bias_values = [0, 0.5] # for debugging purposes, only run 0
#models = ["resnet18", "mobilenet_v3_small", "vit_base"] # Only keep resnet18
models = ["resnet18"]
attack_types = ["no", "scaling_attack", "label_flipping_attack"]
#full_list_defences = ["fedavg", "krum", "trim_mean", "median", "flame", "shieldfl", "divide_and_conquer", "signguard", "flare", "romoa"] # Keep only 
defences = ['fedavg', 'krum', 'shieldfl', 'signguard']
isGrouped_list = [True, False]
group_size_list = [10, 20] # for debugging purposes, only run 10
nbyz_list = [25, 50] # for debugging purposes, only run 2

base_args = [
    "--nworkers", "500",
    "--batch_size", "128",
    "--niter", "2500", #2500
    "--lr", "0.05",
    "--test_every", "10", #10
    "--gpu", "1",
]

def execute_command(cmd_list):
    """
    Helper function to build, print, and run a single experiment command.
    Includes try/except for error handling.
    """
    # Build the final command
    cmd = [sys.executable, "main.py"] + base_args + cmd_list
    
    # Use shlex.join for safe and readable printing
    print("Running:", shlex.join(cmd))
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"--- COMMAND FAILED ---")
        print(f"Command: {shlex.join(cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"----------------------")
    except FileNotFoundError:
        print(f"--- ERROR: 'main.py' not found. ---")
        print(f"Make sure you are running this script from the correct directory.")
        # Exit the script if main.py is not found, as no experiments can run.
        sys.exit(1)

def run_experiment_set(base_cmd_args, attack_type):
    """
    Helper function to run the correct set of experiments
    based on the attack type (with or without nbyz).
    """
    if attack_type == "no":
        # If there is no attack, run the command once without --nbyz
        execute_command(base_cmd_args)
    else:
        # If there is an attack, loop through all nbyz values
        for n_byz in nbyz_list:
            # Add the --nbyz argument for this specific run
            attack_cmd_args = base_cmd_args + ["--nbyz", str(n_byz)]
            execute_command(attack_cmd_args)

# --- Main Experiment Loops ---
def calculate_total_experiments():
    """
    Calculates the total number of experiments that will be run
    based on the lists defined at the top of the script.
    """
    # 1. Calculate the number of base combinations
    base_combinations = (
        len(datasets) *
        len(bias_values) *
        len(models) *
        len(defences)
    )
    
    # 2. Calculate the number of runs for the inner loops
    #    (attacks, grouping, and nbyz)
    inner_loop_runs = 0
    for attack in attack_types:
        for toGroup in isGrouped_list:
            if toGroup:
                # Grouped experiments
                if attack == "no":
                    inner_loop_runs += len(group_size_list)
                else:
                    inner_loop_runs += len(group_size_list) * len(nbyz_list)
            else:
                # Non-grouped experiments
                if attack == "no":
                    inner_loop_runs += 1
                else:
                    inner_loop_runs += len(nbyz_list)
                    
    total = base_combinations * inner_loop_runs
    return total

total_experiments = calculate_total_experiments()
print(f"Total number of experiments to run: {total_experiments}")

for dataset in datasets:
    for bias in bias_values:
        for model in models:
            for attack_type in attack_types:
                for defence in defences:
                    for toGroup in isGrouped_list:
                        
                        # Build the common arguments for this run
                        current_run_args = [
                            "--dataset", dataset,
                            "--bias", str(bias),
                            "--net", model,
                            "--byz_type", attack_type,
                            "--aggregation", defence,
                            "--isGrouped", str(toGroup)
                        ]
                        

                        if toGroup:
                            # --- Grouped Experiments ---
                            for g_size in group_size_list:
                                # Add the group_size argument
                                grouped_args = current_run_args + ["--group_size", str(g_size)]
                                # Run the set of experiments (handles 'no' vs. 'attack' logic)
                                run_experiment_set(grouped_args, attack_type)
                        else:
                            # --- Non-Grouped Experiments ---
                            # No --group_size argument is added
                            # Run the set of experiments (handles 'no' vs. 'attack' logic)
                            run_experiment_set(current_run_args, attack_type)

print("--- Benchmark Sweep Complete ---")
import subprocess
import sys
import shlex

#!/usr/bin-env python3
# /m:/PythonTests/newSafeFL/SAFEFL/run_benchmarks.py
#
# --- Corrected and Refactored Benchmark Script ---

# customize these lists as needed
datasets = ['FEMNIST','MNIST'] # for debugging purposes, only run MNIST
bias_values = [0,0.25, 0.5] # for debugging purposes, only run 0
models = ["mobilenet_v3_small", "eff_net"][::-1]
attack_types = ["scaling_attack"] #, "scaling_attack", "label_flipping_attack"
defences = ['fedavg', 'krum', 'shieldfl', 'signguard']
isGrouped_list = [True, False]
group_size_list = [5, 10] 
nbyz_list = [10] 

base_args = [
    "--nworkers", "100",
    "--batch_size", "256",
    "--niter", "3000", #2500
    "--lr", "0.05",
    "--test_every", "30", #10
    "--gpu", "1",
]
print("THIS IS FOR THE SECOND POD NAME 'ATTACK TYPE' 'scaling_attack'")
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
    
    *** NOTE: This function has been updated to account for the
    special 'fedavg' logic. ***
    """
    # 1. Calculate the number of base combinations
    #    (Note: 'defences' loop is now handled *inside* the inner runs)
    base_combinations = (
        len(datasets) *
        len(bias_values) *
        len(models)
    )
    
    # 2. Calculate the number of runs for the inner loops
    #    (attacks, defences, grouping, and nbyz)
    inner_loop_runs = 0
    for attack in attack_types:
        for defence in defences:
            if defence == 'fedavg':
                # --- FedAvg Special Case ---
                # Runs once (as non-grouped)
                if attack == "no":
                    inner_loop_runs += 1
                else:
                    inner_loop_runs += len(nbyz_list) # Runs for each nbyz
            else:
                # --- All Other Defences ---
                # Runs for both grouped and non-grouped
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
                    
                    # --- LOGIC FIX ---
                    # The 'fedavg' special case is now handled here.
                    # We build its args and run it, OR
                    # we proceed to the grouping logic for all other defenses.
                    
                    if defence == 'fedavg':
                        # --- FedAvg Special Case ---
                        # Build args for a *single* non-grouped run
                        fedavg_args = [
                            "--dataset", dataset,
                            "--bias", str(bias),
                            "--net", model,
                            "--byz_type", attack_type,
                            "--aggregation", defence,
                            "--isGrouped", "False" # Hardcode as False
                        ]
                        run_experiment_set(fedavg_args, attack_type)
                    else:
                        # --- All Other Defences ---
                        # Loop through grouped and non-grouped
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
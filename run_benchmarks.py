import subprocess
import sys
import shlex
# femnist fedavg no bias, no att. 
# [0.1582, 0.3840, 0.4906, 0.5742, 0.6357, 0.6336, 0.7013, 0.7385, 0.7653, 0.7712, 0.7705, 0.7885, 0.7990, 0.7899, 0.8107, 0.8107, 0.8252, 0.8091, 0.8220, 0.8384, 0.8366, 0.8348, 0.8417, 0.8337, 0.8363, 0.8453, 0.8442, 0.8425, 0.8339, 0.8491, 0.8494, 0.8520, 0.8542, 0.8556, 0.8483, 0.8447, 0.8549, 0.8535, 0.8551, 0.8569, 0.8559, 0.8580, 0.8517, 0.8163, 0.8484, 0.8537, 0.8578, 0.8588, 0.8579, 0.8626, 0.8585, 0.8612, 0.8597, 0.8642]
#Running: /workspace/.venv_torch/bin/python main.py --nworkers 100 --batch_size 256 --niter 3000 --lr 0.05 --test_every 30 --gpu 1 --dataset FEMNIST --bias 0 --net eff_net --byz_type no --aggregation fedavg --isGrouped False

#!/usr/bin-env python3
# /m:/PythonTests/newSafeFL/SAFEFL/run_benchmarks.py
#
# --- Corrected and Refactored Benchmark Script ---

# customize these lists as needed
datasets = ['FEMNIST','MNIST'] # for debugging purposes, only run MNIST
bias_values = [0,0.25, 0.5] # for debugging purposes, only run 0
models = ["mobilenet_v3_small", "eff_net"][::-1]
attack_types = ["label_flipping_attack"] #, "scaling_attack", "label_flipping_attack"
defences = ['fedavg', 'krum', 'shieldfl', 'signguard']
isGrouped_list = [True, False]
group_size_list = [5, 10]
nbyz_list = [0, 10]

# prompt the user to enter the models list, attack types list and the defences list
#explain to the user how to enter the list as a string separated by commas. Then give the options for the models, attack types and defences.
print("Enter the models list as a string separated by commas. The options are: mobilenet_v3_small, eff_net")
print("Enter the attack types list as a string separated by commas. The options are: no, label_flipping_attack, scaling_attack")
print("Enter the defences list as a string separated by commas. The options are: fedavg, krum, shieldfl, signguard")    
models = input("Enter the models list: ").split(",")
attack_types = input("Enter the attack types list: ").split(",")
defences = input("Enter the defences list: ").split(",")


base_args = [
    "--nworkers", "100",
    "--batch_size", "128",
    "--niter", "1000", #2500
    "--lr", "0.1",
    "--test_every", "10", #10
    "--gpu", "1",
]
print("THIS IS FOR THE third POD NAME 'ATTACK TYPE' 'label_flipping_attack'")
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
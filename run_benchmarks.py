import subprocess
import sys
import shlex

#!/usr/bin-env python3
# /m:/PythonTests/newSafeFL/SAFEFL/run_benchmarks.py
#
# --- Corrected and Refactored Benchmark Script ---

# customize these lists as needed
datasets = ["FEMNIST", 'MNIST', 'CIFAR10']
bias_values = [0, 0.5]
models = ["resnet18", "mobilenet_v3_small", "vit_base"] # Only keep resnet18
models = ["resnet18"]
attack_types = ["no", "scaling_attack", "label_flipping_attack"]
full_list_defences = ["fedavg", "krum", "trim_mean", "median", "flame", "shieldfl", "divide_and_conquer", "signguard", "flare", "romoa"] # Keep only 
defences = ['fedavg', 'krum', 'shieldfl', 'signguard']
isGrouped_list = [True, False]
group_size_list = [2, 5, 10, 20]
nbyz_list = [25, 50]

base_args = [
    "--nworkers", "500",
    "--batch_size", "32",
    "--niter", "2", #2500
    "--lr", "0.05",
    "--test_every", "1", #10
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
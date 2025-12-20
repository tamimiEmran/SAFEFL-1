import json
import math
import sys

# --- CONFIGURATION (Matches your run_benchmarks.py) ---
datasets = ['FEMNIST', 'MNIST']
bias_values = [0, 0.5]
models = ["mobilenet_v3_small"]
attack_types = ['no', 'scaling_attack', "label_flipping_attack"]
defences = ['fedavg', 'krum', 'shieldfl', 'signguard']
isGrouped_list = [True, False]
group_size_list = [10]
nbyz_list = [20] 

# Base args for command generation
femnist_base_args = ["--nworkers", "200", "--batch_size", "32", "--niter", "1000", "--lr", "0.1", "--test_every", "10"]
mnist_base_args = ["--nworkers", "200", "--batch_size", "32", "--niter", "500", "--lr", "0.1", "--test_every", "10"]

def generate_command(args_dict):
    dataset = args_dict['dataset']
    base = femnist_base_args if dataset == "FEMNIST" else mnist_base_args
    cmd = ["python", "main.py"] + base
    cmd.extend(["--dataset", dataset, "--bias", str(args_dict['bias']), "--net", args_dict['net']])
    cmd.extend(["--byz_type", args_dict['byz_type'], "--aggregation", args_dict['aggregation']])
    
    # Use string "False" to match your original script style, even though it causes the bug
    cmd.extend(["--isGrouped", str(args_dict['isGrouped'])])
    
    if args_dict['isGrouped'] and 'group_size' in args_dict:
        cmd.extend(["--group_size", str(args_dict['group_size'])])
    if args_dict['byz_type'] != "no" and 'nbyz' in args_dict:
        cmd.extend(["--nbyz", str(args_dict['nbyz'])])
    
    cmd.extend(["--gpu", "0"]) 
    return " ".join(cmd)

def check_match(expected, actual_meta):
    reasons = []

    # 1. Basic Fields
    if expected['dataset'] != actual_meta.get('dataset'): reasons.append(f"Dataset: {expected['dataset']} vs {actual_meta.get('dataset')}")
    if expected['net'] != actual_meta.get('net'): reasons.append("Net mismatch")
    if expected['byz_type'] != actual_meta.get('byz_type'): reasons.append(f"Attack: {expected['byz_type']} vs {actual_meta.get('byz_type')}")
    if expected['aggregation'] != actual_meta.get('aggregation'): reasons.append(f"Def: {expected['aggregation']} vs {actual_meta.get('aggregation')}")
    
    # 2. Bias (Float comparison)
    if not math.isclose(expected['bias'], float(actual_meta.get('bias', -1)), rel_tol=1e-9):
        reasons.append(f"Bias: {expected['bias']} vs {actual_meta.get('bias')}")

    # 3. Grouping Smart Check (Fixes the "False" string bug)
    exp_grouped = expected['isGrouped']
    act_grouped = actual_meta.get('isGrouped')
    if isinstance(act_grouped, str): act_grouped = (act_grouped.lower() == 'true')
    
    act_gsize = actual_meta.get('group_size', 0)

    # If we expected False, but got True, accept it ONLY IF group_size is 0
    # This handles the case where Python parsed "False" as True, but the logic didn't actually group them.
    if exp_grouped is False and act_grouped is True:
        if act_gsize != 0:
            reasons.append(f"Grouping: Expected False, got True with size {act_gsize}")
    elif exp_grouped != act_grouped:
        reasons.append(f"Grouping: Expected {exp_grouped}, got {act_grouped}")

    # 4. Group Size (Only if we expected grouping)
    if exp_grouped:
        if expected['group_size'] != act_gsize:
            reasons.append(f"GroupSize: {expected['group_size']} vs {act_gsize}")

    # 5. NByz (Ignore if attack is "no")
    if expected['byz_type'] != 'no':
        if expected['nbyz'] != actual_meta.get('nbyz'):
             reasons.append(f"NByz: {expected['nbyz']} vs {actual_meta.get('nbyz')}")

    return (len(reasons) == 0), reasons

def main():
    try:
        with open("results.json", 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("results.json not found.")
        sys.exit(1)

    # Build Expected Configs
    expected_configs = []
    for dataset in datasets:
        for bias in bias_values:
            for model in models:
                for attack in attack_types:
                    for defence in defences:
                        if defence == 'fedavg':
                            base = {"dataset": dataset, "bias": bias, "net": model, "byz_type": attack, "aggregation": defence, "isGrouped": False}
                            if attack == "no": expected_configs.append(base)
                            else: 
                                for n in nbyz_list: 
                                    c = base.copy(); c['nbyz'] = n; expected_configs.append(c)
                        else:
                            for toGroup in isGrouped_list:
                                base = {"dataset": dataset, "bias": bias, "net": model, "byz_type": attack, "aggregation": defence, "isGrouped": toGroup}
                                if toGroup:
                                    for g in group_size_list:
                                        c = base.copy(); c['group_size'] = g
                                        if attack == "no": expected_configs.append(c)
                                        else: 
                                            for n in nbyz_list: 
                                                c2 = c.copy(); c2['nbyz'] = n; expected_configs.append(c2)
                                else:
                                    if attack == "no": expected_configs.append(base)
                                    else: 
                                        for n in nbyz_list: 
                                            c = base.copy(); c['nbyz'] = n; expected_configs.append(c)

    # Check
    failed_commands = []
    matches = 0
    near_misses = [] # To store why things failed

    for config in expected_configs:
        found = False
        best_fail_reason = "No similar entry found"
        
        for entry in data:
            is_match, reasons = check_match(config, entry.get('meta_data', {}))
            if is_match:
                found = True
                matches += 1
                break
            
            # Save the reason for the most relevant-looking entry (same dataset/def/attack)
            meta = entry.get('meta_data', {})
            if (meta.get('dataset') == config['dataset'] and 
                meta.get('aggregation') == config['aggregation'] and 
                meta.get('byz_type') == config['byz_type'] and
                math.isclose(config['bias'], float(meta.get('bias', -1)), rel_tol=1e-9)):
                best_fail_reason = ", ".join(reasons)

        if not found:
            cmd = generate_command(config)
            failed_commands.append((cmd, best_fail_reason))

    print(f"Total Expected: {len(expected_configs)}")
    print(f"Total Found:    {matches}")
    print(f"Missing:        {len(failed_commands)}")
    print("-" * 60)

    if failed_commands:
        print("MISSING EXPERIMENTS & LIKELY REASONS:")
        for cmd, reason in failed_commands:
            print(f"Reason: {reason}")
            print(f"Command: {cmd}")
            print("-" * 20)
    else:
        print("All experiments accounted for! (Any 'isGrouped' discrepancies were auto-resolved).")

if __name__ == "__main__":
    main()
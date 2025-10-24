import subprocess
import sys

#!/usr/bin/env python3
# /m:/PythonTests/newSafeFL/SAFEFL/run_benchmarks.py

# customize these lists as needed
datasets = ["FEMNIST"]
bias_values = [0, 0.1, 0.3, 0.5][::-1]

base_args = [
    "--nworkers", "100",
    "--batch_size", "32",
    "--niter", "500", #2500
    "--lr", "0.05",
    "--test_every", "10", #10
    "--gpu", "1",
    "--aggregation", "fedavg",
    "--net", "resnet18",
]

for dataset in datasets:
    for bias in bias_values:
        cmd = [sys.executable, "main.py"] + base_args + ["--dataset", dataset, "--bias", str(bias)]
        print("Running:", " ".join(cmd))
        try:
            # use check=True to raise on non-zero exit; remove if you want to continue regardless
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed (dataset={dataset}, bias={bias}): returncode={e.returncode}")
            # continue to next combination
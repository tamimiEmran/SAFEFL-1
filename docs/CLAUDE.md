# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAFEFL is an MPC-friendly framework for Private and Robust Federated Learning. It implements various federated learning aggregation rules and attacks, including support for Multi-Party Computation (MPC) through the MP-SPDZ framework.

The project is based on the work of FLTrust authors and has been extended to use PyTorch and support additional aggregation methods and attacks. It includes 13+ aggregation rules, 7+ attack methods, and supports both MNIST and HAR datasets.

## Key Components

1. **Aggregation Rules**: Implementation of various federated learning aggregation rules like FedAvg, Krum, Trimmed Mean, FLTrust, etc. (in `aggregation_rules.py`)
2. **Attack Methods**: Implementation of attacks including Label Flipping, Krum Attack, Scaling Attack, etc. (in `attacks.py`)
3. **Models**: Linear regression model for classification (in `models/lr.py`)
4. **Datasets**: MNIST and HAR datasets, with custom data loading functionality (in `data_loaders.py`)
5. **MPC Integration**: Multi-Party Computation support using MP-SPDZ framework
6. **Visualization Dashboard**: Streamlit-based dashboard for monitoring experiments (in `app.py` and `pages/`)

## Running Commands

### Setup and Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install MPC Framework** (optional, only needed for MPC experiments):
```bash
bash mpc_install.sh
```

3. **Download HAR dataset** (if needed):
```bash
cd data && bash har.sh
```

### Development Commands

- **No testing framework**: This project does not use a formal testing framework
- **No linting/formatting**: No automated linting or code formatting tools configured

### Running Experiments

#### Single Experiment

Run a single federated learning experiment with specific parameters:

```bash
python main.py --dataset MNIST --aggregation fedavg --nworkers 30 --nbyz 6 --byz_type scaling_attack
```

Get help on available parameters:
```bash
python main.py -h
```

#### Batch Experiments

Run multiple experiments with different configurations:

```bash
python runExperiments.py
```

Run hierarchical federated learning experiments:

```bash
python HFL_exp.py
```

Modify these scripts to customize batch parameters before running.

### Visualization Dashboard

Start the visualization dashboard:

```bash
streamlit run app.py
```

## Command Line Parameters

Important parameters for running experiments:

- `--dataset`: Dataset to use (MNIST, HAR)
- `--aggregation`: Aggregation rule (fedavg, krum, trim_mean, fltrust, flame, etc.)
- `--nworkers`: Number of workers/clients
- `--nbyz`: Number of Byzantine/malicious clients
- `--byz_type`: Attack type (no, scaling_attack, label_flipping_attack, etc.)
- `--niter`: Number of iterations/epochs
- `--bias`: Data distribution bias degree (0-1)
- `--nruns`: Number of runs for averaging results
- `--test_every`: Interval for testing the model

HierarchicalFL specific parameters:
- `--n_groups`: Number of groups for hierarchical federated learning
- `--assumed_mal_prct`: Assumed percentage of malicious users

MPC-specific parameters:
- `--mpspdz`: Flag to enable MPC
- `--protocol`: MPC protocol to use
- `--players`: Number of computation parties

## Architecture Overview

The project follows a modular architecture where experiments are orchestrated through `main.py` which coordinates:

1. **Data Pipeline**: `data_loaders.py` handles dataset loading and client data distribution
2. **Model Training**: Linear regression models in `models/` are trained using federated learning
3. **Aggregation Layer**: `aggregation_rules.py` implements 13+ different aggregation algorithms that combine client gradients
4. **Security Layer**: `attacks.py` implements various Byzantine attacks to test robustness
5. **MPC Integration**: `mpspdz/` contains Multi-Party Computation implementations for privacy-preserving aggregation
6. **Visualization**: Streamlit dashboard (`app.py`, `pages/`) provides real-time monitoring

Key architectural decision: All aggregation rules operate on gradients rather than model parameters for consistency and MPC compatibility.

## Project Structure

- `main.py`: Main entry point, coordinates training loop and aggregation
- `runExperiments.py`: Batch experiment runner with multiple configurations
- `HFL_exp.py`: Hierarchical federated learning experiment runner
- `aggregation_rules.py`: 13+ aggregation algorithms (FedAvg, Krum, FLTrust, etc.)
- `attacks.py`: 7+ attack implementations (scaling, label flipping, etc.)
- `data_loaders.py`: Dataset loading and client data distribution
- `models/`: Neural network model definitions (linear regression)
- `app.py`: Streamlit dashboard entry point
- `pages/`: Dashboard pages for different visualizations
- `results/`: Experimental results storage
- `mpspdz/`: MP-SPDZ Multi-Party Computation integration

## Extending the Framework

### Adding a New Aggregation Rule

1. Add implementation in `aggregation_rules.py`
2. Add a case for the aggregation rule in the `main` function of `main.py`

### Adding a New Attack

1. Add implementation in `attacks.py`
2. Add the attack name to the `get_byz` function in `main.py`

### Adding a New Model

1. Add a new file in the `models/` folder
2. Expand the `get_net` function in `main.py`

### Adding a New Dataset

1. Add loading functionality to the `load_data` function in `data_loaders.py`
2. Add dataset dimensions to the `get_shapes` function
3. Extend the `assign_data` function for data distribution
4. For scaling attack support, extend `scaling_attack_insert_backdoor` and `add_backdoor` in `attacks.py`

## MPC Setup

To use the MPC functionality:

1. Install MP-SPDZ using the provided script:
   ```bash
   bash mpc_install.sh
   ```

2. Run experiments with the `--mpspdz` flag and appropriate protocol settings:
   ```bash
   python main.py --mpspdz --protocol semi2k --players 2 --aggregation fedavg
   ```

Supported protocols:
- `semi2k`: 2 or more parties in a semi-honest, dishonest majority setting
- `spdz2k`: 2 or more parties in a malicious, dishonest majority setting
- `replicated2k`: 3 parties in a semi-honest, honest majority setting
- `psReplicated2k`: 3 parties in a malicious, honest majority setting
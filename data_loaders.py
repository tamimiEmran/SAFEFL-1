from __future__ import print_function

import numpy as np
import random
import math
import os

import torch
import torchvision
import torch.utils.data


def get_shapes(dataset):
    """
    Get the input and output shapes of the data examples for each dataset used.
    dataset: name of the dataset used
    """
    if dataset == 'HAR':
        num_inputs = 561
        num_outputs = 6
        num_labels = 6
    elif dataset == 'MNIST':
        num_inputs = 784  # 28x28 pixels
        num_outputs = 10  # 10 digit classes
        num_labels = 10
    elif dataset == 'CIFAR10':
        num_inputs = 3 * 32 * 32  # flattened
        num_outputs = 10
        num_labels = 10
    elif dataset == 'CIFAR100':
        num_inputs = 3 * 32 * 32
        num_outputs = 100
        num_labels = 100
    elif dataset == 'FEMNIST':
        # True FEMNIST has 62 classes (10 digits + 52 letters)
        # Images are 28x28, grayscale
        num_inputs = 28 * 28
        num_outputs = 62
        num_labels = 62
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    return num_inputs, num_outputs, num_labels

def load_data(dataset, seed):
    """
    Load the dataset from the drive.
    The har datasets need to be downloaded first with the provided scripts in /data.
    dataset: name of the dataset
    seed: seed for randomness
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset == 'HAR':
        train_dir = os.path.join("data", "HAR", "train", "")
        test_dir = os.path.join("data", "HAR", "test", "")

        file = open(train_dir + "X_train.txt", 'r')
        X_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(train_dir + "y_train.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "X_test.txt", 'r')
        X_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(test_dir + "y_test.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        # Loading which datapoint belongs to which client
        file = open(train_dir + "subject_train.txt", 'r')
        train_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "subject_test.txt", 'r')
        test_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        y_train, y_test, X_train, X_test = [], [], [], []

        clients = np.concatenate((train_clients, test_clients))
        for client in range(1, 31):
            mask = tuple([clients == client])
            x_client = X[mask]
            y_client = y[mask]

            split = np.concatenate((np.ones(int(np.ceil(0.75*len(y_client))), dtype=bool), np.zeros(int(np.floor(0.25*len(y_client))), dtype=bool)))
            np.random.shuffle(split)  # Generate mask for train test split with ~0.75 1
            x_train_client = x_client[split]
            y_train_client = y_client[split]
            x_test_client = x_client[np.invert(split)]
            y_test_client = y_client[np.invert(split)]

            # Attach vector of client id to training data for data assignment in assign_data()
            x_train_client = np.insert(x_train_client, 0, client, axis=1)
            if len(X_train) == 0:
                X_train = x_train_client
                X_test = x_test_client
                y_test = y_test_client
                y_train = y_train_client
            else:
                X_train = np.append(X_train, x_train_client, axis=0)
                X_test = np.append(X_test, x_test_client, axis=0)
                y_test = np.append(y_test, y_test_client)
                y_train = np.append(y_train, y_train_client)

        tensor_train_X = torch.tensor(X_train, dtype=torch.float32)
        tensor_test_X = torch.tensor(X_test, dtype=torch.float32)
        tensor_train_y = torch.tensor(y_train, dtype=torch.int64) - 1
        tensor_test_y = torch.tensor(y_test, dtype=torch.int64) - 1
        train_dataset = torch.utils.data.TensorDataset(tensor_train_X, tensor_train_y)
        test_dataset = torch.utils.data.TensorDataset(tensor_test_X, tensor_test_y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


    elif dataset == 'MNIST':
        # Define transforms: normalization and conversion to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load MNIST datasets
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True,
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True,
            transform=transform
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=100, 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=100, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        
    elif dataset == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2470, 0.2435, 0.2616)),
            torchvision.transforms.Lambda(lambda t: t.view(-1))  # (3072,)
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2470, 0.2435, 0.2616)),
            torchvision.transforms.Lambda(lambda t: t.view(-1))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, worker_init_fn=seed_worker, generator=g
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, worker_init_fn=seed_worker, generator=g
        )

    elif dataset == 'CIFAR100':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                             (0.2675, 0.2565, 0.2761)),
            torchvision.transforms.Lambda(lambda t: t.view(-1))  # (3072,)
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                             (0.2675, 0.2565, 0.2761)),
            torchvision.transforms.Lambda(lambda t: t.view(-1))
        ])
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, worker_init_fn=seed_worker, generator=g
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, worker_init_fn=seed_worker, generator=g
        )

    elif dataset == 'FEMNIST':
        
        from datasets import load_dataset  # pip install datasets
        ds = load_dataset('flwrlabs/femnist')

        def _to_tensor_split_flat(split):
            imgs = np.stack([np.array(x, dtype=np.uint8) for x in split["image"]], axis=0)  # (N, 28, 28)
            labels = np.array(split["character"], dtype=np.int64)
            timgs = torch.tensor(imgs, dtype=torch.float32) / 255.0        # (N, 28, 28)
            # normalize then flatten
            timgs = (timgs - 0.1307) / 0.3081                               # channelwise for grayscale
            timgs = timgs.view(timgs.size(0), -1)                           # (N, 784)
            tlabels = torch.tensor(labels, dtype=torch.long)
            return torch.utils.data.TensorDataset(timgs, tlabels)

        ds_split = ds["train"].train_test_split(test_size=0.05, seed=seed)
        train_dataset = _to_tensor_split_flat(ds_split["train"])
        test_dataset  = _to_tensor_split_flat(ds_split["test"])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True,
            worker_init_fn=seed_worker, generator=g
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False,
            worker_init_fn=seed_worker, generator=g
        )

    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    return train_loader, test_loader


def _assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1):
    """
    Assign the data to the clients.
    train_data: dataloader of the training dataset
    bias: degree of non-iid between the classes loaded by each client
    device: device used in training and inference
    num_labels: number of classes
    num_workers: number of benign and malicious clients used during training
    server_pc: number of data examples in the server dataset
    p: bias probability in server dataset
    dataset: name of the dataset
    seed: seed for randomness
    """
    other_group_size = (1 - bias) / (num_labels - 1)
    if dataset == "HAR":
        worker_per_group = 30 / num_labels

    elif dataset == "MNIST":
        worker_per_group = num_workers / num_labels

    else:
        raise NotImplementedError

    # assign training data to each worker
    if dataset == "HAR":
        each_worker_data = [[] for _ in range(30)]
        each_worker_label = [[] for _ in range(30)]
    elif dataset == "MNIST":
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
    else:
        raise NotImplementedError
    

    server_data = []
    server_label = []

    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])
    server_counter = [0 for _ in range(num_labels)]

    # compute the labels needed for each class
    if dataset == "HAR":
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            for (x, y) in zip(data, label):
                clientId = int(x[0].item())-1
                x = x[1:len(x)]
                x = x.reshape(1, 561)
                # Assign x and y to appropriate client or server based on method by original code
                if server_counter[int(y.cpu().numpy())] < samp_dis[int(y.cpu().numpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.cpu().numpy())] += 1
                else:
                    each_worker_data[clientId].append(x)
                    each_worker_label[clientId].append(y)

    elif dataset == "MNIST":
        # --- Build exactly num_workers workers; distribute labels evenly ---
        worker_labels = []
        base = num_workers // num_labels
        rem = num_workers % num_labels
        for i in range(num_labels):
            count = base + (1 if i < rem else 0)  # spread the remainder
            for _ in range(count):
                worker_labels.append({'main_label': i, 'bias': bias})

        # Shuffle worker assignments (seeded for reproducibility)
        random.seed(seed)
        random.shuffle(worker_labels)

        # For each data sample
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            for (x, y) in zip(data, label):
                y_int = int(y.item())

                # Check if it should go to server
                if server_counter[y_int] < samp_dis[y_int]:
                    x_flat = x.view(1, 784)  # Flatten 28x28 to 784
                    server_data.append(x_flat)
                    server_label.append(y)
                    server_counter[y_int] += 1
                    continue

                # Otherwise assign to a worker based on the label and bias
                assigned = False
                for worker_idx, worker_pref in enumerate(worker_labels):
                    main_label = worker_pref['main_label']
                    worker_bias = worker_pref['bias']

                    # Prefer main label with 'bias' prob; others with 'other_group_size' prob
                    if (y_int == main_label and random.random() < worker_bias) or \
                       (y_int != main_label and random.random() < other_group_size):
                        x_flat = x.view(1, 784)
                        each_worker_data[worker_idx].append(x_flat)
                        each_worker_label[worker_idx].append(y)
                        assigned = True
                        break

                # Fallback: ensure we never drop a sample
                if not assigned:
                    worker_idx = random.randrange(len(worker_labels))
                    x_flat = x.view(1, 784)
                    each_worker_data[worker_idx].append(x_flat)
                    each_worker_label[worker_idx].append(y)



    else:
       raise NotImplementedError

    # Format server data
    if server_pc != 0:
        if len(server_data) > 0:
            server_data = torch.cat(server_data, dim=0)
            server_label = torch.stack(server_label, dim=0)
        else:
            if dataset == "HAR":
                server_data = torch.empty(size=(0, 561)).to(device)
            elif dataset == "MNIST":
                server_data = torch.empty(size=(0, 784)).to(device)
            else:
                raise NotImplementedError
            server_label = torch.empty(size=(0, )).to(device)
    else:
        if dataset == "HAR":
            server_data = torch.empty(size=(0, 561)).to(device)
        elif dataset == "MNIST":
            server_data = torch.empty(size=(0, 784)).to(device)
        else:
            raise NotImplementedError
        server_label = torch.empty(size=(0, )).to(device)

    # Process worker data
    for i in range(len(each_worker_data)):
        if len(each_worker_data[i]) > 0:
            each_worker_data[i] = torch.cat(each_worker_data[i], dim=0)
            each_worker_label[i] = torch.stack(each_worker_label[i], dim=0)
        else:
            if dataset == "HAR":
                each_worker_data[i] = torch.empty(size=(0, 561)).to(device)
            elif dataset == "MNIST":
                each_worker_data[i] = torch.empty(size=(0, 784)).to(device)
            else:
                raise NotImplementedError
            each_worker_label[i] = torch.empty(size=(0,), dtype=torch.long).to(device)

    # Randomly permute workers if needed
    if dataset == "HAR":
        random_order = np.random.RandomState(seed=seed).permutation(30)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
    elif dataset == "MNIST":
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label


def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1):
    """
    Assign the data to the clients.
    Supports: HAR, MNIST, CIFAR10, CIFAR100, FEMNIST (flattened).
    - Builds exactly `num_workers` workers for non-HAR datasets.
    - Never drops samples: falls back to a random worker if bias rules don't assign.
    - Uses dataset-correct label count internally (ignores the num_labels arg to avoid mismatches).
    """

    # ---- Dataset-specific dims & label counts ----
    if dataset == "HAR":
        D = 561
        num_labels_eff = 6
        total_workers = 30
    elif dataset == "MNIST":
        D = 784
        num_labels_eff = 10
        total_workers = num_workers
    elif dataset == "CIFAR10":
        D = 3072
        num_labels_eff = 10
        total_workers = num_workers
    elif dataset == "CIFAR100":
        D = 3072
        num_labels_eff = 100
        total_workers = num_workers
    elif dataset == "FEMNIST":
        D = 784
        num_labels_eff = 62
        total_workers = num_workers
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    # ---- Bias math (guard single-class edge case) ----
    if num_labels_eff > 1:
        other_group_size = (1 - bias) / (num_labels_eff - 1)
    else:
        other_group_size = 0.0  # degenerate case; not applicable to supported datasets anyway

    # ---- Init per-worker containers ----
    each_worker_data = [[] for _ in range(total_workers)]
    each_worker_label = [[] for _ in range(total_workers)]

    # ---- Server buffers ----
    server_data, server_label = [], []

    # ---- Server class distribution samp_dis (biased toward class 1 by fraction p) ----
    samp_dis = [0 for _ in range(num_labels_eff)]
    num1 = int(server_pc * p) if num_labels_eff > 1 else 0
    if num_labels_eff > 1:
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels_eff - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.0
        for other_num in range(num_labels_eff - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels_eff - 1] = server_pc - np.sum(samp_dis[:num_labels_eff - 1])
    else:
        # Single-class: everything is class 0
        samp_dis[0] = server_pc

    server_counter = [0 for _ in range(num_labels_eff)]

    # =========================
    # HAR path (original logic)
    # =========================
    if dataset == "HAR":
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            for (x, y) in zip(data, label):
                clientId = int(x[0].item()) - 1
                x = x[1:len(x)].reshape(1, D)  # (1,561)
                y_idx = int(y.item())

                if server_counter[y_idx] < samp_dis[y_idx]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[y_idx] += 1
                else:
                    each_worker_data[clientId].append(x)
                    each_worker_label[clientId].append(y)


    # ==========================================================
    # Unified path for MNIST / CIFAR10 / CIFAR100 / FEMNIST
    # ==========================================================
    else:
        # Build exactly total_workers workers with even main-label distribution
        worker_labels = []
        base = total_workers // num_labels_eff
        rem = total_workers % num_labels_eff
        for i in range(num_labels_eff):
            count = base + (1 if i < rem else 0)
            for _ in range(count):
                worker_labels.append({'main_label': i, 'bias': bias})

        # Precompute candidate pools per class
        workers_by_label = {c: [] for c in range(num_labels_eff)}
        for idx, wp in enumerate(worker_labels):
            workers_by_label[wp['main_label']].append(idx)

        all_indices = list(range(len(worker_labels)))

        # Track per-worker loads to balance assignments
        worker_loads = [0] * len(worker_labels)

        rng = random.Random(seed)  # local RNG
        # No need to keep a fixed iteration order anymore

        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            for (x, y) in zip(data, label):
                y_idx = int(y.item())

                # Send to server if quota not filled
                if server_counter[y_idx] < samp_dis[y_idx]:
                    x_flat = x.view(1, D)  # defensive flatten
                    server_data.append(x_flat)
                    server_label.append(y)
                    server_counter[y_idx] += 1
                    continue

                # Choose candidate pool once per sample
                if num_labels_eff > 1 and rng.random() < bias:
                    candidates = workers_by_label.get(y_idx, [])
                else:
                    # all workers whose main_label != y_idx
                    candidates = [w for w in all_indices if worker_labels[w]['main_label'] != y_idx]



                # Pick the least-loaded candidate (tie-break randomly)
                min_load = min(worker_loads[w] for w in candidates)
                least_loaded = [w for w in candidates if worker_loads[w] == min_load]
                worker_idx = rng.choice(least_loaded)

                x_flat = x.view(1, D)
                each_worker_data[worker_idx].append(x_flat)
                each_worker_label[worker_idx].append(y)
                worker_loads[worker_idx] += 1

    # ---- Format server tensors ----
    if server_pc != 0:
        if len(server_data) > 0:
            server_data = torch.cat(server_data, dim=0)                        # (S, D)
            server_label = torch.stack(server_label, dim=0)                    # (S,)
        else:
            server_data = torch.empty(size=(0, D), dtype=torch.float32).to(device)
            server_label = torch.empty(size=(0,), dtype=torch.long).to(device)
    else:
        server_data = torch.empty(size=(0, D), dtype=torch.float32).to(device)
        server_label = torch.empty(size=(0,), dtype=torch.long).to(device)

    # ---- Format worker tensors ----
    for i in range(len(each_worker_data)):
        if len(each_worker_data[i]) > 0:
            each_worker_data[i] = torch.cat(each_worker_data[i], dim=0)        # (Ni, D)
            each_worker_label[i] = torch.stack(each_worker_label[i], dim=0)    # (Ni,)
        else:
            each_worker_data[i] = torch.empty(size=(0, D), dtype=torch.float32).to(device)
            each_worker_label[i] = torch.empty(size=(0,), dtype=torch.long).to(device)

    # ---- Randomly permute workers (consistency with original behavior) ----
    random_order = np.random.RandomState(seed=seed).permutation(total_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label


def plot_iidness(each_worker_label, dataset="CIFAR10", figsize=(10, 6), top_n_workers=50, show=True):
    """Visualize the level of IID-ness across workers.

    - each_worker_label: list of tensors (Ni,) per worker containing integer labels
    - dataset: used to infer num_classes when needed
    - figsize: matplotlib figure size
    - top_n_workers: when there are many workers, plot only the top N by sample count
    - show: whether to call plt.show() (set False for headless runs)

    Produces:
    - heatmap of normalized class histograms per worker (workers on y-axis, classes on x-axis)
    - prints summary statistics: mean KL divergence to global distribution, per-worker entropy mean/std
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import numpy as np
    from scipy.stats import entropy

    # infer number of classes from dataset if possible
    dataset_map = {"CIFAR10": 10, "CIFAR100": 100, "MNIST": 10, "FEMNIST": 62, "HAR": 6}
    num_classes = dataset_map.get(dataset.upper(), None)
    if num_classes is None:
        # fallback: infer from labels
        uniq = set()
        for l in each_worker_label:
            if isinstance(l, torch.Tensor):
                uniq.update(l.cpu().numpy().tolist())
            else:
                uniq.update(list(l))
        num_classes = int(max(uniq)) + 1

    # Build per-worker histograms
    worker_counts = []
    for l in each_worker_label:
        if isinstance(l, torch.Tensor):
            labels = l.cpu().numpy().astype(int)
        else:
            labels = np.array(l, dtype=int)
        if labels.size == 0:
            counts = np.zeros(num_classes, dtype=float)
        else:
            counts = np.bincount(labels, minlength=num_classes).astype(float)
        worker_counts.append(counts)

    worker_counts = np.stack(worker_counts, axis=0)  # (W, C)

    # Optionally reduce to top_n_workers by sample count for visualization
    worker_totals = worker_counts.sum(axis=1)
    if worker_counts.shape[0] > top_n_workers:
        top_idx = np.argsort(-worker_totals)[:top_n_workers]
        heat_counts = worker_counts[top_idx]
        worker_ids = top_idx
    else:
        heat_counts = worker_counts
        worker_ids = np.arange(worker_counts.shape[0])

    # Normalize rows to probabilities (avoid division by zero)
    row_sums = heat_counts.sum(axis=1, keepdims=True)
    probs = np.divide(heat_counts, row_sums, where=row_sums > 0)
    probs[row_sums.squeeze() == 0] = 0.0

    # Global distribution (over all workers)
    global_counts = worker_counts.sum(axis=0)
    global_probs = global_counts / (global_counts.sum() + 1e-12)

    # Compute per-worker KL divergence to global and entropies
    kl_divs = []
    entropies = []
    for p in probs:
        # smoothing to avoid zeros in KL
        p_s = p + 1e-12
        g_s = global_probs + 1e-12
        kl_divs.append(entropy(p_s, qk=g_s))
        entropies.append(entropy(p_s))

    kl_divs = np.array(kl_divs)
    entropies = np.array(entropies)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(probs, cmap='viridis', cbar=True)
    plt.xlabel('Class')
    plt.ylabel('Worker (truncated)')
    plt.title(f'Per-worker class distribution, a uniform distribution is equal to {1/num_classes:.3f} per class')

    # Add small textual summary below the plot
    summary = (
        f'Workers plotted: {probs.shape[0]} / {worker_counts.shape[0]} | '
        f'Mean KL->global: {kl_divs.mean():.4f} | Std KL: {kl_divs.std():.4f} | '
        f'Mean entropy: {entropies.mean():.4f} | Std entropy: {entropies.std():.4f}'
    )
    plt.gcf().text(0.01, -0.05, summary, fontsize=10)

    if show:
        plt.show()
    else:
        plt.close()

    # Also return the computed metrics and arrays for programmatic use
    return {
        'worker_probs': probs,
        'global_probs': global_probs,
        'kl_divs': kl_divs,
        'entropies': entropies,
        'worker_totals': worker_totals,
        'worker_ids': worker_ids,
    }


def show_dataloader_image(dataloader, dataset=None, index=0, unnormalize=True):
    """Display a single image+label from a DataLoader.

    Arguments:
    - dataloader: torch.utils.data.DataLoader
    - dataset: optional dataset name string ('CIFAR10','CIFAR100','MNIST','FEMNIST') to map label names and unnormalize correctly
    - index: which sample within the first batch to show (default 0)
    - unnormalize: if True, attempt to undo the Normalize transform for display

    Returns: (image_np, label_idx, label_name)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # label name maps
    cifar10_classes = [
        'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
    ]
    cifar100_classes = ['apple',
                        'aquarium_fish',
                        'baby',
                        'bear',
                        'beaver',
                        'bed',
                        'bee',
                        'beetle',
                        'bicycle',
                        'bottle',
                        'bowl',
                        'boy',
                        'bridge',
                        'bus',
                        'butterfly',
                        'camel',
                        'can',
                        'castle',
                        'caterpillar',
                        'cattle',
                        'chair',
                        'chimpanzee',
                        'clock',
                        'cloud',
                        'cockroach',
                        'couch',
                        'crab',
                        'crocodile',
                        'cup',
                        'dinosaur',
                        'dolphin',
                        'elephant',
                        'flatfish',
                        'forest',
                        'fox',
                        'girl',
                        'hamster',
                        'house',
                        'kangaroo',
                        'keyboard',
                        'lamp',
                        'lawn_mower',
                        'leopard',
                        'lion',
                        'lizard',
                        'lobster',
                        'man',
                        'maple_tree',
                        'motorcycle',
                        'mountain',
                        'mouse',
                        'mushroom',
                        'oak_tree',
                        'orange',
                        'orchid',
                        'otter',
                        'palm_tree',
                        'pear',
                        'pickup_truck',
                        'pine_tree',
                        'plain',
                        'plate',
                        'poppy',
                        'porcupine',
                        'possum',
                        'rabbit',
                        'raccoon',
                        'ray',
                        'road',
                        'rocket',
                        'rose',
                        'sea',
                        'seal',
                        'shark',
                        'shrew',
                        'skunk',
                        'skyscraper',
                        'snail',
                        'snake',
                        'spider',
                        'squirrel',
                        'streetcar',
                        'sunflower',
                        'sweet_pepper',
                        'table',
                        'tank',
                        'telephone',
                        'television',
                        'tiger',
                        'tractor',
                        'train',
                        'trout',
                        'tulip',
                        'turtle',
                        'wardrobe',
                        'whale',
                        'willow_tree',
                        'wolf',
                        'woman',
                        'worm']
    # FEMNIST mapping: 0-9 -> '0'..'9', 10-35 -> 'A'..'Z', 36-61 -> 'a'..'z'
    femnist_classes = [str(d) for d in range(10)] + [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]

    # get one batch
    it = iter(dataloader)
    batch = next(it)
    imgs, labels = batch[0], batch[1]
    # ensure on cpu for numpy conversion
    imgs = imgs.to('cpu')
    labels = labels.to('cpu')

    # select sample
    img = imgs[index]
    label_idx = int(labels[index].item())

    # If the loader flattened images (common in this repo), try to reshape
    if img.dim() == 1:
        # try common shapes
        if dataset and dataset.upper() in ('CIFAR10', 'CIFAR100'):
            img = img.view(3, 32, 32)
        else:
            # assume grayscale 28x28
            img = img.view(1, 28, 28)

    # Undo normalization if requested
    img_np = img.numpy()
    if unnormalize and dataset is not None:
        ds = dataset.upper()
        if ds == 'CIFAR10':
            mean = np.array([0.4914, 0.4822, 0.4465])[:, None, None]
            std = np.array([0.2470, 0.2435, 0.2616])[:, None, None]
            img_np = img_np * std + mean
        elif ds == 'CIFAR100':
            mean = np.array([0.5071, 0.4867, 0.4408])[:, None, None]
            std = np.array([0.2675, 0.2565, 0.2761])[:, None, None]
            img_np = img_np * std + mean
        elif ds in ('MNIST', 'FEMNIST'):
            mean = 0.1307
            std = 0.3081
            img_np = img_np * std + mean

    # Convert to HWC for display
    if img_np.shape[0] == 3:
        disp = np.transpose(img_np, (1, 2, 0))
    else:
        disp = img_np.squeeze()

    # Clip to [0,1]
    disp = np.clip(disp, 0.0, 1.0)

    # Determine label name
    label_name = str(label_idx)
    if dataset is not None:
        ds = dataset.upper()
        if ds == 'CIFAR10':
            label_name = cifar10_classes[label_idx]
        elif ds == 'CIFAR100':
            if 0 <= label_idx < len(cifar100_classes):
                label_name = cifar100_classes[label_idx]
            else:
                label_name = str(label_idx)
        elif ds == 'FEMNIST':
            if 0 <= label_idx < len(femnist_classes):
                label_name = femnist_classes[label_idx]
            else:
                label_name = str(label_idx)
        elif ds == 'MNIST':
            label_name = str(label_idx)

    # Plot
    plt.figure(figsize=(4, 4))
    if disp.ndim == 2:
        plt.imshow(disp, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(disp)
    plt.axis('off')
    plt.title(f'{dataset or "dataset"} label: {label_name} (idx {label_idx})')
    plt.show()

    return label_idx, label_name



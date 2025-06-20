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
    else:
        raise NotImplementedError
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
        


    else:
        raise NotImplementedError
    return train_loader, test_loader


def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1):
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
        # Create a biased distribution for workers
        worker_labels = []
        for i in range(num_labels):
            label_workers = int(worker_per_group)
            for _ in range(label_workers):
                worker_labels.append({
                    'main_label': i,
                    'bias': bias
                })
        
        # Shuffle worker assignments
        random.seed(seed)
        random.shuffle(worker_labels)
        
        # For each data sample
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)
            
            for (x, y) in zip(data, label):
                y_int = y.item()
                
                # Check if it should go to server
                if server_counter[y_int] < samp_dis[y_int]:
                    x_flat = x.view(1, 784)  # Flatten 28x28 to 784
                    server_data.append(x_flat)
                    server_label.append(y)
                    server_counter[y_int] += 1
                    continue
                
                # Otherwise assign to a worker based on the label and bias
                for worker_idx, worker_pref in enumerate(worker_labels):
                    main_label = worker_pref['main_label']
                    worker_bias = worker_pref['bias']
                    
                    # If this is the worker's main label, higher chance to get it
                    if y_int == main_label and random.random() < worker_bias:
                        x_flat = x.view(1, 784)  # Flatten 28x28 to 784
                        each_worker_data[worker_idx].append(x_flat)
                        each_worker_label[worker_idx].append(y)
                        break
                    # For other labels, lower chance based on other_group_size
                    elif y_int != main_label and random.random() < other_group_size:
                        x_flat = x.view(1, 784)  # Flatten 28x28 to 784
                        each_worker_data[worker_idx].append(x_flat)
                        each_worker_label[worker_idx].append(y)
                        break

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
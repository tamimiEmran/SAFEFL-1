import attacks

import math
import os
import collections
from functools import reduce

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import hdbscan
import copy
import utils
import heirichalFL as hfl

# Copyright (c) 2015, Leland McInnes
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Available at: https://github.com/scikit-learn-contrib/hdbscan
import torch
import numpy as np
from functools import reduce

def divide_and_conquer_bb_probs(gradients, net, lr, f, byz, device,
                                niters, c, b,
                                alpha=1.0, beta=1.0):
    """
    Divide-and-conquer aggregation + Beta-Binomial maliciousness probabilities.

    Returns:
      p_malicious: list of length n_clients with values in [0,1]
    """
    # 1) flatten gradients
    param_list = [torch.cat([xx.reshape(-1) for xx in x])[:, None]
                  for x in gradients]
    # let the malicious clients perform the byzantine attack
    param_list = byz(param_list, net, lr, f, device)
    
    n = len(param_list)
    survive_counts = np.zeros(n, dtype=int)
    
    D = param_list[0].shape[0]
    
    # 2) collect survive counts over random subspace tests
    for _ in range(niters):
        # random dimension d in [1, b)
        d = np.random.randint(1, high=b)
        # random mask of size D picking d coords
        idx = torch.randperm(D, device=device)[:d]
        sel = [p[idx] for p in param_list]            # each is (d,1)
        
        stacked = torch.cat(sel, dim=1)               # (d, n)
        mean   = stacked.mean(dim=1, keepdim=True)    # (d,1)
        centered = (stacked - mean).T                 # (n, d)
        
        # top right singular vector
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        top_vec = Vt[0]                               # (d,)
        
        # outlier scores = squared projection
        scores = (centered @ top_vec).cpu().numpy()**2
        
        # pick the lowest (n - f*c) scores as "good"
        k_good = n - int(f * c)
        good_idx = np.argsort(scores)[:k_good]
        
        # count survivals
        survive_counts[good_idx] += 1
    
    # 3) Beta–Binomial posterior mean for benign probability
    #    theta_i ~ Beta(alpha + k_i, beta + niters - k_i)
    #    E[theta_i] = (alpha + k_i) / (alpha + beta + niters)
    denom = (alpha + beta + niters)
    theta_hat = (alpha + survive_counts) / denom     # shape (n,)
    p_malicious = (1.0 - theta_hat).tolist()

    
    return p_malicious


def fltrust(gradients, net, lr, f, byz, device):
    """
    Based on the description in https://arxiv.org/abs/2012.13995
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f, device)
    n = len(param_list) - 1

    # use the last gradient (server update) as the trusted source
    baseline = param_list[-1].squeeze()
    sim = []
    new_param_list = []

    # compute similarity
    for each_param_list in param_list:
        each_param_array = each_param_list.squeeze()

        sim.append(torch.dot(baseline, each_param_array) / (torch.norm(baseline) + 1e-9) / (
                    torch.norm(each_param_array) + 1e-9))

    sim = torch.stack(sim)[:-1]

    # clip similarities and get trust scores
    sim = F.relu(sim)
    normalized_weights = sim / (torch.sum(sim).item() + 1e-9)

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(
            param_list[i] * normalized_weights[i] / (torch.norm(param_list[i]) + 1e-9) * torch.norm(baseline))

    # compute global update
    global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    # update global model
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def fedavg(gradients, net, lr, f, byz, device, data_sizes):
    """
    Based on the description in https://arxiv.org/abs/1602.05629
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    data_size: amount of training data of each worker device
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)

    n = len(param_list)
    total_data_size = sum(data_sizes)

    # compute global model update
    global_update = torch.zeros(param_list[0].size()).to(device)
    for i, grad in enumerate(param_list):
        global_update += grad * data_sizes[i]
    global_update /= total_data_size

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def krum(gradients, net, lr, f, byz, device):
    """
    Based on the description in https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    # compute pairwise Euclidean distance
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(param_list[i] - param_list[j])
            dist[i, j], dist[j, i] = d, d

    # sort distances and get model with smallest sum of distances to closest n-f-2 models
    sorted_dist, _ = torch.sort(dist, dim=-1)
    global_update = param_list[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]

    # update global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def trim_mean(gradients, net, lr, f, byz, device):
    """
    Based on the description in https://arxiv.org/abs/1803.01498
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    # trim f biggest and smallest values of gradients
    sorted, _ = torch.sort(torch.cat(param_list, dim=1), dim=-1)
    global_update = torch.mean(sorted[:, f:(n - f)], dim=-1)

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def median(gradients, net, lr, f, byz, device):
    """
    Based on the description in https://arxiv.org/abs/1803.01498
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    # compute median of gradients
    global_update, _ = torch.median(torch.cat(param_list, dim=1), dim=-1)

    # update global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def flame(gradients, net, lr, f, byz, device, epsilon, delta):
    """
    Based on the description in https://arxiv.org/abs/2101.02281
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    epsilon: parameter for differential privacy
    delta: parameter for differential privacy
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    # compute pairwise cosine distances
    cos_dist = torch.zeros((n, n), dtype=torch.double).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - F.cosine_similarity(param_list[i], param_list[j], dim=0, eps=1e-9)
            cos_dist[i, j], cos_dist[j, i] = d, d

    # clustering of gradients
    np_cos_dist = cos_dist.cpu().numpy()
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=1, min_cluster_size=(n // 2) + 1,
                                cluster_selection_epsilon=0.0, allow_single_cluster=True).fit(np_cos_dist)

    # compute clipping bound
    euclid_dist = []
    for grad in param_list:
        euclid_dist.append(torch.norm(lr * grad, p=2))

    clipping_bound, _ = torch.median(torch.stack(euclid_dist).reshape((-1, 1)), dim=0)

    # gradient clipping
    clipped_gradients = []
    for i in range(n):
        if clusterer.labels_[i] == 0:
            gamma = clipping_bound / euclid_dist[i]
            clipped_gradients.append(-lr * param_list[i] * torch.min(torch.ones((1,)).to(device), gamma))

    # aggregation
    global_update = torch.mean(torch.cat(clipped_gradients, dim=1), dim=-1)

    # adaptive noise
    std = (clipping_bound * np.sqrt(2 * np.log(1.25 / delta)) / epsilon) ** 2
    global_update += torch.normal(mean=0, std=std.item(), size=tuple(global_update.size())).to(device)

    # update the global model
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())))
        idx += torch.numel(param)


def shieldfl(gradients, net, lr, f, byz, device, previous_gloabl_gradient, iteration, previous_gradients):
    
    """
    Based on the description in https://ieeexplore.ieee.org/document/9762272
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    previous_global_gradient: global model updated of the previous
    iteration: iteration of training process
    previous_gradient: local model updates of previous iteration
    """

    kappa = 0  # the paper gave no indication on how to set this parameter

    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    copy_params = [param.clone() for param in param_list]

    # gradient normalization
    for i in range(f, n):  # benign workers always normalize their gradients
        if iteration == 0:  # emulate selective SGD
            over_threshold = (param_list[i] >= kappa)
        else:
            over_threshold = torch.logical_or((param_list[i] >= kappa),
                                              ((param_list[i] - previous_gradients[i]) >= kappa))
        param_list[i] = over_threshold * param_list[i]

        max_value = torch.max(param_list[i])
        min_value = torch.min(param_list[i])
        param_list[i] = (param_list[i] - min_value) / (max_value - min_value)  # normalize to [0, 1]
        param_list[i] = param_list[i] / torch.norm(param_list[i], p=2.0)  # normalize with Euclidean norm

    for i in range(0, f):  # ASSUMPTION: byzantine workers know that ShieldFL is used and normalize their gradients
        max_value = torch.max(param_list[i])
        min_value = torch.min(param_list[i])
        param_list[i] = (param_list[i] - min_value) / (max_value - min_value)  # normalize to [0, 1]
        param_list[i] = param_list[i] / torch.norm(param_list[i], p=2.0)  # normalize with Euclidean norm

    if (iteration == 0):  # if there is no prev gradient
        previous_gloabl_gradient = torch.mean(torch.cat(param_list, dim=1), dim=-1, keepdim=True)

    checked_gradients = []
    cos_sim = []
    for param in param_list:  # check if gradients are normalized
        sum = torch.sum(torch.square(param))  # emulates secure judgement
        if (math.isclose(sum.item(), 1.0, rel_tol=1e-05, abs_tol=1e-08)):
            checked_gradients.append(param)
            cos_sim.append(F.cosine_similarity(param, previous_gloabl_gradient, dim=0,
                                               eps=1e-9))  # emulates secure cosine similarity

    poison_baseline = checked_gradients[torch.argmin(torch.stack(cos_sim))]  # find poison baseline gradient

    cos_sim_poison = []
    for grad in checked_gradients:  # compute cos_sim to poison baseline gradient
        cos_sim_poison.append(F.cosine_similarity(grad, poison_baseline, dim=0, eps=1e-9))

    cos_sim_poison = torch.stack(cos_sim_poison)
    confidence = torch.ones(cos_sim_poison.size()).to(device) - cos_sim_poison
    normalized_conf = F.normalize(confidence, p=1, dim=0,
                                  eps=1e-12)  # normalized confidence passed on cos_sim to poison baseline gradient

    # compute global update
    global_update = torch.zeros(checked_gradients[0].size()).to(device)
    for i, grad in enumerate(checked_gradients):  # aggregate all gradients
        global_update += grad * normalized_conf[i]

    # update the global model
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)

    return global_update, copy_params  # return gradients of gobal model and local models for next iteration of iteration


def flod(gradients, net, lr, f, byz, device, threshold):
    """
    Based on the description in https://eprint.iacr.org/2021/993
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    threshold: parameter for clipping weights
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f, device)
    n = len(param_list) - 1

    # sgn encoding
    sgn_param_list = []
    for param in param_list:
        sgn_param_list.append(torch.sign(param))

    # boolean encoding
    bool_param_list = []
    for param in sgn_param_list:
        bool_param_list.append(param == 1)

    # hamming distance
    hd = []
    baseline = bool_param_list[-1]
    for i in range(n):
        hd.append(torch.sum(torch.bitwise_xor(bool_param_list[i], baseline)))

    # tau-clipping
    weight = [F.relu(threshold - hd_i) for hd_i in hd]

    # compute global update
    global_update = torch.zeros(sgn_param_list[0].size()).to(device)
    for i in range(n):
        global_update += weight[i] * sgn_param_list[i]

    weight_sum = torch.sum(torch.stack(weight))
    if weight_sum > 0:
        global_update /= weight_sum

    # update the global model
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def divide_and_conquer(gradients, net, lr, f, byz, device, niters, c, b):
    """
    The divide and conquer aggregation rule defined in https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    threshold: parameter for clipping weights
    niters: number of iterations to compute good sets
    c: filtering fraction, percentage of number of malicious clients filtered
    b: dimension of subsamples must be smaller, then the dimension of the gradients
    """

    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f, device)

    good_set = list()

    for i in range(niters):
        random_dimension = np.random.randint(1, high=b, dtype=int)
        r_mask = torch.tensor([True if i < random_dimension else False for i in range(len(param_list[0]))])
        r_mask = r_mask[torch.randperm(len(param_list[0]))].to(device)[:, None]  # craft random selection of random number of parameters
        selected_gradients = [torch.masked_select(param_list[i], r_mask)[:, None] for i in range(len(param_list))]
        mean = torch.mean(torch.cat(selected_gradients, dim=-1), dim=-1)[:, None]
        selected_gradients = torch.sub(torch.cat(selected_gradients, dim=-1), mean).T  # center gradients and
        # transpose to match dimension to their implementation

        _, _, rightSingular = torch.linalg.svd(selected_gradients, full_matrices=False)  # compute top right singular eigenvector
        topeigen = rightSingular[0, :]  # rows of v are ordered right singular vectors
        outlier_score = [torch.dot(selected_gradients[i], topeigen).item()**2 for i in range(len(param_list))]
        # this is my assumption because their algorithm would compute the dot product of full gradient with topeigen
        # which would have different dimensions and therefore can not be computed
        sorted_indices = np.argsort(outlier_score)[0:int(len(param_list)-f*c)]  # this assumes that the aggregation knows
        # the actual number of malicious clients
        good_set.append(sorted_indices)

    good_indices = reduce(np.intersect1d, good_set)
    if len(good_indices) == 0:
        print("No good gradients found this round. Consider lowering c or niters")
        return  # No update this round

    x = torch.cat([param_list[i] for i in good_indices], dim=-1)
    global_update = torch.mean(x, dim=-1)

    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def mpspdz_aggregation(gradients, net, lr, f, byz, device, param_num, port, chunk_size, parties):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    param_num: number of parameters per gradient
    port: port computation parties are listing on
    chunk_size: amount of values submitted at one time
    parties: number of computation parties
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    param_list_python = torch.reshape(torch.cat(param_list, dim=0), (-1,)).tolist()  # convert tensors to list

    import mpspdz.ExternalIO.mpc_client as m
    os.chdir("mpspdz")
    output = m.client(0, parties, port, param_num, n, chunk_size, param_list_python, precision=12)
    os.chdir("..")

    global_update = torch.tensor(output).to(device)  # convert python list to tensor

    # update global model
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def foolsgold(gradients, net, lr, f, byz, device, gradient_history):
    """
    Based on the description in https://arxiv.org/abs/1808.04866
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    gradient_history: aggregation of previous gradients per worker.
    """

    # reference implementation: https://github.com/DistributedML/FoolsGold
    # FoolsGold has individual learning rates. Global learning rate lr has no effect

    kappa = 1
    eps = 10e-5

    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)

    n = len(param_list)

    for i in range(n):
        norm = torch.norm(param_list[i])
        if(norm > 1):
            param_list[i] /= norm

    # updates history
    gradient_history = [gradient_history[i] + param_list[i] for i in range(n)]

    cos_dist = torch.zeros((n, n), dtype=torch.double).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            d = F.cosine_similarity(gradient_history[i], gradient_history[j], dim=0, eps=1e-9)
            cos_dist[i, j], cos_dist[j, i] = d, d

    v, _ = torch.max(cos_dist, dim=1)

    # pardoning
    for i in range(n):
        for j in range(n):
            if v[j] > v[i]:
                cos_dist[i][j] *= v[i]/v[j]

    alpha = torch.clamp(1 - torch.max(cos_dist, dim=1)[0], min=0, max=1)

    # logit function
    alpha /= torch.max(alpha, dim=0, keepdim=True)[0]
    alpha = kappa * (torch.logit(alpha, eps=eps) + 0.5)
    alpha = torch.clamp(alpha, min=0, max=1)

    # calculate global update
    global_update = torch.zeros(param_list[0].size()).to(device)
    for i, grad in enumerate(param_list):
        global_update += grad * alpha[i]

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-1)        # FoolsGold uses individual learning rates
        idx += torch.numel(param)

    return gradient_history


def contra(gradients, net, lr, f, byz, device, gradient_history, reputation, cos_dist, C=1):
    """
    Based on the description in https://par.nsf.gov/servlets/purl/10294585
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    gradient_history: aggregation of previous gradients per worker.
    reputation: reputation of each worker.
    cos_dist: pairwise cosine similarity
    C: fraction of clients to select
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)

    n = len(param_list)

    # parameters
    lambda_reputation = C * (C - 1)     # lambda to adjust probability
    J = int(C * n)                      # no. of clients selected in each round
    k = n - f - 1                       # top-k cosine similarities
    big_delta = 0.1                     # defaults: 0.1 for image classification, 0.05 loan dataset
    t = 0                               # some arbitrary threshold with no information on how to set it in paper
    eps = 1e-5

    probability = C + lambda_reputation * reputation
    selected_clients = torch.topk(probability, J)[1]    # only needing indices, not actual values

    # updates history
    gradient_history = [(gradient_history[i] + param_list[i] if (i in selected_clients) else gradient_history[i]) for i in range(n)]

    # compute pairwise cosine similarity
    for i in range(n):
        if not i in selected_clients:   # skip not selected clients
            continue
        for j in range(i + 1, n):
            d = F.cosine_similarity(gradient_history[i], gradient_history[j], dim=0, eps=1e-9)
            cos_dist[i, j], cos_dist[j, i] = d, d

    # compute alignment level
    tau = torch.mean(torch.topk(cos_dist, k, dim=1)[0], dim=1).to(device)
    reputation = torch.where(tau > t, reputation + big_delta, reputation - big_delta)        # reweighting reputation

    # re-weighting cosine similarity
    for i in range(n):
        for j in range(n):
            if tau[j] > tau[i]:
                cos_dist[i][j] *= tau[i]/tau[j]

    lr_m = 1 - tau
    reputation /= torch.max(reputation, dim=0, keepdim=True)[0]     # re-weight to [0, 1]

    lr_m /= torch.max(lr_m, dim=0, keepdim=True)[0]     # re-weight to [0, 1]
    lr_m = torch.clamp(torch.logit(lr_m, eps=eps) + 0.5, min=0, max=1)      # clamping just like in FoolsGold, no information if that is correct

    # compute global model update
    global_update = torch.zeros(param_list[0].size()).to(device)
    for i, grad in enumerate(param_list):
        global_update += grad * lr_m[i]

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)

    return gradient_history, reputation, cos_dist

def signguard(gradients, net, lr, f, byz, device, seed):
    """
    Based on the description in https://arxiv.org/abs/2109.05872
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    seed: seed for randomness
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    num_params = param_list[0].size(0)
    selection_fraction = 0.1

    # lower and upper bound L,R for gradient norm
    L = 0.1
    R = 3.0
    S1 = []
    S2 = []

    # compute l2-norm
    l2_norm = torch.stack([torch.norm(g.flatten(), p=2.0) for g in param_list])

    # compute element wise sign
    num_selection = int(num_params * selection_fraction)
    perm = torch.randperm(num_params)
    idx = perm[:num_selection]
    sign_grads = [torch.sign(g[idx]) for g in param_list]

    # norm-threshold filtering
    M = torch.median(l2_norm)
    for i in range(n):
        if L <= l2_norm[i] / M and l2_norm[i] / M <= R:
            S1.append(i)

    # compute sign statistics
    sign_pos = torch.stack([grad.eq(1.0).float().mean() for grad in sign_grads])
    sign_zero = torch.stack([grad.eq(0.0).float().mean() for grad in sign_grads])
    sign_neg = torch.stack([grad.eq(-1.0).float().mean() for grad in sign_grads])

    # sign-based clustering
    sign_feat = torch.stack([sign_pos, sign_zero, sign_neg], dim=1).detach().cpu().numpy()
    cluster = KMeans(n_clusters=2, max_iter=20, random_state=seed)
    labels = cluster.fit_predict(sign_feat)

    labels_tensor = torch.from_numpy(labels).to(device)
    count = torch.bincount(labels_tensor)
    largest_cluster = torch.argmax(count)

    for i, value in enumerate(labels_tensor):
        if value == largest_cluster:
            S2.append(i)

    # compute intersection of S1 and S2
    S = [i for i in S1 if i in S2]

    # global update
    global_update = torch.zeros(param_list[0].size()).to(device)
    for i in S:
        global_update += param_list[i]
    global_update *= 1 / len(S)

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def flare(gradients, net, lr, f, byz, device, server_data):
    """
    Based on the description in https://dl.acm.org/doi/10.1145/3488932.3517395
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """
    grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]

    # let the malicious clients (first f clients) perform the byzantine attack
    grad_list = byz(grad_list, net, lr, f, device)

    nclients = len(grad_list)

    # Create models for each client to determine penultimate layer representation PLR of auxiliary data (server dataset)
    plrs = []
    for client in range(nclients):
        localmodel = copy.deepcopy(net)
        for j, param in enumerate(localmodel.parameters()):
            param.add_(gradients[client][j], alpha=-1)  # alpha -1 for gradient descent
        localmodel.eval()  # activate eval mode
        modelPLR = localmodel(server_data)
        plrs.append(modelPLR)

    # compute maximum mean discrepancy MMD between PLRs of clients
    mmds = torch.zeros((nclients, nclients))
    for client in range(nclients):
        for otherClient in range(client, nclients):
            mmds[otherClient][client] = mmds[client][otherClient] = utils.MMD(plrs[client], plrs[otherClient], device)

    # get k nearest neighbors to each client based on MMD
    k = round(0.5 * nclients)
    neighbors = torch.zeros(nclients, k, dtype=torch.int)

    for client in range(nclients):
        neighbors[client] = torch.argsort(mmds[client])[0:k]

    # count times client is selected as neighbor
    counts = torch.zeros(nclients, dtype=torch.int)
    for row in neighbors:
        for value in row:
            counts[value.item()] += 1

    counts = torch.exp(counts)
    sumCounts = torch.sum(counts)

    # compute global update
    new_param_list = []
    for i in range(nclients):
        new_param_list.append(
           grad_list[i] * counts[i] / (sumCounts + 1e-9))

    global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)



def romoa(gradients, net, lr, f, byz, device, F, prev_global_update, seed):   # adapted from the original implementation provided by the authors
    """
    Based on the description in https://link.springer.com/chapter/10.1007/978-3-030-88418-5_23
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    F: sanitization factors of last epoch
    prev_global_update: previous global update
    seed: seed for random number generator
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)

    n = len(param_list)
    num_params = param_list[0].size(0)
    beta = 0.5
    gamma = 1/n

    # offsets for parameters of each layer/tensor
    offset = [0]
    for i, t in enumerate(gradients[0]):
        offset.append(offset[i] + torch.numel(t))

    ### parameter selection
    # whole parameter selection
    num_select = int(gamma * num_params)
    indices = [torch.topk(torch.abs(params.flatten()), k=num_select, sorted=False)[1] for params in param_list]     # indices of large garidents for each client
    element_level_idx = torch.unique(torch.cat(indices, dim=0))           # combining all indices and filtering out duplicates

    # layer-wise parameter selection
    layer_level_idx = []
    for i, j in zip(offset, offset[1:]):
        num_select = max(int(gamma * (j - i)), 1)
        indices = [torch.topk(torch.abs(params.flatten()[i:j]), k=num_select, sorted=False)[1] for params in param_list]
        layer_level_idx.append(torch.unique(torch.cat(indices, dim=0)))

    ### distance caluclations
    prev_global_update = prev_global_update.reshape(-1, 1)
    grad_mean, _ = torch.median(torch.stack(param_list, dim=1), dim=1)

    # element-wise cosine similarity
    cosine_element = torch.empty(size=(n, element_level_idx.size(0))).to(device)
    w0 = prev_global_update[element_level_idx]
    w1 = grad_mean[element_level_idx]
    v0 = torch.cat([w0, w1], dim=1)

    for i, grad in enumerate(param_list):
        w2 = grad[element_level_idx].reshape(-1, 1)
        v1 = torch.cat([w0, w2], dim=1)
        cosine_element[i] = -torch.nn.functional.cosine_similarity(v0, v1, dim=1)

    # layer-wise cosine similarity
    cosine_layer = torch.empty(size=(n, len(layer_level_idx)), dtype=torch.float).to(device)

    for worker, grad in enumerate(param_list):
        for layer, i, j in zip(range(len(gradients[0])), offset, offset[1:]):
            v0 = grad_mean[i:j][layer_level_idx[layer]]
            v1 = grad[i:j][layer_level_idx[layer]]

            cos_dist = -torch.nn.functional.cosine_similarity(v0, v1, dim=0)

            cosine_layer[worker][layer] = torch.where(torch.isnan(cos_dist), torch.tensor([-1], dtype=torch.float).to(device), cos_dist)

    # layer-wise pearson distance
    pearson_layer = torch.empty(size=(n, len(layer_level_idx)), dtype=torch.float).to(device)

    for worker, grad in enumerate(param_list):
        for layer, i, j in zip(range(len(gradients[0])), offset, offset[1:]):
            v0 = grad_mean[i:j][layer_level_idx[layer]]
            v1 = grad[i:j][layer_level_idx[layer]]

            try:
                dist = pearsonr(v0.detach().cpu().flatten().numpy(), v1.detach().cpu().flatten().numpy())[0]
            except ValueError:
                dist = -1
            dist = torch.tensor([dist], dtype=torch.float)

            pearson_layer[worker][layer] = torch.where(torch.isnan(dist), torch.tensor([-1], dtype=torch.float), dist)

    ### Sanitization Factor
    eps = 1e-5
    cluster = KMeans(n_clusters=2, n_init=n // 3, random_state=seed)        # same configuration as Romoa code

    values = torch.zeros(size=(3, n, num_params)).to(device)   # 3 for num of distances
    for i, distance in enumerate(["cosine_element", "cosine_layer", "pearson_layer"]):
        if distance == "cosine_element":
            label = cluster.fit_predict([t.detach().cpu().numpy() for t in cosine_element])
        elif distance == "cosine_layer":
            label = cluster.fit_predict([t.detach().cpu().numpy() for t in cosine_layer])
        else:
            label = cluster.fit_predict([t.detach().cpu().numpy() for t in pearson_layer])

        counter = dict(collections.Counter(label))
        weight = torch.tensor([counter[x] for x in label], dtype=torch.float).reshape(-1, 1).to(device)
        weight = (weight - weight.min()) / (weight.max() - weight.min() + eps)
        weight = weight / (weight.sum() + eps)

        if distance == "cosine_element":
            centroid = torch.sum(weight * cosine_element, dim=0)

            for j in range(n):
                score = 1 - torch.abs((cosine_element[j] - centroid) / (centroid + eps))
                values[i][j][element_level_idx] = score

        elif distance == "cosine_layer":
            centroid = torch.sum(weight * cosine_layer, dim=0)

            for j in range(n):
                score = 1 - torch.abs((cosine_layer[j] - centroid) / (centroid + eps))

                for k in range(len(offset) - 1):
                    values[i][j][offset[k]:offset[k+1]] = score[k]

        else:
            centroid = torch.sum(weight * pearson_layer, dim=0)

            for j in range(n):
                score = 1 - torch.abs((pearson_layer[j] - centroid) / (centroid + eps))

                for k in range(len(offset) - 1):
                    values[i][j][offset[k]:offset[k + 1]] = score[k]

    # Softmax
    if n * num_params > 6e5:
        F_t = torch.zeros(size=(n, num_params)).to(device)
        start, step = 0, num_params // 10
        stop = step
        while start < num_params:
            values_part = values[:, :, start:stop]
            F_part, _ = torch.min(torch.sign(values_part) * torch.exp(torch.abs(values_part) * n), dim=0)
            F_t[:, start:stop] = torch.nn.functional.softmax(F_part, dim=0)
            start = stop
            stop = min(start + step, num_params)
    else:
        F_t, _ = torch.min(torch.sign(values) * torch.exp(torch.abs(values) * n), dim=0)
        F_t = torch.nn.functional.softmax(F_t, dim=0)

    F = (1 - beta) * F_t + beta * F
    F = torch.where(torch.isnan(F), torch.zeros(size=(n, num_params)).to(device), F)       # set nan to 0

    # gradient clip
    upper, _ = torch.max(torch.stack(param_list, dim=1), dim=0)
    lower, _ = torch.min(torch.stack(param_list, dim=1), dim=0)
    max_value = torch.median(upper)
    min_value = torch.median(lower)
    param_list = [torch.clip(grad, max=max_value, min=min_value) for grad in param_list]

    # compute global model update
    global_update = torch.zeros(param_list[0].size()).to(device)
    for i, grad in enumerate(param_list):
        global_update += grad * F[i].reshape(-1, 1)    # element-wise multiplication

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)

    return F_t, global_update


def _heirichalFL(gradients, net, lr, f, byz, device, heirichal_params, seed):
    """
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    seed: seed for random number generator
    heirichal_params: dictionary containing user membership, user score, round, and number of groups
    """
    # Validate required keys in heirichal_params
    KEYS_set = set(["assumed_mal_prct", 'user membership', 'user score', 'round', 'num groups', 'history'])
    heirichal_params_keys = set(heirichal_params.keys())
    #make sure that all KEYS_set are in heirichal_params_keys
    if not KEYS_set.issubset(heirichal_params_keys):
        raise ValueError(f"heirichal_params should have the keys {KEYS_set}")
    
    KEYS_history_set = set(['round_num', 'user_membership', 'user_score_adjustment', 'group_scores', 'global_gradient', "user_scores"])
    heirichal_params_keys_history = set(heirichal_params['history'][0].keys())
    #make sure that all KEYS_history_set are in heirichal_params_keys_history
    if not KEYS_history_set.issubset(heirichal_params_keys_history):
        raise ValueError(f"history should have the keys {KEYS_history_set}")
    
    

    if "GT malicious" not in heirichal_params:
        heirichal_params["GT malicious"] = [True] * f + [False] * (len(gradients) - f)
        assert len(heirichal_params["GT malicious"]) == len(gradients), "GT malicious should have the same length as gradients"

    # Update round number
    heirichal_params['round'] += 1
    
    # Initialize current round record
    current_round_record = {
        "round_num": heirichal_params['round'],
        "user_membership": [],
        "user_score_adjustment": [],
        "group_scores": {},
        "user_scores": [],
        "global_gradient": None
    }
    
    skip_filtering = False
    if (heirichal_params['round'] < 20):
        skip_filtering = True
        
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # Let the malicious clients perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
        
    number_of_users = len(param_list)
    
    # Simulate groups and assign users
    heirichal_params = hfl.simulate_groups(heirichal_params, number_of_users, seed)

    # Record user membership for current round
    current_round_record["user_membership"] = heirichal_params["user membership"].copy()
    
    # Aggregate gradients for scoring
    group_gradients_for_scoring, filtered_users_None = hfl.aggregate_groups(param_list, device, seed, heirichal_params, skip_filtering=True)
    
    # Score groups based on their behavior
    groups_scores = hfl.score_groups(group_gradients_for_scoring, heirichal_params)
    current_round_record["group_scores"] = groups_scores.copy()
    
    # Update user scores based on group scores
    heirichal_params, user_scores_adjustments, current_user_scores = hfl.update_user_scores(heirichal_params, groups_scores)
    current_round_record["user_score_adjustment"] = user_scores_adjustments.copy()
    current_round_record["user_scores"] = current_user_scores.copy()
    
    # Aggregate gradients for model update
    group_gradients, filtered_users = hfl.aggregate_groups(param_list, device, seed, heirichal_params)
    
    # Shuffle users across groups
    heirichal_params = hfl.shuffle_users(heirichal_params, number_of_users, seed)
    
    # Handle empty group gradients
    if len(group_gradients) == 0:
        print("No gradients after filtering, skipping filtering")
        skip_filtering = True
        
    if skip_filtering:
        # If no gradients in group_gradients, use group_gradients_for_scoring
        group_gradients = group_gradients_for_scoring
        filtered_users = filtered_users_None
    
    # Apply robust aggregation to get global update
    heirichal_params["current group scores"] = groups_scores.copy()
    robust_update = hfl.robust_groups_aggregation(group_gradients, net, lr, device, heirichal_params, number_of_users)
    current_round_record["global_gradient"] = robust_update.clone().detach()
    current_round_record["filtered_users"] = filtered_users.copy()
    
    # Add current round record to history
    heirichal_params["history"].append(current_round_record)

    #
    utils.save_data_to_csv(heirichal_params, f)
    
    return heirichal_params






def heirichalFL(gradients, net, lr, f, byz, device, heirichal_params, seed):
    """
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    seed: seed for random number generator
    heirichal_params: dictionary containing user membership, user score, round, and number of groups
    """
    # Validate required keys in heirichal_params
    KEYS_set = set(["assumed_mal_prct", 'user membership', 'user score', 'round', 'num groups', 'history'])
    heirichal_params_keys = set(heirichal_params.keys())
    #make sure that all KEYS_set are in heirichal_params_keys
    if not KEYS_set.issubset(heirichal_params_keys):
        raise ValueError(f"heirichal_params should have the keys {KEYS_set}")
    
    KEYS_history_set = set(['round_num', 'user_membership', 'user_score_adjustment', 'group_scores', 'global_gradient', "user_scores"])
    heirichal_params_keys_history = set(heirichal_params['history'][0].keys())
    #make sure that all KEYS_history_set are in heirichal_params_keys_history
    if not KEYS_history_set.issubset(heirichal_params_keys_history):
        raise ValueError(f"history should have the keys {KEYS_history_set}")
    
    

    if "GT malicious" not in heirichal_params:
        heirichal_params["GT malicious"] = [True] * f + [False] * (len(gradients) - f)
        assert len(heirichal_params["GT malicious"]) == len(gradients), "GT malicious should have the same length as gradients"

    # Update round number
    heirichal_params['round'] += 1
    
    # Initialize current round record
    current_round_record = {
        "round_num": heirichal_params['round'],
        "user_membership": [],
        "user_score_adjustment": [],
        "group_scores": {},
        "user_scores": [],
        "global_gradient": None
    }
    
    skip_filtering = False
    if (heirichal_params['round'] < 20):
        skip_filtering = True
        
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # Let the malicious clients perform the byzantine attack
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
        
    number_of_users = len(param_list)
    
    # Simulate groups and assign users
    heirichal_params = hfl.simulate_groups(heirichal_params, number_of_users, seed)

    # Record user membership for current round
    current_round_record["user_membership"] = heirichal_params["user membership"].copy()
    
    # Aggregate gradients for scoring
    group_gradients_for_scoring, filtered_users_None = hfl.aggregate_groups(param_list, device, seed, heirichal_params, skip_filtering=True)
    
    # Score groups based on their behavior
    groups_scores = hfl.score_groups(group_gradients_for_scoring, heirichal_params)
    current_round_record["group_scores"] = groups_scores.copy()
    
    # Update user scores based on group scores
    heirichal_params, user_scores_adjustments, current_user_scores = hfl.update_user_scores(heirichal_params, groups_scores)
    current_round_record["user_score_adjustment"] = user_scores_adjustments.copy()
    current_round_record["user_scores"] = current_user_scores.copy()
    
    # Aggregate gradients for model update
    group_gradients, filtered_users = hfl.aggregate_groups(param_list, device, seed, heirichal_params)
    
    # Shuffle users across groups
    heirichal_params = hfl.shuffle_users(heirichal_params, number_of_users, seed)
    
    # Handle empty group gradients
    if len(group_gradients) == 0:
        print("No gradients after filtering, skipping filtering")
        skip_filtering = True
        
    if skip_filtering:
        # If no gradients in group_gradients, use group_gradients_for_scoring
        group_gradients = group_gradients_for_scoring
        filtered_users = filtered_users_None
    
    # Apply robust aggregation to get global update
    heirichal_params["current group scores"] = groups_scores.copy()
    robust_update = hfl.robust_groups_aggregation(group_gradients, net, lr, device, heirichal_params, number_of_users)
    current_round_record["global_gradient"] = robust_update.clone().detach()
    current_round_record["filtered_users"] = filtered_users.copy()
    
    # Add current round record to history
    heirichal_params["history"].append(current_round_record)

    #
    utils.save_data_to_csv(heirichal_params, f)
    
    return heirichal_params






def bayesian_fl_aggregation_rule(gradients, net, lr, f, byz, device, heirichal_params, seed):
    """
    Aggregation rule for FL using Bayesian inference to update node fault probabilities.
    """
    # --- 1. Standard setup, parameter validation, round update ---
    REQUIRED_BBN_KEYS = ['bbn_config']
    if not all(key in heirichal_params for key in REQUIRED_BBN_KEYS):
        raise ValueError(f"heirichal_params missing BBN configuration keys: {REQUIRED_BBN_KEYS}")
    
    # Ensure user_score is initialized and is a numpy array for BBN
    if not isinstance(heirichal_params.get('user_score'), np.ndarray):
        initial_prob = heirichal_params['bbn_config'].get('initial_fault_probability', 0.1) # Default
        heirichal_params['user_score'] = np.full(len(gradients), initial_prob)
    current_node_fault_probs = heirichal_params['user_score']
    
    # Validate other keys (copied from your heirichalFL)
    KEYS_set = set(["assumed_mal_prct", 'user membership', 'user score', 'round', 'num groups', 'history', 'bbn_config'])
    if not KEYS_set.issubset(heirichal_params.keys()):
        raise ValueError(f"heirichal_params should have the keys {KEYS_set}")
    
    if heirichal_params['history']: # Check if history is not empty
        KEYS_history_set = set(['round_num', 'user_membership', 'user_score_adjustment', 'group_scores', 'global_gradient', "user_scores"])
        if not KEYS_history_set.issubset(heirichal_params['history'][0].keys()):
             raise ValueError(f"history should have the keys {KEYS_history_set}")
    
    if "GT malicious" not in heirichal_params: # Ground truth for simulation
        heirichal_params["GT malicious"] = [True] * f + [False] * (len(gradients) - f)

    heirichal_params['round'] += 1
    current_round_record = {
        "round_num": heirichal_params['round'],
        "user_membership": [], # Will be populated by hfl.simulate_groups or similar
        "user_score_adjustment": [], # BBN directly gives new scores, so adjustment might be implicit
        "group_scores_observed": {}, # Raw scores from hfl.score_groups
        "user_scores_prior_bbn": current_node_fault_probs.tolist(),
        "user_scores": [], # Will be posterior from BBN
        "global_gradient": None,
        "filtered_users": [] # If filtering is done
    }
    
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
        
    number_of_users = len(param_list)
    

    number_of_users = len(param_list)
    node_ids = list(range(number_of_users))

    

    groups_for_bbn = heuristic_suspect_isolation_grouping_strategy(
        node_ids,
        current_node_fault_probs,
        heirichal_params['bbn_config']['suspect_threshold_for_heuristic_grouping'],
        heirichal_params['bbn_config']['num_clearance_groups_for_heuristic_grouping']
    )
    # Update user_membership_dict based on groups_for_bbn for consistency if needed by hfl.score_groups
    user_membership_dict_for_bbn = {}
    for group_idx, user_list in enumerate(groups_for_bbn):
        for user_id in user_list:
            user_membership_dict_for_bbn[user_id] = group_idx # group_idx serves as group_id
    
    current_round_record["user_membership"] = user_membership_dict_for_bbn # Log FL grouping
    
    

    # --- 4. Get group observations for BBN
    # TODO: we need to aggregate the users groups based on groups_for_bbn and gradients list
    # TODO: we need to score groups based on their gradients (the observation function) from 0 to 1. group-gradients -> probability of being malicious from 0 to 1.
    # output is group_observations_for_bbn (dict group_id -> scaled_score_0_to_1)
    current_round_record["group_scores_observed"] = group_observations_for_bbn.copy()

    # --- 6. BBN Update Step --- #CAREFUL: THE FUNCTION WILL KEEP INITIALIZING BBN BECAUSE IT IS CALLED IN FOR EACH FL ROUND. IS THIS EXPECTED? 
    bbn_instance = IterativeBBN(number_of_users, heirichal_params['bbn_config'])
    bbn_instance.build_iteration_network(current_node_fault_probs, groups_for_bbn)
    bbn_instance.set_group_observations(group_observations_for_bbn)
    updated_node_fault_probs = bbn_instance.infer_node_posteriors()
    
    heirichal_params['user_score'] = updated_node_fault_probs # Update main user scores with BBN posteriors
    current_round_record["user_scores"] = updated_node_fault_probs.tolist()
    current_round_record["user_score_adjustment"] = (updated_node_fault_probs - current_node_fault_probs).tolist()


    # --- 7. Robust Aggregation using BBN-updated probabilities ---
    # TODO: from the groups_for_bbn, give zero weight to the group that the suspected malicious user in it.
    # then give weights to the groups based on the BBN probabilities of the users in the groups. 
    #Then simply aggregate the gradients of the groups using simple weighted average. (dont implement or use a funcion, just do it here)


    # --- 8. Logging ---
    current_round_record["global_gradient"] = robust_update.clone().detach()
    heirichal_params["history"].append(current_round_record)
    
    # Assuming utils.save_data_to_csv exists and works with heirichal_params
    # utils.save_data_to_csv(heirichal_params, f) 

    # --- 9. Return heirichal_params ---
    return heirichal_params

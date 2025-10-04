import torch
import random
def _group_and_sum_gradientsFORSIGNGUARD(param_list, bayesian_params = {}):
    groups, group_gradients = _SG_group_and_sum_gradients(param_list, bayesian_params)

    group_gradients = [group_gradients[gid] for gid in sorted(group_gradients.keys())]


    return group_gradients

from collections import defaultdict
def _SG_group_and_sum_gradients(param_list, bayesian_params):

    sg_aggregators_len = bayesian_params.get('sg_aggregators_len', 5)
    n = len(param_list)
    seed = bayesian_params.get('seed', 47)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    sg_agg_assignment = {}
    for agg_id in range(sg_aggregators_len):
        for idx in indices[agg_id::sg_aggregators_len]:
            sg_agg_assignment[idx] = agg_id
    agg_to_ids = defaultdict(list)

    for cid, aid in sg_agg_assignment.items():
        agg_to_ids[aid].append(cid)


    strategy = bayesian_params.get("shuffling_strategy", "random")



    

    if strategy == "random":
        # randomly pick 10% of each aid
        randomly_picked_pct_from_sg_agg = bayesian_params.get('randomly_picked_pct_from_sg_agg', 0.1)
        selected_ids = []
        for aid, ids in agg_to_ids.items():
            k = max(2, int(randomly_picked_pct_from_sg_agg * len(ids)))
            selected_ids.extend(rng.sample(ids, k))

        

        # Extract group gradients
        group_gradients = {}
        for aid, indices in agg_to_ids.items():
            # compute global model update
            gradients_to_be_summed = [param_list[idx] for idx in indices if idx in selected_ids]
            group_gradients[aid]  = torch.sum(torch.stack(gradients_to_be_summed), dim=0)
        groups = {aid: (aid, indices) for aid, indices in agg_to_ids.items()}
        return groups, group_gradients
    else:
        raise ValueError(f"Unknown shuffling strategy: {bayesian_params.get('shuffling_strategy')}")
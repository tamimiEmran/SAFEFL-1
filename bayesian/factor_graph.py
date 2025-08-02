
from pgmax import fgraph
from pgmax.factor import EnumFactor
from itertools import product
from pgmax.infer import BP, get_marginals
import numpy as np
from components import observation_function, expectation_function, likelihood_function


def initialize_factor_graph(gradients):

    num_nodes = len(gradients)
    # Initialize the factor graph
    graph = fgraph.FactorGraph()
    graph.add_variables(num_nodes)  # F_1 … F_N

    return graph

def factor_graph_marginals(gradients, factor_graph_params):
    """
    Aggregation rule for FL using Bayesian inference to update node fault probabilities.
    """
    num_nodes = len(gradients)
    round_id = factor_graph_params.get("round_id", 1)
    if round_id == 1:
        # Initialize the factor graph
        graph = initialize_factor_graph(gradients)

    group_gradients = factor_graph_params.get("group_gradients", {})
    group_participants = factor_graph_params.get("group_participants", {})
    observed_scores = observation_function(group_gradients)


    for group_id, indices in group_participants.values():
        
        indices_cardinality = [(i, 2) for i in indices]  # each node can be either faulty or not
        factor_configs = np.array(list(product([0,1], repeat=len(indices))))  # all combinations of faulty/not faulty for the group

        likelihoods = []  # likelihoods for each configuration
        for config in factor_configs:
            #config is tuple
            expectation_config = expectation_function(config)
            likelihood_config = likelihood_function(observed_scores[group_id], expectation_config)
            likelihoods.append(likelihood_config)

        likelihoods = np.array(likelihoods)
        factor = EnumFactor(factor_configs= factor_configs,
                            variables= indices_cardinality,
                            log_potentials= likelihoods)
        
        graph.add_factor(factor, indices)  # add the factor to the graph
    
    bp_state = graph.to_bp_state()  # convert to BP state
    bp = BP(bp_state, temperature=1.0)  # create BP object
    bp_arrays = bp.run(max_iter=100, tol=1e-5)  # run BP inference # or use damping
    beliefs = bp.get_beliefs(bp_arrays)  # get beliefs
    marginals = get_marginals(beliefs)  # get marginals

    return marginals
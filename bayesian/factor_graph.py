
from pgmax import fgraph, vgroup
from pgmax.factor import EnumFactor
from itertools import product
from pgmax.infer import BP, get_marginals
import numpy as np
from .components import observation_function, expectation_function, likelihood_function


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


import pandas as pd
def _maybe_init_bayesian_and_csv(bayesian_params, num_nodes):
    # get round_id 
    round_id = bayesian_params.get("current_round", 0)
    if round_id == 0:
        # first round, initialize grouping
        INITIAL_THRESHOLD = bayesian_params.get("initial_threshold", 0.5)
        bayesian_params["latent_variables"] = {id: INITIAL_THRESHOLD for id in range(num_nodes)}
        
        variables = vgroup.NDVarArray(num_states=2, shape=(num_nodes,))  # Each node can be either faulty (1) or not faulty (0)
        graph = fgraph.FactorGraph(
            variable_groups= [variables]
        )
        
        bayesian_params["graph"] = graph
        bayesian_params["variables"] = variables # Store variables for easy access
        DIR = r'M:\PythonTests\newSafeFL\SAFEFL\score_function_viz\observation_scores.csv'
        # initialize csv (keep original path & spelling)
        pd.DataFrame(columns=["round_id", "group_id", "numberOfMal", "score", 'avgMalScore', 'avgNormScore', 'minMalScore', 'maxNormScore']).to_csv(
            DIR, index=False
        )

    graph = bayesian_params["graph"]
    variables = bayesian_params["variables"]
    return round_id, graph, variables



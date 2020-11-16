"""Extract data"""

import os
import random

import numpy as np
import pandas as pd

np.random.seed(10)


def load_generators(data_dir):
    """Load generator data"""

    return pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset', 'generators', 'generators.csv')).set_index('DUID')


def load_nodes(data_dir):
    """Load node data"""

    return (pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset', 'network', 'network_nodes.csv'))
            .set_index('NODE_ID'))


def load_ac_edges(data_dir):
    """Load edge data"""

    return (pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset', 'network', 'network_edges.csv'))
            .set_index('LINE_ID'))


def load_hvdc_edges(data_dir):
    """Load HVDC edges"""

    return (pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset', 'network', 'network_hvdc_links.csv'))
            .set_index('HVDC_LINK_ID'))


def load_interconnectors(data_dir):
    """Load interconnector data"""

    return (pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset', 'network', 'network_ac_interconnector_links.csv'))
            .set_index('INTERCONNECTOR_ID'))


def load_flow_limits(data_dir):
    """Load interconnector flow limits"""

    return (pd.read_csv(
        os.path.join(data_dir, 'egrimod-nem-dataset', 'network', 'network_ac_interconnector_flow_limits.csv'))
            .set_index('INTERCONNECTOR_ID'))


def load_scenarios(scenarios_dir):
    """Load scenario data"""

    return pd.read_pickle(os.path.join(scenarios_dir, '2_scenarios.pickle'))


def get_node_index(nodes):
    """Get node index"""

    return nodes.index.to_list()


def get_generator_index(generators):
    """Get generator index"""

    # Only keep fossil and scheduled generators
    mask = (generators['FUEL_CAT'] == 'Fossil') & (generators['SCHEDULE_TYPE'] == 'SCHEDULED')

    return generators[mask].index.to_list()


def get_ac_edge_index(ac_edges):
    """Get edge index"""

    # Edge indices
    edges_1 = ac_edges.apply(lambda x: (x['FROM_NODE'], x['TO_NODE']), axis=1).to_list()
    edges_2 = ac_edges.apply(lambda x: (x['TO_NODE'], x['FROM_NODE']), axis=1).to_list()

    # All combinations of from and to nodes
    all_edges = edges_1 + edges_2

    return list(set(all_edges))


def get_hvdc_edge_index(hvdc_edges):
    """Get HVDC edge index"""

    return hvdc_edges.index.to_list()


def get_aggregated_edge_index(interconnectors):
    """Get flow index"""

    return list(set([f'{i}-{j}' for i in interconnectors.index.to_list() for j in ['FORWARD', 'REVERSE']]))


def get_scenario_index(scenarios):
    """Get scenario index"""

    return scenarios.columns.to_list()


def get_region_index(nodes):
    """Get region index"""

    return nodes['NEM_REGION'].unique().tolist()


def get_ac_interconnector_branches(interconnectors):
    """Get AC interconnector branches"""

    # Forward and reverse branch nodes
    forward = interconnectors.apply(lambda x: (x['FROM_NODE'], x['TO_NODE']), axis=1).add_suffix('-FORWARD')
    reverse = interconnectors.apply(lambda x: (x['TO_NODE'], x['FROM_NODE']), axis=1).add_suffix('-REVERSE')

    # Reset index
    ac_branches = pd.concat([forward, reverse]).rename('BRANCH').reset_index()
    ac_branches.index = [f'L_{i + 1}' for i in ac_branches.index]

    return ac_branches


def get_ac_interconnector_branch_index(interconnectors):
    """Get interconnector branch index"""

    return get_ac_interconnector_branches(interconnectors).index.to_list()


def get_binary_expansion_integer_index(options):
    """Set of integers used when implementing binary expansion"""

    # Largest integer in set
    largest_integer = options['parameters'].get('P_BINARY_EXPANSION_LARGEST_INTEGER', -1)

    return range(0, largest_integer + 1)


def get_admittance_matrix(nodes, ac_edges, tmp_dir, use_cache):
    """Construct admittance matrix for network"""

    if use_cache:
        try:
            return pd.read_pickle(os.path.join(tmp_dir, 'admittance_matrix.pickle'))
        except FileNotFoundError:
            pass

    # Initialise DataFrame
    admittance_matrix = pd.DataFrame(data=0j, index=nodes.index, columns=nodes.index)

    # Off-diagonal elements
    for index, row in ac_edges.iterrows():
        fn, tn = row['FROM_NODE'], row['TO_NODE']
        admittance_matrix.loc[fn, tn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']
        admittance_matrix.loc[tn, fn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']

    # Diagonal elements
    for i in nodes.index:
        admittance_matrix.loc[i, i] = - admittance_matrix.loc[i, :].sum()

    # Add shunt susceptance to diagonal elements
    for index, row in ac_edges.iterrows():
        fn, tn = row['FROM_NODE'], row['TO_NODE']
        admittance_matrix.loc[fn, fn] += (row['B_PU'] / 2) * row['NUM_LINES']
        admittance_matrix.loc[tn, tn] += (row['B_PU'] / 2) * row['NUM_LINES']

    # Cache admittance matrix
    admittance_matrix.to_pickle(os.path.join(tmp_dir, 'admittance_matrix.pickle'))

    return admittance_matrix


def get_branch_susceptance(nodes, ac_edges, tmp_dir, use_cache):
    """Get admittance matrix elements"""

    # Get admittance matrix
    admittance_matrix = get_admittance_matrix(nodes, ac_edges, tmp_dir, use_cache)

    # All AC edges
    ac_edges_index = get_ac_edge_index(ac_edges)

    # Admittance matrix elements
    matrix_elements = {k: v for k, v in pd.DataFrame(np.imag(admittance_matrix),
                                                     index=admittance_matrix.index,
                                                     columns=admittance_matrix.columns).unstack().to_dict().items()}

    # Filtered matrix elements - scale by 100 (base power)
    output = {k: matrix_elements[k] * 100 for k in ac_edges_index}

    return output


def get_hvdc_forward_limit(hvdc_edges):
    """Get HVDC forward limit"""

    return hvdc_edges['FORWARD_LIMIT_MW'].to_dict()


def get_hvdc_reverse_limit(hvdc_edges):
    """Get HVDC reverse limit"""

    return {k: -v for k, v in hvdc_edges['REVERSE_LIMIT_MW'].to_dict().items()}


def get_reference_node_indicator(nodes):
    """Get reference node indicators"""

    # Using VIC and TAS Regional Reference Nodes as voltage angle reference nodes for the NEM's two AC grids
    rrn_mask = nodes['NEM_REGION'].isin(['VIC1', 'TAS1']) & (nodes['RRN'] == 1)

    return rrn_mask.astype(int).to_dict()


def get_generator_attribute(generators, attribute):
    """Get max generator output"""

    return {k: v for k, v in generators[attribute].to_dict().items() if k in get_generator_index(generators)}


def get_generator_min_output(generators):
    """Get min generator output - assumed to be 0"""

    return {k: float(0) for k in get_generator_index(generators)}


def get_hvdc_incidence_matrix(nodes, hvdc_edges):
    """Get HVDC incidence matrix"""

    # Initialise incidence matrix for HVDC links
    df = pd.DataFrame(index=nodes.index, columns=hvdc_edges.index, data=0)

    for index, row in hvdc_edges.iterrows():
        # From nodes assigned a value of 1
        df.loc[row['FROM_NODE'], index] = 1

        # To nodes assigned a value of -1
        df.loc[row['TO_NODE'], index] = -1

    # Convert to dictionary
    return df.unstack().reorder_levels([1, 0]).to_dict()


def get_interconnector_incidence_matrix(interconnectors, nodes):
    """Incidence matrix showing if AC interconnector branch is defined as (+) or (-) flow for each node"""

    # Branches constituting AC interconnectors
    interconnector_branches = get_ac_interconnector_branches(interconnectors)

    # Initialise interconnector branch - node incidence matrix
    df = pd.DataFrame(index=interconnector_branches.index, columns=nodes.index, data=0)

    for index, row in df.iterrows():
        # Branch from node
        from_node = interconnector_branches.loc[index, 'BRANCH'][0]

        # Branch to node
        to_node = interconnector_branches.loc[index, 'BRANCH'][1]

        # Update values in matrix
        df.loc[index, from_node] = 1
        df.loc[index, to_node] = -1

    return df.T.unstack().reorder_levels([1, 0]).to_dict()


def get_network_interconnector_flow_limit(flow_limits):
    """Get aggregate AC interconnector flow limits"""

    # Forward and reverse power flow limits
    forward = {k + '-FORWARD': float(v['FORWARD_LIMIT_MW']) for k, v in flow_limits.iterrows()}
    reverse = {k + '-REVERSE': float(v['REVERSE_LIMIT_MW']) for k, v in flow_limits.iterrows()}

    return {**forward, **reverse}


def get_network_fixed_injection(scenarios):
    """Get fixed power injection at each node"""

    return scenarios.loc['hydro'].add(scenarios.loc['intermittent']).stack().reorder_levels([1, 0]).to_dict()


def get_network_demand(scenarios):
    """Get demand at each node"""

    return scenarios.loc['demand'].stack().reorder_levels([1, 0]).to_dict()


def get_network_region_demand_proportion(scenarios, nodes):
    """Proportion of total network demand consumed by each region"""

    # Total demand consumed by each region
    region_demand = (scenarios.loc['demand'].join(nodes[['NEM_REGION']])
                     .set_index('NEM_REGION', append=True).groupby('NEM_REGION').sum())

    # Proportion of total network demand consumed by each region
    region_demand_proportion = region_demand.div(region_demand.sum())

    return region_demand_proportion.stack().reorder_levels([1, 0]).to_dict()


def get_scenario_duration(scenarios):
    """Get relative scenario duration"""

    return {k: v / 8760 for k, v in scenarios.loc['hours'].T['duration'].to_dict().items()}


def get_generator_node(generators):
    """Node to which each generator is assigned"""

    # All generators used within model
    generator_index = get_generator_index(generators)

    return {k: v for k, v in generators['NODE'].astype(int).to_dict().items() if k in generator_index}


def get_network_graph(nodes, ac_edges):
    """Graph containing connections between all network nodes"""

    # Initialise network graph container
    network_graph = {n: set() for n in nodes.index}

    for index, row in ac_edges.iterrows():
        network_graph[row['FROM_NODE']].add(row['TO_NODE'])
        network_graph[row['TO_NODE']].add(row['FROM_NODE'])

    return network_graph


def get_ac_interconnector_branch_id_map(interconnectors):
    """Get branches that constitute each interconnector"""

    # AC interconnector branches
    branches = get_ac_interconnector_branches(interconnectors)

    # Branches constituting each interconnector
    return branches.reset_index().groupby('INTERCONNECTOR_ID').apply(lambda x: list(x['index'])).to_dict()


def get_ac_interconnector_branch_nodes_map(interconnectors):
    """Get mapping between branch IDs and nodes constituting those branches"""

    return get_ac_interconnector_branches(interconnectors)['BRANCH'].to_dict()


def get_hvdc_from_node(hvdc_edges):
    """Get HVDC 'from' node"""

    return hvdc_edges['FROM_NODE'].to_dict()


def get_hvdc_to_node(hvdc_edges):
    """Get HVDC 'to' node"""

    return hvdc_edges['TO_NODE'].to_dict()


def get_node_generator_map(generators, nodes):
    """Get generators assigned to each node"""

    # All generators used in model
    generator_index = get_generator_index(generators)

    # List of generators assigned to each node
    generator_node_map = (generators.loc[generator_index, 'NODE'].astype(int).reset_index().groupby('NODE')['DUID']
                          .apply(lambda x: list(x)).to_dict())

    # All nodes used within the model
    node_index = get_node_index(nodes)

    return {k: generator_node_map[k] if k in generator_node_map.keys() else [] for k in node_index}


def get_region_rrn_node_map(nodes):
    """Get regional reference node for each region"""

    return nodes.loc[nodes['RRN'] == 1, 'NEM_REGION'].reset_index().set_index('NEM_REGION')['NODE_ID'].to_dict()


def get_generator_srmc(generators):
    """Short-run marginal cost for each generator"""

    # Perturb SRMCs by a small amount to assist solver find a unique solution
    random.seed(10)
    srmc_perturbation = pd.Series({i: random.uniform(0, 2) for i in generators.index.to_list()})
    srmc = generators['SRMC_2016-17'] + srmc_perturbation

    # Extract SRMCs
    generator_index = get_generator_index(generators)

    return {k: v for k, v in srmc.to_dict().items() if k in generator_index}


def get_case_data(data_dir, scenarios_dir, tmp_dir, options, use_cache):
    """Get case data"""

    # Load generator data
    generators = load_generators(data_dir)

    # Load network data
    nodes = load_nodes(data_dir)
    ac_edges = load_ac_edges(data_dir)
    hvdc_edges = load_hvdc_edges(data_dir)
    flow_limits = load_flow_limits(data_dir)
    interconnectors = load_interconnectors(data_dir)

    # Load scenario data
    scenarios = load_scenarios(scenarios_dir)

    # Get model data
    data_dict = {
        'S_NODES': get_node_index(nodes),
        'S_GENERATORS': get_generator_index(generators),
        'S_AC_EDGES': get_ac_edge_index(ac_edges),
        'S_AC_AGGREGATED_EDGES': get_aggregated_edge_index(flow_limits),
        'S_HVDC_EDGES': get_hvdc_edge_index(hvdc_edges),
        'S_SCENARIOS': get_scenario_index(scenarios),
        'S_REGIONS': get_region_index(nodes),
        'S_INTERCONNECTORS': get_ac_interconnector_branch_index(interconnectors),
        'S_BINARY_EXPANSION_INTEGERS': get_binary_expansion_integer_index(options),
        'P_NETWORK_BRANCH_SUSCEPTANCE': get_branch_susceptance(nodes, ac_edges, tmp_dir, use_cache),
        'P_NETWORK_HVDC_FORWARD_LIMIT': get_hvdc_forward_limit(hvdc_edges),
        'P_NETWORK_HVDC_REVERSE_LIMIT': get_hvdc_reverse_limit(hvdc_edges),
        'P_NETWORK_REFERENCE_NODE_INDICATOR': get_reference_node_indicator(nodes),
        'P_GENERATOR_MAX_OUTPUT': get_generator_attribute(generators, 'REG_CAP'),
        'P_GENERATOR_MIN_OUTPUT': get_generator_min_output(generators),
        'P_GENERATOR_SRMC': get_generator_srmc(generators),
        'P_GENERATOR_EMISSIONS_INTENSITY': get_generator_attribute(generators, 'EMISSIONS'),
        'P_NETWORK_HVDC_INCIDENCE_MAT': get_hvdc_incidence_matrix(nodes, hvdc_edges),
        'P_NETWORK_INTERCONNECTOR_INCIDENCE_MAT': get_interconnector_incidence_matrix(interconnectors, nodes),
        'P_NETWORK_INTERCONNECTOR_LIMIT': get_network_interconnector_flow_limit(flow_limits),
        'P_BINARY_EXPANSION_LARGEST_INTEGER': options['parameters'].get('P_BINARY_EXPANSION_LARGEST_INTEGER', -1),
        'P_NETWORK_FIXED_INJECTION': get_network_fixed_injection(scenarios),
        'P_NETWORK_DEMAND': get_network_demand(scenarios),
        'P_NETWORK_REGION_DEMAND_PROPORTION': get_network_region_demand_proportion(scenarios, nodes),
        'P_SCENARIO_DURATION': get_scenario_duration(scenarios),
        'P_GENERATOR_NODE': get_generator_node(generators),
        'P_NETWORK_GRAPH': get_network_graph(nodes, ac_edges),
        'P_NETWORK_INTERCONNECTOR_BRANCH_ID_MAP': get_ac_interconnector_branch_id_map(interconnectors),
        'P_NETWORK_INTERCONNECTOR_BRANCH_NODES_MAP': get_ac_interconnector_branch_nodes_map(interconnectors),
        'P_NETWORK_HVDC_FROM_NODE': get_hvdc_from_node(hvdc_edges),
        'P_NETWORK_HVDC_TO_NODE': get_hvdc_to_node(hvdc_edges),
        'P_NETWORK_NODE_GENERATOR_MAP': get_node_generator_map(generators, nodes),
        'P_POLICY_FIXED_BASELINE': options['parameters'].get('P_POLICY_FIXED_BASELINE', -1),
        'P_POLICY_PERMIT_PRICE_TARGET': options['parameters'].get('P_POLICY_PERMIT_PRICE_TARGET', -1),
        'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': options['parameters'].get('P_POLICY_WEIGHTED_RRN_PRICE_TARGET', -1),
        'P_REGION_RRN_NODE_MAP': get_region_rrn_node_map(nodes),
    }

    return data_dict


if __name__ == '__main__':
    # Data directories for scenarios
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir,
                                  os.path.pardir, 'data')
    scenario_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir,
                                      '1_create_scenarios', 'output')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'tmp')

    # Load generator data
    d_generators = load_generators(data_directory)

    # Load network data
    d_nodes = load_nodes(data_directory)
    d_ac_edges = load_ac_edges(data_directory)
    d_hvdc_edges = load_hvdc_edges(data_directory)
    d_flow_limits = load_flow_limits(data_directory)
    d_interconnectors = load_interconnectors(data_directory)

    # Load scenario data
    d_scenarios = load_scenarios(scenario_directory)

    # Options
    d_options = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_PERMIT_PRICE_TARGET': 30,
            'P_POLICY_FIXED_BASELINE': 0.95,
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 36,
        },
        'mode': 'feasibility',
    }

    # Construct case data
    case_data = get_case_data(data_directory, scenario_directory, tmp_directory, d_options, use_cache=True)

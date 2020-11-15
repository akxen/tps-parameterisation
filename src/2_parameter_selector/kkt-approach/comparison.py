"""Check model definitions"""

import os

from parameter_selector import OrganiseData, check_bau_model, create_model

from utils.data import get_case_data, load_scenarios
from model import construct_model, configure_model, solve_model


def extract_value(v):
    """Extract model value"""

    if type(v) in [int, float]:
        return v
    else:
        return v.value


def compare_indexed_component(m_1, m_2, m_1_name, m_2_name):
    """Check that parameters are same for both models"""

    # Extract model attributes
    obj_1 = m_1.__getattribute__(m_1_name)
    obj_2 = m_2.__getattribute__(m_2_name)

    # Check keys match
    keys_1 = list(obj_1.keys())
    keys_1.sort()

    keys_2 = list(obj_2.keys())
    keys_2.sort()

    assert keys_1 == keys_2, 'Keys do not match'

    # Difference in values
    difference = {k: abs(obj_1[k] - extract_value(v)) for k, v in obj_2.items()}

    # Max difference
    max_difference = max([v for k, v in difference.items()])

    # Combine into single DataFrame
    output = {'difference': difference, 'max_difference': max_difference}

    return output


def compare_singleton_component(m_1, m_2, m_1_name, m_2_name):
    """Compare non indexed parameters"""

    # Extract model attributes
    obj_1 = m_1.__getattribute__(m_1_name)
    obj_2 = m_2.__getattribute__(m_2_name)

    # Check both parameters are not indexed
    cond_1 = not obj_1.is_indexed()
    cond_2 = not obj_2.is_indexed()

    assert cond_1 and cond_2, 'Parameter is indexed'

    # Difference between parameters
    output = {'difference': abs(obj_1.value - obj_2.value), 'max_difference': abs(obj_1.value - obj_2.value)}

    return output


def compare_block_component(m_1, m_2, m_1_name, m_2_name):
    """Compare block parameters"""

    # Extract model attributes
    m_1_params = {(k, i): extract_value(j) for k in m_1.SCENARIO.keys()
                  for i, j in m_1.SCENARIO[k].__getattribute__(m_1_name).items()}
    m_2_params = {k: extract_value(v) for k, v in m_2.__getattribute__(m_2_name).items()}

    # Check keys match
    keys_1 = list(m_1_params.keys())
    keys_1.sort()

    keys_2 = list(m_2_params.keys())
    keys_2.sort()

    assert keys_1 == keys_2, 'Keys do not match'

    # Difference in values
    difference = {k: abs(m_1_params[k] - extract_value(v)) for k, v in m_2_params.items()}

    # Max difference
    max_difference = max([v for k, v in difference.items()])

    # Combine into single DataFrame
    output = {'difference': difference, 'max_difference': max_difference}

    return output


def check_parameters(m_1, m_2):
    """Check model parameters"""

    indexed_param_names = [
        ('B', 'P_NETWORK_BRANCH_SUSCEPTANCE'),
        ('P_H_MAX', 'P_NETWORK_HVDC_FORWARD_LIMIT'),
        ('P_H_MIN', 'P_NETWORK_HVDC_REVERSE_LIMIT'),
        ('S_R', 'P_NETWORK_REFERENCE_NODE_INDICATOR'),
        ('P_MAX', 'P_GENERATOR_MAX_OUTPUT'),
        ('P_MIN', 'P_GENERATOR_MIN_OUTPUT'),
        ('C', 'P_GENERATOR_SRMC'),
        ('E', 'P_GENERATOR_EMISSIONS_INTENSITY'),
        ('K', 'P_NETWORK_HVDC_INCIDENCE_MAT'),
        ('S_L', 'P_NETWORK_INTERCONNECTOR_INCIDENCE_MAT'),
        ('F', 'P_NETWORK_INTERCONNECTOR_LIMIT'),

    ] + [(f'M_{i}{j}', f'P_M_{i}{j}') for i in range(1, 7) for j in range(1, 3)]

    singleton_param_names = [
        ('BASE_POWER', 'P_NETWORK_BASE_POWER'),
        ('PHI', 'P_POLICY_FIXED_BASELINE'),
        ('M_71', 'P_M_71'),
        ('M_72', 'P_M_72'),
        # ('PHI_MIN', 'P_BIN_EXP_MIN_BASELINE'),
        # ('PHI_MAX', 'P_BIN_EXP_MAX_BASELINE'),
        # ('PHI_DELTA', 'P_BIN_EXP_BASELINE_DELTA'),
    ]

    block_param_names = [
        ('R', 'P_NETWORK_FIXED_INJECTION'),
        ('D', 'P_NETWORK_DEMAND'),
        ('ZETA', 'P_NETWORK_REGION_DEMAND_PROPORTION'),
        # ('RHO', 'P_SCENARIO_DURATION'),
    ]

    indexed_param_output = {
        (k_1, k_2): compare_indexed_component(m_1, m_2, k_1, k_2) for k_1, k_2 in indexed_param_names
    }

    singleton_param_output = {
        (k_1, k_2): compare_singleton_component(m_1, m_2, k_1, k_2) for k_1, k_2 in singleton_param_names
    }

    block_param_output = {
        (k_1, k_2): compare_block_component(m_1, m_2, k_1, k_2) for k_1, k_2 in block_param_names
    }

    # Combine output into single dictionary
    output = {**indexed_param_output, **singleton_param_output, **block_param_output}

    return output


def check_variables(m_1, m_2):
    """Check variable values"""

    block_variable_names = [
        ('p', 'V_PRIMAL_GENERATOR_POWER'),
        ('p_H', 'V_PRIMAL_HVDC_FLOW'),
        ('theta', 'V_PRIMAL_VOLTAGE_ANGLE'),
    ]

    singleton_variable_names = [
        ('tau', 'V_DUAL_PERMIT_MARKET'),
    ]

    block_variable_output = {
        (k_1, k_2): compare_block_component(m_1, m_2, k_1, k_2) for k_1, k_2 in block_variable_names
    }

    singleton_variable_output = {
        (k_1, k_2): compare_singleton_component(m_1, m_2, k_1, k_2) for k_1, k_2 in singleton_variable_names
    }

    # Combine output into single dictionary
    output = {**block_variable_output, **singleton_variable_output}

    return output


def compare_expression(m_1, m_2, m_1_name, m_2_name):
    """Compare expression result"""

    output = {
        'difference': abs(m_1.__getattribute__(m_1_name).expr() - m_2.__getattribute__(m_2_name).expr()),
        'max_difference': abs(m_1.__getattribute__(m_1_name).expr() - m_2.__getattribute__(m_2_name).expr()),
    }

    return output


def check_expressions(m_1, m_2):
    """Check expression result"""

    expression_names = [
        ('AVERAGE_ELECTRICITY_PRICE', 'E_AVERAGE_ELECTRICITY_PRICE'),
        ('WEIGHTED_RRN_PRICE', 'E_WEIGHTED_RRN_PRICE'),
    ]

    expression_output = {
        (k_1, k_2): compare_expression(m_1, m_2, k_1, k_2) for k_1, k_2 in expression_names
    }

    return expression_output


if __name__ == '__main__':
    # Data directories for scenarios
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'data')
    scenario_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, '1_create_scenarios',
                                      'output')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'utils', 'tmp')

    # Original model
    data_1 = OrganiseData()
    model_1 = create_model(use_pu=True, variable_baseline=False, objective_type='feasibility')

    # Load scenario data
    scenarios_1 = data_1.df_scenarios
    scenarios_2 = load_scenarios(scenario_directory)

    # New model
    model_options = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_FIXED_BASELINE': 1,
            'P_POLICY_PERMIT_PRICE_TARGET': 30,
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 36,
        },
        'mode': 'feasibility',
    }

    # Get case data
    case_data = get_case_data(data_directory, scenario_directory, tmp_directory, model_options, use_cache=True)

    # Construct model
    model_2 = construct_model(case_data, use_pu=True)
    model_2 = configure_model(model_2, model_options)

    # Difference
    param_check = check_parameters(model_1, model_2)
    param_max_diff = {k: v['max_difference'] for k, v in param_check.items()}
    for k, v in param_max_diff.items():
        print(k, v)

    # Run models
    model_1 = check_bau_model()
    model_2 = solve_model(model_2)

    var_check = check_variables(model_1, model_2)
    var_max_diff = {k: v['max_difference'] for k, v in var_check.items()}
    for k, v in var_max_diff.items():
        print(k, v)

    expr_check = check_expressions(model_1, model_2)
    expr_max_diff = {k: v['max_difference'] for k, v in expr_check.items()}
    for k, v in expr_max_diff.items():
        print(k, v)

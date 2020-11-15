"""Run scenario analysis"""

import os
import pickle
import hashlib
from collections import OrderedDict

from utils.data import get_case_data
from model import construct_model, configure_model, solve_model, get_solution


def get_model_hash(params):
    """
    Hash input parameters - used to generate filename

    Parameters
    ----------
    params : dict
        Model parameters

    Returns
    -------
    model_hash : str
        String used to identify model which is based on model parameters
    """

    # Get parameters as an ordered dict - sort by keys
    params_ordered = str(OrderedDict(sorted(params.items())))

    # Convert parameter dictionary to string. Use string to generate a hash which identifies the model.
    model_hash = hashlib.sha256(str(params_ordered).encode('utf-8')).hexdigest()[:20]

    return model_hash


def check_if_solved(params, output_dir):
    """
    Check if a solution for the given case already exists

    Parameters
    ----------
    params : dict
        Model parameters

    output_dir : str
        Root output directory

    Returns
    -------
    results_exist : bool
        Indicates if models results already exist for given parameters (True=model exists already exist)
    """

    # Construct expected filename based on model parameter hash
    model_hash = get_model_hash(params)
    filename = f'{model_hash}.pickle'

    # Subdirectory in which results are located - run mode is the subdirectory's name
    directory = params['mode']

    # Check if results already exist
    if filename in os.listdir(os.path.join(output_dir, directory)):
        return True
    else:
        return False


def save_solution(solution, output_dir):
    """
    Save model results

    Parameters
    ----------
    solution : dict
        Results obtained from scenario

    output_dir : str
        Root output directory
    """

    # Construct model hash from parameters
    model_hash = get_model_hash(solution['parameters'])
    directory = solution['parameters']['mode']

    with open(os.path.join(output_dir, directory, f'{model_hash}.pickle'), 'wb') as f:
        pickle.dump(solution, f)


def run_scenario(data, params, output_dir):
    """
    Run model using specified parameters

    Parameters
    ----------
    data : dict
        Case data

    params : dict
        Model options and parameters to be applied for given scenario

    output_dir : str
        Root output directory
    """

    # Check if results already exist - skip if True
    if check_if_solved(params, output_dir):
        print('Case already solved - skipping:', params)
        return None

    # Construct model and solve
    print('Running:', params)
    m = construct_model(data, use_pu=True)
    m = configure_model(m, params)
    m = solve_model(m)

    # Extract and save results
    solution = get_solution(m, params)
    save_solution(solution, output_dir)


def run_emissions_intensity_baseline_scenarios(data_dir, scenario_dir, tmp_dir, output_dir):
    """
    Run model using different (fixed) emissions intensity baselines

    Parameters
    ----------
    data_dir : str
        Root directory containing files used to construct model cases

    scenario_dir : str
        Directory containing k-means clustered scenarios

    tmp_dir : str
        Directory used to cache results (e.g. admittance matrix) when constructing case files

    output_dir : str
        Root output directory for model results
    """

    # Model parameters
    params = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,  # Placeholder - not used when 'mode' is 'feasibility'
            'P_POLICY_FIXED_BASELINE': 1.3,
            'P_POLICY_PERMIT_PRICE_TARGET': 30,  # Placeholder - not used when 'mode' is 'feasibility'
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 20,  # Placeholder - not used when 'mode' is 'feasibility'
        },
        'mode': 'feasibility',
    }

    # Baselines to investigate
    # baselines = np.linspace(1.1, 0.9, 41)
    baselines = [1.5, 1.4]

    for i in baselines:
        # Update fixed baseline
        params['parameters']['P_POLICY_FIXED_BASELINE'] = i

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, params, use_cache=True)

        # Run scenario
        run_scenario(case_data, params, output_dir)


def get_bau_average_price(data_dir, scenario_dir, tmp_dir):
    """
    Run model with arbitrarily high baseline to obtain business-as-usual price

    Parameters
    ----------
    data_dir : str
        Root directory containing files used to construct model cases

    scenario_dir : str
        Directory containing k-means clustered scenarios

    tmp_dir : str
        Directory used to cache results (e.g. admittance matrix) when constructing case files
    """

    # Model parameters
    params = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,  # Placeholder - not used when 'mode' is 'feasibility'
            'P_POLICY_FIXED_BASELINE': 1.5,
            'P_POLICY_PERMIT_PRICE_TARGET': 30,  # Placeholder - not used when 'mode' is 'feasibility'
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 20,  # Placeholder - not used when 'mode' is 'feasibility'
        },
        'mode': 'feasibility',
    }

    # Construct data used for case
    data = get_case_data(data_dir, scenario_dir, tmp_dir, params, use_cache=True)

    # Construct model and solve
    print('Running:', params)
    m = construct_model(data, use_pu=True)
    m = configure_model(m, params)
    m = solve_model(m)

    # Extract and save results
    solution = get_solution(m, params)

    # BAU price must be scaled by 100 if use_pu=True
    bau_price = solution['solution']['E_AVERAGE_ELECTRICITY_PRICE'] * 100

    return bau_price


def run_weighted_rrn_price_targeting_scenarios(data_dir, scenario_dir, tmp_dir, output_dir):
    """
    Run model using weighted RRN price targets

    Parameters
    ----------
    data_dir : str
        Root directory containing files used to construct model cases

    scenario_dir : str
        Directory containing k-means clustered scenarios

    tmp_dir : str
        Directory used to cache results (e.g. admittance matrix) when constructing case files

    output_dir : str
        Root output directory for model results
    """

    # Model parameters
    params = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_FIXED_BASELINE': 1.3,  # Placeholder - not used when 'mode' is 'weighted_rrn_price_target'
            'P_POLICY_PERMIT_PRICE_TARGET': 30,  # Placeholder - not used when 'mode' is 'weighted_rrn_price_target'
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 20,
        },
        'mode': 'weighted_rrn_price_target',
    }

    # Get BAU price
    bau_price = get_bau_average_price(data_dir, scenario_dir, tmp_dir)

    # Price target relative to BAU price
    # relative_price_target = [1.1, 1.2, 1.3, 1.4, 0.8]
    relative_price_target = [1.1, 1.2]

    for i in relative_price_target:
        # Update RRN price target
        params['parameters']['P_POLICY_WEIGHTED_RRN_PRICE_TARGET'] = bau_price * i

        # Keep track of BAU multiplier
        params['parameters']['BAU_MULTIPLIER'] = i

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, params, use_cache=True)

        # Run scenario
        run_scenario(case_data, params, output_dir)


def run_permit_price_targeting_scenarios(data_dir, scenario_dir, tmp_dir, output_dir):
    """
    Run model using permit price targets

    Parameters
    ----------
    data_dir : str
        Root directory containing files used to construct model cases

    scenario_dir : str
        Directory containing k-means clustered scenarios

    tmp_dir : str
        Directory used to cache results (e.g. admittance matrix) when constructing case files

    output_dir : str
        Root output directory for model results
    """

    # Model parameters
    params = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_FIXED_BASELINE': 1.3,  # Placeholder - not used when 'mode' is 'permit_price_target'
            'P_POLICY_PERMIT_PRICE_TARGET': 30,
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 20,  # Placeholder - not used when 'mode' is 'permit_price_target'
        },
        'mode': 'permit_price_target',
    }

    # Permit price targets
    # permit_price_targets = [25, 50, 75, 100]
    permit_price_targets = [25, 50]

    for i in permit_price_targets:
        # Update permit price target
        params['parameters']['P_POLICY_PERMIT_PRICE_TARGET'] = i

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, params, use_cache=True)

        # Run scenario
        run_scenario(case_data, params, output_dir)


if __name__ == '__main__':
    # Data and output directories
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')
    scenario_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_create_scenarios', 'output')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'utils', 'tmp')
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Run scenarios
    run_emissions_intensity_baseline_scenarios(data_directory, scenario_directory, tmp_directory, output_directory)
    run_weighted_rrn_price_targeting_scenarios(data_directory, scenario_directory, tmp_directory, output_directory)
    run_permit_price_targeting_scenarios(data_directory, scenario_directory, tmp_directory, output_directory)

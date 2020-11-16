"""Run scenario analysis"""

import os
import pickle
import hashlib
from collections import OrderedDict

import numpy as np

from utils.data import get_case_data
from model import construct_model, configure_model, solve_model, get_solution


def get_model_hash(options):
    """
    Hash input options - used to generate filename

    Parameters
    ----------
    options : dict
        Model options and parameters

    Returns
    -------
    model_hash : str
        String used to identify model which is based on model options
    """

    # Get options as an ordered dict - sort by keys
    options_ordered = str(OrderedDict(sorted(options.items())))

    # Convert parameter dictionary to string. Use string to generate a hash which identifies the model.
    model_hash = hashlib.sha256(str(options_ordered).encode('utf-8')).hexdigest()[:20]

    return model_hash


def check_if_solved(options, output_dir):
    """
    Check if a solution for the given case already exists

    Parameters
    ----------
    options : dict
        Model options

    output_dir : str
        Root output directory

    Returns
    -------
    results_exist : bool
        Indicates if models results already exist for given parameters (True=model exists already exist)
    """

    # Construct expected filename based on model parameter hash
    model_hash = get_model_hash(options)
    filename = f'{model_hash}.pickle'

    # Subdirectory in which results are located - run mode is the subdirectory's name
    directory = options['mode']

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

    # Construct model hash from model options
    model_hash = get_model_hash(solution['options'])
    directory = solution['options']['mode']

    with open(os.path.join(output_dir, directory, f'{model_hash}.pickle'), 'wb') as f:
        pickle.dump(solution, f)


def run_scenario(data, options, output_dir):
    """
    Run model using specified parameters

    Parameters
    ----------
    data : dict
        Case data

    options : dict
        Model options and parameters to be applied for given scenario

    output_dir : str
        Root output directory
    """

    # Check if results already exist - skip if True
    if check_if_solved(options, output_dir):
        print('Case already solved - skipping:', options)
        return None

    # Construct model and solve
    print('Running:', options)
    m = construct_model(data, use_pu=True)
    m = configure_model(m, options)
    m = solve_model(m)

    # Extract and save results
    solution = get_solution(m, options)
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

    # Model options
    options = {
        'parameters': {
            'P_POLICY_FIXED_BASELINE': 1.3,
        },
        'mode': 'feasibility',
    }

    # Baselines to investigate
    baselines = np.linspace(1.1, 0.9, 41)

    for i in baselines:
        # Update fixed baseline
        options['parameters']['P_POLICY_FIXED_BASELINE'] = i

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, options, use_cache=True)

        # Run scenario
        run_scenario(case_data, options, output_dir)


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
            'P_POLICY_FIXED_BASELINE': 1.5,
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

    # Model options
    options = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 20,
        },
        'mode': 'weighted_rrn_price_target',
    }

    # Get BAU price
    bau_price = get_bau_average_price(data_dir, scenario_dir, tmp_dir)

    # Price target relative to BAU price
    relative_price_target = [1.1, 1.2, 1.3, 1.4, 0.8]

    for i in relative_price_target:
        # Update RRN price target
        options['parameters']['P_POLICY_WEIGHTED_RRN_PRICE_TARGET'] = bau_price * i

        # Keep track of BAU multiplier and BAU price
        options['parameters']['BAU_MULTIPLIER'] = i
        options['parameters']['BAU_PRICE'] = bau_price

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, options, use_cache=True)

        # Run scenario
        run_scenario(case_data, options, output_dir)


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

    # Model options
    options = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_PERMIT_PRICE_TARGET': 30,
        },
        'mode': 'permit_price_target',
    }

    # Permit price targets
    permit_price_targets = [25, 50, 75, 100]

    for i in permit_price_targets:
        # Update permit price target
        options['parameters']['P_POLICY_PERMIT_PRICE_TARGET'] = i

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, options, use_cache=True)

        # Run scenario
        run_scenario(case_data, options, output_dir)


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

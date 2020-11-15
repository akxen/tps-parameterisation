"""Run scenario analysis"""

import os
import pickle
import hashlib
from collections import OrderedDict

from utils.data import get_case_data
from model import construct_model, configure_model, solve_model, get_solution

# def run_emissions_intensity_baseline_scenarios():
#     """Run model for different emissions intensity baseline scenarios"""
#
#     # Instantiate model object
#     model = create_model(use_pu=True, variable_baseline=False, objective_type='feasibility')
#     opt.options['mip tolerances absmipgap'] = 1e-6
#     opt.options['emphasis mip'] = 1  # Emphasise feasibility
#
#     # Voltage angle difference limits
#     for angle_limit in [pi / 2, pi / 3]:
#         #     for angle_limit in [pi/2]:
#
#         # Update voltage angle difference limit
#         model.THETA_DELTA = angle_limit
#
#         # Solve model for different PHI scenarios
#         for baseline in np.linspace(1.1, 0.9, 41):
#             #         for baseline in [1.1]:
#
#             # Dictionary in which to store results
#             fixed_baseline_results = dict()
#
#             # Update baseline
#             print('Emissions intensity baseline: {0} tCO2/MWh'.format(baseline))
#             model.PHI = baseline
#
#             # Filename corresponding to case
#             filename = f'fixed_baseline_results_phi_{model.PHI.value:.3f}_angle_limit_{angle_limit:.3f}.pickle'
#
#             # Check if case should be processed
#             process_case = check_processed(filename, overwrite=False)
#             if process_case:
#                 pass
#             else:
#                 print(f'Already processed. Skipping: {filename}')
#                 continue
#
#             # Model results
#             res = opt.solve(model, keepfiles=False, tee=True, warmstart=True)
#             model.solutions.store_to(res)
#
#             # Place results in DataFrame
#             try:
#                 df = pd.DataFrame(res['Solution'][0])
#                 fixed_baseline_results = {'FIXED_BASELINE': model.PHI.value, 'results': df}
#             except:
#                 fixed_baseline_results = {'FIXED_BASELINE': model.PHI.value, 'results': 'infeasible'}
#                 print('Baseline {0} is infeasible'.format(model.PHI.value))
#
#                 # Try to print results
#             try:
#                 print(model.AVERAGE_ELECTRICITY_PRICE.display())
#             except:
#                 pass
#
#             try:
#                 print(model.tau.display())
#             except:
#                 pass
#
#             # Save results
#             with open(os.path.join(paths.output_dir, filename), 'wb') as f:
#                 pickle.dump(fixed_baseline_results, f)


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
    """Save model results"""

    # Construct model hash from parameters
    model_hash = get_model_hash(solution['parameters'])
    directory = solution['parameters']['mode']

    with open(os.path.join(output_dir, directory, f'{model_hash}.pickle'), 'wb') as f:
        pickle.dump(solution, f)


def run_scenario(data, params, output_dir):
    """Run model using specified parameters"""

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
    """Run model using different (fixed) emissions intensity baselines"""

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
    baselines = [1.5, 1.4]

    for i in baselines:
        # Update fixed baseline
        params['parameters']['P_POLICY_FIXED_BASELINE'] = i

        # Construct data used for case
        case_data = get_case_data(data_dir, scenario_dir, tmp_dir, params, use_cache=True)

        # Run scenario
        run_scenario(case_data, params, output_dir)


def run_weighted_rrn_price_targeting_scenarios():
    """Run model using weighted RRN price targets"""
    pass


def run_permit_price_targeting_scenarios():
    """Run model using permit price targets"""
    pass


if __name__ == '__main__':
    # Data directories for scenarios
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'data')
    scenario_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, '1_create_scenarios',
                                      'output')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'utils', 'tmp')
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Run scenarios
    run_emissions_intensity_baseline_scenarios(data_directory, scenario_directory, tmp_directory, output_directory)
    # run_weighted_rrn_price_targeting_scenarios()
    # run_permit_price_targeting_scenarios()

    # with open(os.path.join(output_directory, 'feasibility', '72c251483609517c9d7d.pickle'), 'rb') as f:
    #     a = pickle.load(f)
    #
    # with open(os.path.join(output_directory, 'feasibility', '5093c4acde6c7d259e14.pickle'), 'rb') as f:
    #     b = pickle.load(f)

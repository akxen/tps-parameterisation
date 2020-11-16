"""Construct plots"""

import os
import pickle

import matplotlib.pyplot as plt


def load_results(result_dir):
    """Load all results in a given directory"""

    # All files in directory
    files = [i for i in os.listdir(result_dir) if i.endswith('.pickle')]

    # Container for results
    results = []
    for i in files:
        with open(os.path.join(result_dir, i), 'rb') as f:
            results.append(pickle.load(f))

    return results


def wholesale_price_vs_baseline(feasibility_results, weighted_rrn_price_target_results):
    """Permit price as function of the emissions intensity baseline"""

    # Business-as-usual price
    bau_price = (weighted_rrn_price_target_results[0]['options']['parameters']['P_POLICY_WEIGHTED_RRN_PRICE_TARGET']
                 / weighted_rrn_price_target_results[0]['options']['parameters']['BAU_MULTIPLIER'])

    # Extract feasibility results
    x_1, y_1 = [], []
    for i in feasibility_results:
        x_1.append(i['options']['parameters']['P_POLICY_FIXED_BASELINE'])

        # Price relative to BAU case
        relative_price = (i['solution']['E_AVERAGE_ELECTRICITY_PRICE'] * 100) / bau_price
        y_1.append(relative_price)

    # Extract wholesale price targeting results
    x_2, y_2 = [], []
    for i in weighted_rrn_price_target_results:
        x_2.append(i['solution']['E_BIN_EXP_DISCRETISED_BASELINE'])
        y_2.append(i['options']['parameters']['BAU_MULTIPLIER'])

    fig, ax = plt.subplots()
    # Plot results from fixed baseline sensitivity analysis
    ax.plot(x_1, y_1)
    ax.scatter(x_1, y_1)

    # Results from price targeting analysis
    ax.scatter(x_2, y_2)

    plt.show()


def wholesale_price_vs_baseline_error(weighted_rrn_price_target_results):
    """Compare wholesale price with target"""

    # Extract permit price targeting results
    x_1, y_1 = [], []
    for i in weighted_rrn_price_target_results:
        x_1.append(i['options']['parameters']['P_POLICY_WEIGHTED_RRN_PRICE_TARGET'])

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['E_AVERAGE_ELECTRICITY_PRICE'] * 100
        y_1.append(permit_price)

    fig, ax = plt.subplots()

    # Compare realised permit price with target
    ax.scatter(x_1, y_1)

    # Add a line with slope = 1 for reference
    axes_min = min(min(x_1), min(y_1))
    axes_max = max(max(x_1), max(y_1))
    ax.plot([axes_min, axes_max], [axes_min, axes_max])

    plt.show()


def permit_price_vs_baseline(feasibility_results, permit_price_target_results):
    """Permit price as function of the emissions intensity baseline"""

    # Extract feasibility results
    x_1, y_1 = [], []
    for i in feasibility_results:
        x_1.append(i['options']['parameters']['P_POLICY_FIXED_BASELINE'])

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['V_DUAL_PERMIT_MARKET'] * 100
        y_1.append(permit_price)

    # Extract wholesale price targeting results
    x_2, y_2 = [], []
    for i in permit_price_target_results:
        x_2.append(i['solution']['E_BIN_EXP_DISCRETISED_BASELINE'])

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['V_DUAL_PERMIT_MARKET'] * 100
        y_2.append(permit_price)

    fig, ax = plt.subplots()
    # Plot results from fixed baseline sensitivity analysis
    ax.plot(x_1, y_1)
    ax.scatter(x_1, y_1)

    # Results from price targeting analysis
    ax.scatter(x_2, y_2)

    plt.show()


def permit_price_vs_baseline_error(permit_price_target_results):
    """Permit price as function of the emissions intensity baseline"""

    # Extract permit price targeting results
    x_1, y_1 = [], []
    for i in permit_price_target_results:
        x_1.append(i['options']['parameters']['P_POLICY_PERMIT_PRICE_TARGET'])

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['V_DUAL_PERMIT_MARKET'] * 100
        y_1.append(permit_price)

    fig, ax = plt.subplots()

    # Compare realised permit price with target
    ax.scatter(x_1, y_1)

    # Add a line with slope = 1 for reference
    axes_min = min(min(x_1), min(y_1))
    axes_max = max(max(x_1), max(y_1))
    ax.plot([axes_min, axes_max], [axes_min, axes_max])

    plt.show()


def weighted_rrn_price_vs_average_price(feasibility_results):
    """Compare weighted RRN prices with average prices"""

    # Find business-as-usual price - identify average price corresponding to highest baseline
    baseline = 0
    bau_price = 9999
    for i in feasibility_results:
        if i['options']['parameters']['P_POLICY_FIXED_BASELINE'] > baseline:
            baseline = i['options']['parameters']['P_POLICY_FIXED_BASELINE']
            bau_price = i['solution']['E_AVERAGE_ELECTRICITY_PRICE']

    # Extract permit price targeting results
    x_1, y_1 = [], []
    for i in feasibility_results:
        average_price = i['solution']['E_AVERAGE_ELECTRICITY_PRICE'] / bau_price
        x_1.append(average_price)

        # Permit price - must multiply by 100 to correct for per unit scaling
        weighted_rrn_price = i['solution']['E_WEIGHTED_RRN_PRICE'] / bau_price
        y_1.append(weighted_rrn_price)

    fig, ax = plt.subplots()

    # Compare realised permit price with target
    ax.scatter(x_1, y_1)

    # Add a line with slope = 1 for reference
    axes_min = min(min(x_1), min(y_1))
    axes_max = max(max(x_1), max(y_1))
    ax.plot([axes_min, axes_max], [axes_min, axes_max])

    plt.show()


if __name__ == '__main__':
    # Directories containing model output
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_parameter_selector', 'output')
    feasibility_results_directory = os.path.join(output_directory, 'feasibility')
    weighted_rrn_price_target_results_directory = os.path.join(output_directory, 'weighted_rrn_price_target')
    permit_price_target_results_directory = os.path.join(output_directory, 'permit_price_target')

    # Load results
    feasibility_res = load_results(feasibility_results_directory)
    weighted_rrn_price_target_res = load_results(weighted_rrn_price_target_results_directory)
    permit_price_target_res = load_results(permit_price_target_results_directory)

    # Create plots
    wholesale_price_vs_baseline(feasibility_res, weighted_rrn_price_target_res)
    wholesale_price_vs_baseline_error(weighted_rrn_price_target_res)
    permit_price_vs_baseline(feasibility_res, permit_price_target_res)
    permit_price_vs_baseline_error(permit_price_target_res)
    weighted_rrn_price_vs_average_price(feasibility_res)

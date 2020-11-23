"""Construct plots"""

import os
import pickle

import pandas as pd
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


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


def wholesale_price_vs_baseline(ax, feasibility_results, weighted_rrn_price_target_results):
    """Permit price as function of the emissions intensity baseline"""

    # Business-as-usual price
    bau_price = weighted_rrn_price_target_results[0]['options']['parameters']['BAU_PRICE']

    # Extract feasibility results
    feasibility_values = []
    for i in feasibility_results:
        x_1 = i['options']['parameters']['P_POLICY_FIXED_BASELINE']

        # Price relative to BAU case
        relative_price = (i['solution']['E_AVERAGE_ELECTRICITY_PRICE'] * 100) / bau_price
        y_1 = relative_price
        feasibility_values.append((x_1, y_1))

    # Sort values
    sorted_feasibility_result_values = sorted(feasibility_values, key=lambda x: x[0])
    x_1 = [i[0] for i in sorted_feasibility_result_values]
    y_1 = [i[1] for i in sorted_feasibility_result_values]

    # Extract wholesale price targeting results
    x_2, y_2 = [], []
    for i in weighted_rrn_price_target_results:
        x_2.append(i['solution']['E_BIN_EXP_DISCRETISED_BASELINE'])
        y_2.append((i['solution']['E_AVERAGE_ELECTRICITY_PRICE'] * 100) / bau_price)

    # Plot results from fixed baseline sensitivity analysis
    ax.plot(x_1, y_1, linewidth=1.2, color='#24a585', alpha=0.8)
    ax.scatter(x_1, y_1, s=4, color='#24a585', alpha=0.8)

    # Results from price targeting analysis
    ax.scatter(x_2, y_2, s=30, color='#e81526', marker='+', zorder=3, alpha=0.8)

    # BAU multipliers
    multipliers = sorted(
        list(set([i['options']['parameters']['BAU_MULTIPLIER'] for i in weighted_rrn_price_target_results]))
    )

    for i in multipliers:
        ax.plot([0.9, 1.1], [i, i], linestyle=':', linewidth=0.5, color='k')

    # Add label to horizontal lines specifying average price target
    fontsize = 8
    labelsize = 7
    ax.text(1.075, 1.405, '$\hat{\lambda} = 1.4$', fontsize=5)
    ax.text(1.075, 1.305, '$\hat{\lambda} = 1.3$', fontsize=5)
    ax.text(1.075, 1.205, '$\hat{\lambda} = 1.2$', fontsize=5)
    ax.text(1.075, 1.105, '$\hat{\lambda} = 1.1$', fontsize=5)
    ax.text(1.075, 0.805, '$\hat{\lambda} = 0.8$', fontsize=5)

    # Format axes
    ax.set_ylabel('Average price \n relative to BAU', fontsize=fontsize)
    ax.set_xlabel('Emissions intensity baseline (tCO$_{2}$/MWh)\n(a)', fontsize=fontsize)

    # Format ticks
    ax.minorticks_on()
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    # Set axis limits
    ax.set_ylim(0.75, 1.6)
    ax.set_xlim(0.89, 1.11)

    return ax


def wholesale_price_vs_baseline_error(ax, weighted_rrn_price_target_results):
    """Compare wholesale price with target"""

    # Business-as-usual price
    bau_price = weighted_rrn_price_target_results[0]['options']['parameters']['BAU_PRICE']

    # Extract permit price targeting results
    x_1, y_1 = [], []
    for i in weighted_rrn_price_target_results:
        x_1.append(i['options']['parameters']['P_POLICY_WEIGHTED_RRN_PRICE_TARGET'] / bau_price)

        # Average price - must multiply by 100 to correct for per unit scaling
        average_price = i['solution']['E_AVERAGE_ELECTRICITY_PRICE'] * 100
        y_1.append(average_price / bau_price)

    # Compare realised average price with target
    ax.scatter(x_1, y_1, marker='+', color='#e81526', alpha=0.8)

    # Add a line with slope = 1 for reference
    axes_min = min(min(x_1), min(y_1))
    axes_max = max(max(x_1), max(y_1))
    ax.plot([axes_min, axes_max], [axes_min, axes_max], linestyle=':', linewidth=0.8, color='black', alpha=0.8)

    fontsize = 8
    labelsize = 7
    ax.set_xlabel('Target price relative to BAU\n(b)', fontsize=fontsize)
    ax.set_ylabel('Average price\nrelative to BAU', fontsize=fontsize)

    # Format ticks
    ax.minorticks_on()
    ax.tick_params(axis='x', labelsize=8)

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    return ax


def permit_price_vs_baseline(ax, feasibility_results, permit_price_target_results):
    """Permit price as function of the emissions intensity baseline"""

    # Extract feasibility results
    feasibility_result_values = []
    for i in feasibility_results:
        x_1 = i['options']['parameters']['P_POLICY_FIXED_BASELINE']

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['V_DUAL_PERMIT_MARKET'] * 100
        y_1 = permit_price

        feasibility_result_values.append((x_1, y_1))

    # Sort values
    sorted_feasibility_result_values = sorted(feasibility_result_values, key=lambda x: x[0])
    sorted_x_1 = [i[0] for i in sorted_feasibility_result_values]
    sorted_y_1 = [i[1] for i in sorted_feasibility_result_values]

    # Extract wholesale price targeting results
    x_2, y_2 = [], []
    for i in permit_price_target_results:
        x_2.append(i['solution']['E_BIN_EXP_DISCRETISED_BASELINE'])

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['V_DUAL_PERMIT_MARKET'] * 100
        y_2.append(permit_price)

    # Plot results from fixed baseline sensitivity analysis
    ax.plot(sorted_x_1, sorted_y_1, color='#db1313', alpha=0.8, linewidth=1.2)
    ax.scatter(sorted_x_1, sorted_y_1, color='#db1313', s=4, alpha=0.8)

    # Results from price targeting analysis
    ax.scatter(x_2, y_2, color='blue', marker='2', zorder=3, s=30)

    # Permit price targets
    price_targets = sorted(
        list(set([i['options']['parameters']['P_POLICY_PERMIT_PRICE_TARGET'] for i in permit_price_target_results]))
    )

    for i in price_targets:
        ax.plot([0.9, 1.1], [i, i], linestyle=':', linewidth=0.5, color='k')

    # Axes limits
    ax.set_ylim([-5, 130])

    # Set axes labels
    fontsize = 8
    labelsize = 7
    ax.set_ylabel('Permit price \n (\$/tCO$_{2}$)', fontsize=fontsize)
    ax.set_xlabel('Emissions intensity baseline (tCO$_{2}$/MWh)\n(c)', fontsize=fontsize)

    # Format ticks
    ax.minorticks_on()
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))

    # Add permit price target labels to horizontal lines
    ax.text(1.077, 100.5, r'$\hat{\tau} = 100$', fontsize=5)
    ax.text(1.077, 75.5, r'$\hat{\tau} = 75$', fontsize=5)
    ax.text(1.077, 50.5, r'$\hat{\tau} = 50$', fontsize=5)
    ax.text(1.077, 25.5, r'$\hat{\tau} = 25$', fontsize=5)

    return ax


def permit_price_vs_baseline_error(ax, permit_price_target_results):
    """Permit price as function of the emissions intensity baseline"""

    # Extract permit price targeting results
    x_1, y_1 = [], []
    for i in permit_price_target_results:
        x_1.append(i['options']['parameters']['P_POLICY_PERMIT_PRICE_TARGET'])

        # Permit price - must multiply by 100 to correct for per unit scaling
        permit_price = i['solution']['V_DUAL_PERMIT_MARKET'] * 100
        y_1.append(permit_price)

    # Compare realised permit price with target
    ax.scatter(x_1, y_1, marker='2', color='blue', alpha=0.8)

    # Add a line with slope = 1 for reference
    ax.plot([0, 110], [0, 110], linestyle=':', linewidth=0.8, color='black', alpha=0.8)

    # Format labels
    fontsize = 8
    ax.set_ylabel('Permit price\n(\$/tCO$_{2}$)', fontsize=fontsize, labelpad=0.5)
    ax.set_xlabel('Target permit price (\$/tCO$_{2}$)\n(d)', fontsize=fontsize)

    # Axes limits
    ax.set_ylim([-5, 120])
    ax.set_xlim([-5, 120])

    # Format ticks
    labelsize = 7
    ax.minorticks_on()
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    # Format ticks
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    return ax


def plot_weighted_rrn_price_vs_average_price(feasibility_results, output_dir):
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

    # Initialise figure
    fig, ax = plt.subplots()

    # Compare realised permit price with target
    ax.scatter(x_1, y_1, marker='o', color='#ce0037', alpha=0.5, s=7, zorder=3)

    # Add a line with slope = 1 for reference
    axes_min = min(min(x_1), min(y_1))
    axes_max = max(max(x_1), max(y_1))
    ax.plot([axes_min, axes_max], [axes_min, axes_max], linestyle=':', color='k', alpha=0.5)

    # Axes labels
    fontsize = 8
    ax.set_xlabel('Average price relative to BAU', fontsize=fontsize)
    ax.set_ylabel('Weighted RRN price\nrelative to BAU', fontsize=fontsize)

    # Format ticks
    labelsize = 7
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    # Axes limits
    ax.set_ylim([0.8, 1.6])
    ax.set_xlim([0.8, 1.6])

    # Tick locations
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

    # Set figure size
    fig.set_size_inches(2.9, 2.9 / 1.7)
    fig.subplots_adjust(left=0.22, bottom=0.23, right=0.97, top=0.97)

    fig.savefig(os.path.join(output_dir, 'figures', 'manuscript', 'weighted_rrn_prices.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'figures', 'manuscript', 'weighted_rrn_prices.pdf'))

    plt.show()


def plot_price_targeting_results(feasibility_results, weighted_rrn_price_target_results, permit_price_target_results,
                                 output_dir):
    """Plot price targeting results"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1 = wholesale_price_vs_baseline(ax1, feasibility_results, weighted_rrn_price_target_results)
    ax2 = wholesale_price_vs_baseline_error(ax2, weighted_rrn_price_target_results)
    ax3 = permit_price_vs_baseline(ax3, feasibility_results, permit_price_target_results)
    ax4 = permit_price_vs_baseline_error(ax4, permit_price_target_res)

    # Set figure size
    fig.set_size_inches(6.45, 6.45 / 1.4)
    fig.subplots_adjust(left=0.1, bottom=0.125, right=0.98, top=0.985, wspace=0.25, hspace=0.33)

    fig.savefig(os.path.join(output_dir, 'figures', 'manuscript', 'price_targets.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'figures', 'manuscript', 'price_targets.pdf'))

    plt.show()


def srmc_comparison(ax, feasibility_results, data_dir):
    """Plot SRMCs as a function of the emissions intensity baseline"""

    # Extract cost parameters
    cost_parameters = [
        {
            'duid': k,
            'srmc': i['solution']['P_GENERATOR_SRMC'][k] * 100,
            'emissions_intensity': i['solution']['P_GENERATOR_EMISSIONS_INTENSITY'][k],
            'baseline': i['options']['parameters']['P_POLICY_FIXED_BASELINE'],
            'permit_price': i['solution']['V_DUAL_PERMIT_MARKET'] * 100,
        }
        for i in feasibility_results for k in i['solution']['P_GENERATOR_SRMC'].keys()]

    # Combine into single DataFrame
    df = pd.DataFrame(cost_parameters)

    # Compute net liability faced by each generator
    df['net_liability'] = (df['emissions_intensity'] - df['baseline']) * df['permit_price']

    # Compute new SRMC
    df['new_srmc'] = df['srmc'] + df['net_liability']
    generators = pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset', 'generators', 'generators.csv'),
                             index_col='DUID')

    # Add fuel type
    df = pd.merge(df, generators[['FUEL_TYPE']], left_on='duid', right_index=True)

    # Aggregate SRMC statistics - mean, standard deviation for each fuel type and baseline
    agg = (df.groupby(['baseline', 'FUEL_TYPE'])['new_srmc'].apply(lambda x: {'mean': x.mean(), 'std': x.std()})
           .unstack())

    # Compute upper and lower bands to be plotted (mean +/- standard deviation)
    agg['upper_band'] = agg['mean'] + agg['std']
    agg['lower_band'] = agg['mean'] - agg['std']

    # Sort index
    agg = agg.sort_index(ascending=True)

    # Extract values for brown coal generators
    brown_mean = agg.loc[(slice(None), 'Brown coal', slice(None)), 'mean'].tolist()
    brown_upper = agg.loc[(slice(None), 'Brown coal', slice(None)), 'upper_band'].tolist()
    brown_lower = agg.loc[(slice(None), 'Brown coal', slice(None)), 'lower_band'].tolist()

    # Extract values for black coal generators
    black_mean = agg.loc[(slice(None), 'Black coal', slice(None)), 'mean'].tolist()
    black_upper = agg.loc[(slice(None), 'Black coal', slice(None)), 'upper_band'].tolist()
    black_lower = agg.loc[(slice(None), 'Black coal', slice(None)), 'lower_band'].tolist()

    # Extract values for gas generators
    gas_mean = agg.loc[(slice(None), 'Natural Gas (Pipeline)', slice(None)), 'mean'].tolist()
    gas_upper = agg.loc[(slice(None), 'Natural Gas (Pipeline)', slice(None)), 'upper_band'].tolist()
    gas_lower = agg.loc[(slice(None), 'Natural Gas (Pipeline)', slice(None)), 'lower_band'].tolist()

    # Baselines - to be usd on x axis
    baselines = agg.index.levels[0].to_list()

    black_coal_colour = '#0071b2'
    brown_coal_colour = '#ce0037'
    gas_colour = '#039642'

    # Plot brown coal SRMCs
    ax.plot(baselines, brown_mean, color=brown_coal_colour, alpha=0.6, linestyle=':', linewidth=0.9)
    ax.plot(baselines, brown_upper, color=brown_coal_colour, alpha=0.6, linewidth=0.9)
    ax.plot(baselines, brown_lower, color=brown_coal_colour, alpha=0.6, linewidth=0.9)
    ax.fill_between(baselines, brown_upper, brown_lower, color=brown_coal_colour, alpha=0.5)

    # Black coal SRMCs
    ax.plot(baselines, black_mean, color=black_coal_colour, alpha=0.6, linestyle=':', linewidth=0.9)
    ax.plot(baselines, black_upper, color=black_coal_colour, alpha=0.6, linewidth=0.9)
    ax.plot(baselines, black_lower, color=black_coal_colour, alpha=0.6, linewidth=0.9)
    ax.fill_between(baselines, black_upper, black_lower, color=black_coal_colour, alpha=0.5)

    # Gas SRMCs
    ax.plot(baselines, gas_mean, color=gas_colour, alpha=0.6, linestyle=':', linewidth=0.9)
    ax.plot(baselines, gas_upper, color=gas_colour, alpha=0.6, linewidth=0.9)
    ax.plot(baselines, gas_lower, color=gas_colour, alpha=0.6, linewidth=0.9)
    ax.fill_between(baselines, gas_upper, gas_lower, color=gas_colour, alpha=0.5)

    # Axes labels
    fontsize = 8
    labelsize = 7
    ax.set_xlabel('Emissions intensity baseline (tCO$_{2}$/MWh)\n(a)', fontsize=fontsize)
    ax.set_ylabel('SRMC (\$/MWh)', labelpad=-0.1, fontsize=fontsize)

    # Add legend
    brown_coal_patch = mpatches.Patch(color='#ce0037', label='Brown coal', alpha=0.5)
    black_coal_patch = mpatches.Patch(color='#112fd9', label='Black coal', alpha=0.5)
    gas_patch = mpatches.Patch(color='#039642', label='Gas', alpha=0.5)
    ax.legend(handles=[black_coal_patch, brown_coal_patch, gas_patch], ncol=3, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, 1.09), fontsize=7)

    # Format tick labels
    ax.minorticks_on()
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    # Set axes limits
    ax.set_xlim(0.9, 1.1)

    # Set major tick locator
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))

    return ax


def generator_output_comparison(ax, feasibility_results):
    """Plot generator output as a function of the emissions intensity baseline"""

    # Generator output parameters
    output_parameters = [
        {
            'scenario': k_1,
            'duid': k_2,
            'power': i['solution']['V_PRIMAL_GENERATOR_POWER'][k_1, k_2] * 100,
            'duration': i['solution']['P_SCENARIO_DURATION'][k_1] * 8760,
            'baseline': i['options']['parameters']['P_POLICY_FIXED_BASELINE'],
        }
        for i in feasibility_results for k_1, k_2 in i['solution']['V_PRIMAL_GENERATOR_POWER'].keys()]

    # Generator power output
    df = pd.DataFrame(output_parameters)

    # Compute energy output
    df['energy'] = df['power'] * df['duration']

    generators = pd.read_csv(os.path.join(data_directory, 'egrimod-nem-dataset', 'generators', 'generators.csv'),
                             index_col='DUID')

    # Add fuel type
    df = pd.merge(df, generators[['FUEL_TYPE']], left_on='duid', right_index=True)

    # Total output by fuel type as a function of the emissions intensity baseline
    total_energy = df.groupby(['baseline', 'FUEL_TYPE'])['energy'].sum().sort_index(ascending=True)

    # Extract baseline for each scenario (x-axis)
    baselines = total_energy.index.levels[0]

    # Extract energy output values for each scenario (y-axis)
    black_coal = total_energy.loc[(slice(None), 'Black coal')].tolist()
    brown_coal = total_energy.loc[(slice(None), 'Brown coal')].tolist()
    gas = total_energy.loc[(slice(None), 'Natural Gas (Pipeline)')].tolist()

    ax.plot(baselines, black_coal, color='#0071b2', alpha=0.7, linewidth=0.9)
    ax.plot(baselines, brown_coal, color='#ce0037', alpha=0.7, linewidth=0.9)
    ax.plot(baselines, gas, color='#039642', alpha=0.7, linewidth=0.9)

    ax.scatter(baselines, black_coal, edgecolors='#0071b2', facecolors='none', s=5, alpha=0.7, linewidth=0.8)
    ax.scatter(baselines, brown_coal, edgecolors='#ce0037', facecolors='none', s=5, alpha=0.7, linewidth=0.8)
    ax.scatter(baselines, gas, edgecolors='#039642', facecolors='none', s=5, alpha=0.7, linewidth=0.8)

    # Use log scale for y-axis
    ax.set_yscale('log')

    # Construct legend
    fontsize = 8
    labelsize = 7
    ax.legend(['Black coal', 'Brown coal', 'Natural gas'], fontsize=labelsize)

    # Format axes labels
    ax.set_xlabel('Emissions intensity baseline (tCO$_{2}$/MWh)\n(b)', fontsize=fontsize)
    ax.set_ylabel('Energy output (MWh)', fontsize=fontsize)

    # Format ticks
    ax.minorticks_on()
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))

    common_line_properties = {'marker': 'o', 'markersize': 3, 'alpha': 0.7, 'markerfacecolor': 'none', 'linewidth': 0.9}
    brown_coal_line = mlines.Line2D([], [], color='#0071b2', label='Black coal', **common_line_properties)
    black_coal_line = mlines.Line2D([], [], color='#ce0037', label='Brown coal', **common_line_properties)
    gas_line = mlines.Line2D([], [], color='#039642', label='Natural gas', **common_line_properties)
    plt.legend(handles=[brown_coal_line, black_coal_line, gas_line], fontsize=labelsize)

    return ax


def plot_srmc_output_comparison(feasibility_results, data_dir, output_dir):
    """Plot generator SRMCs and output as a function of the emissions intensity baseline"""

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    srmc_comparison(ax1, feasibility_results, data_dir)
    generator_output_comparison(ax2, feasibility_results)

    # Format figure size
    fig.set_size_inches(6.45, 6.45 / 2.4)
    fig.subplots_adjust(left=0.07, bottom=0.2, right=0.99, top=0.95, wspace=0.22)

    fig.savefig(os.path.join(output_dir, 'figures', 'manuscript', f'srmc_output_comparison.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'figures', 'manuscript', f'srmc_output_comparison.pdf'))

    plt.show()


if __name__ == '__main__':
    # Directories containing model output
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')
    model_output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_parameter_selector', 'output')
    feasibility_results_directory = os.path.join(model_output_directory, 'feasibility')
    weighted_rrn_price_target_results_directory = os.path.join(model_output_directory, 'weighted_rrn_price_target')
    permit_price_target_results_directory = os.path.join(model_output_directory, 'permit_price_target')

    # Root directory where figures will be saved
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Load results
    feasibility_res = load_results(feasibility_results_directory)
    weighted_rrn_price_target_res = load_results(weighted_rrn_price_target_results_directory)
    permit_price_target_res = load_results(permit_price_target_results_directory)

    # Create plots
    plot_price_targeting_results(feasibility_res, weighted_rrn_price_target_res, permit_price_target_res,
                                 output_directory)
    plot_weighted_rrn_price_vs_average_price(feasibility_res, output_directory)
    plot_srmc_output_comparison(feasibility_res, data_directory, output_directory)

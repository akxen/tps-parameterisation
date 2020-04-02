"""Process results from Pyomo model"""

import os
import re
import pickle

import numpy as np
import pandas as pd


class ResultsProcessor:
    def __init__(self, results_dir, output_dir):
        # Paths to directories
        self.results_dir = results_dir
        self.output_dir = output_dir

    def scenario_results(self, scenario_name, voltage_angle):
        """Collate DataFrame results from individual scenarios

        Param
        -----
        paths : list
            List of strings specifying paths to pickled files containing results

        Returns
        -------
        df_o : Pandas DataFrame
            Collated results for different emissions intensity baseline
            scenarios in a single DataFrame.
        """

        # Result files
        file_paths = [f for f in os.listdir(self.results_dir) if scenario_name in f and voltage_angle in f]

        # Container for fixed emissions intensity baseline scenario results
        collated_results = list()

        for file in file_paths:
            with open(os.path.join(self.results_dir, file), 'rb') as f:
                result = pickle.load(f)

                # Check that DataFrame exists within result object
                if type(result['results']) != pd.core.frame.DataFrame:
                    raise Exception('Expected DataFrame, received: {}'.format(type(result['results'])))

                # Load DataFrame
                df = result['results']

                # Append values depending on the scenario being investigated
                # Discrete emissions intensity baseline value
                if 'PHI_DISCRETE' in result.keys():
                    df.loc['PHI_DISCRETE', 'Variable'] = [{'Value': result['PHI_DISCRETE']}]

                # Price targeting scenarios
                if 'WEIGHTED_RRN_PRICE_TARGET' in result.keys():
                    df['WEIGTHED_RRN_PRICE_TARGET'] = result['WEIGHTED_RRN_PRICE_TARGET']

                if 'WEIGHTED_RRN_PRICE_TARGET_BAU_MULTIPLE' in result.keys():
                    df['WEIGHTED_RRN_PRICE_TARGET_BAU_MULTIPLE'] = result['WEIGHTED_RRN_PRICE_TARGET_BAU_MULTIPLE']

                # Fixed baseline scenarios
                if 'FIXED_BASELINE' in result.keys():
                    df['FIXED_BASELINE'] = result['FIXED_BASELINE']

                # Permit price targeting scenario
                if 'PERMIT_PRICE_TARGET' in result.keys():
                    df['PERMIT_PRICE_TARGET'] = result['PERMIT_PRICE_TARGET']

                # Collate results
                collated_results.append(df)

        # Collated results
        df_o = pd.concat(collated_results)

        return df_o

    def formatted_results(self, scenario_name, voltage_angle):
        """Format scenario results by extracting variable and index names"""

        # Compiled scenario results
        df = self.scenario_results(scenario_name, voltage_angle)

        # Function used in apply statement to extract variable indices from DataFrame index
        def _extract_index(row):
            """Extract variable indices"""

            # Scenario inidices
            # -----------------
            # Extract from index
            scenario_indices = re.findall(r'SCENARIO\[(\d)+\]', row.name)

            # Expect either 1 or 0 values for scenario index
            if len(scenario_indices) == 1:
                scenario_index = int(scenario_indices[0])

            # Raise exception if more than one scenario index encountered
            elif len(scenario_indices) > 1:
                raise (Exception('Only expected 1 scenario index to have one element: {0}'.format(scenario_indices)))
                scenario_index = np.nan

            # If no scenario index, set to null value
            else:
                scenario_index = np.nan

            # Variable indices
            # ----------------
            # Extract all variable indices from index. Expect at most 2 indices.
            variable_indices = re.findall(r'(?<!SCENARIO)\[([\w\d\,\-\#]+)\]', row.name)

            # Empty list = no variable indices - set to null values
            if len(variable_indices) == 0:
                variable_index_1 = np.nan
                variable_index_2 = np.nan

            elif len(variable_indices) == 1:
                # Split variable indices if separated comma
                variable_indices_split = variable_indices[0].split(',')

                # If no variable indices found, set both to null values
                if len(variable_indices_split) == 0:
                    variable_index_1 = np.nan
                    variable_index_2 = np.nan

                # If only 1 variable index found
                elif len(variable_indices_split) == 1:
                    variable_index_1 = variable_indices_split[0]
                    variable_index_2 = np.nan

                # If 2 variable indices found
                elif len(variable_indices_split) == 2:
                    variable_index_1, variable_index_2 = variable_indices_split

                # Else unexpected number of variable indices encountered (more than 2). Raise exception.
                else:
                    raise (Exception(
                        'Unexpected number of variable indices encountered for {0}'.format(variable_indices_split)))

            else:
                raise (Exception('Unexpected number of index components: {0}'.format(variable_indices)))

            # Handle variable name
            if '.' not in row.name:
                if '[' in row.name:
                    variable_name = re.findall(r'(.+)\[', row.name)[0]
                else:
                    variable_name = row.name
            else:
                regex = r'\.(.+)\['
                variable_names = re.findall(regex, row.name)

                if variable_names:
                    variable_name = variable_names[0]
                else:
                    variable_name = np.nan

            return pd.Series(data={'variable_name': variable_name, 'scenario_index': scenario_index,
                                   'variable_index_1': variable_index_1, 'variable_index_2': variable_index_2})

        # DataFrame with extracted indices
        df[['variable_name', 'scenario_index', 'variable_index_1', 'variable_index_2']] = df.apply(_extract_index,
                                                                                                   axis=1)

        return df

    def get_results(self):
        """Process and save model results"""

        for v in ['angle_limit_1.047', 'angle_limit_1.571']:
            # Fixed emissions intensity baseline
            df_fixed_baseline = self.formatted_results('fixed_baseline', v)
            with open(os.path.join(self.output_dir, 'tmp', f'df_fixed_baseline_{v}.pickle'), 'wb') as f:
                pickle.dump(df_fixed_baseline, f)

            # Weighted Regional Reference Node (RRN) price targeting
            df_weighted_rrn_price_target = self.formatted_results('weighted_rrn_price_target', v)
            with open(os.path.join(self.output_dir, 'tmp', f'df_weighted_rrn_price_target_{v}.pickle'), 'wb') as f:
                pickle.dump(df_weighted_rrn_price_target, f)

            # Permit price targeting scenarios
            df_permit_price_target = self.formatted_results('permit_price_target', v)
            with open(os.path.join(self.output_dir, 'tmp', f'df_permit_price_target_{v}.pickle'), 'wb') as f:
                pickle.dump(df_permit_price_target, f)


if __name__ == '__main__':
    # # Data directory
    # data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')
    #
    # # Scenario directory
    # scenarios_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_create_scenarios', 'output')

    # Results directory
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_parameter_selector', 'kkt-approach', 'output')

    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Instantiate results processor object
    results_processor = ResultsProcessor(results_directory, output_directory)

    # Get model results
    results_processor.get_results()

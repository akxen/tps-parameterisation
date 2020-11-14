#!/usr/bin/env python
# coding: utf-8

# # Parameter Selector
# Implements bi-level optimisation model to calibrate a tradable performance standard to achieve environmental and economic objectives.
# 
# ## Import packages

# In[1]:


import os
import re
import time
import pickle
import itertools
from math import pi

import numpy as np
import pandas as pd

from pyomo.environ import *

import matplotlib.pyplot as plt
np.random.seed(10)


# ## Paths
# Paths to relevant data and output directories.

# In[2]:


class DirectoryPaths(object):
    "Paths to relevant directories"
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')
        self.scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios')
        self.output_dir = os.path.join(os.path.curdir, 'output')

paths = DirectoryPaths()


# ## Model data
# Import raw model data.

# In[3]:


class RawData(object):
    "Collect input data"
    
    def __init__(self):
        
        # Paths to directories
        DirectoryPaths.__init__(self)
        
        
        # Network data
        # ------------
        # Nodes
        self.df_n = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_nodes.csv'), index_col='NODE_ID')

        # AC edges
        self.df_e = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_edges.csv'), index_col='LINE_ID')

        # HVDC links
        self.df_hvdc_links = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

        # AC interconnector links
        self.df_ac_i_links = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

        # AC interconnector flow limits
        self.df_ac_i_limits = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')


        # Generators
        # ----------       
        # Generating unit information
        self.df_g = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'generators', 'generators.csv'), index_col='DUID', dtype={'NODE': int})
        self.df_g['SRMC_2016-17'] = self.df_g['SRMC_2016-17'].map(lambda x: x + np.random.uniform(0, 2))
        
               
        # Operating scenarios
        # -------------------
        with open(os.path.join(paths.scenarios_dir, 'output', '2_scenarios.pickle'), 'rb') as f:
            self.df_scenarios = pickle.load(f)

# Create object containing raw model data
raw_data = RawData() 


# ## Organise model data
# Format and organise data.

# In[4]:


class OrganiseData(object):
    "Organise data to be used in mathematical program"
    
    def __init__(self):
        # Load model data
        RawData.__init__(self)
        

    def get_admittance_matrix(self):
        "Construct admittance matrix for network"

        # Initialise dataframe
        df_Y = pd.DataFrame(data=0j, index=self.df_n.index, columns=self.df_n.index)

        # Off-diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, tn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']
            df_Y.loc[tn, fn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']

        # Diagonal elements
        for i in self.df_n.index:
            df_Y.loc[i, i] = - df_Y.loc[i, :].sum()

        # Add shunt susceptance to diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, fn] += (row['B_PU'] / 2) * row['NUM_LINES']
            df_Y.loc[tn, tn] += (row['B_PU'] / 2) * row['NUM_LINES']

        return df_Y
    
    
    def get_HVDC_incidence_matrix(self):
        "Incidence matrix for HVDC links"
        
        # Incidence matrix for HVDC links
        df = pd.DataFrame(index=self.df_n.index, columns=self.df_hvdc_links.index, data=0)

        for index, row in self.df_hvdc_links.iterrows():
            # From nodes assigned a value of 1
            df.loc[row['FROM_NODE'], index] = 1

            # To nodes assigned a value of -1
            df.loc[row['TO_NODE'], index] = -1
        
        return df
    
    
    def get_all_ac_edges(self):
        "Tuples defining from and to nodes for all AC edges (forward and reverse)"
        
        # Set of all AC edges
        edge_set = set()
        
        # Loop through edges, add forward and reverse direction indice tuples to set
        for index, row in model_data.df_e.iterrows():
            edge_set.add((row['FROM_NODE'], row['TO_NODE']))
            edge_set.add((row['TO_NODE'], row['FROM_NODE']))
        
        return edge_set
    
    def get_network_graph(self):
        "Graph containing connections between all network nodes"
        network_graph = {n: set() for n in model_data.df_n.index}

        for index, row in model_data.df_e.iterrows():
            network_graph[row['FROM_NODE']].add(row['TO_NODE'])
            network_graph[row['TO_NODE']].add(row['FROM_NODE'])
        
        return network_graph
    
    
    def get_all_dispatchable_fossil_generator_duids(self):
        "Fossil dispatch generator DUIDs"
        
        # Filter - keeping only fossil and scheduled generators
        mask = (model_data.df_g['FUEL_CAT'] == 'Fossil') & (model_data.df_g['SCHEDULE_TYPE'] == 'SCHEDULED')
        
        return model_data.df_g[mask].index    
    
    
    def get_intermittent_dispatch(self):
        "Dispatch from intermittent generators (solar, wind)"
        
        # Intermittent generator DUIDs
        intermittent_duids_mask = model_data.df_g['FUEL_CAT'].isin(['Wind', 'Solar'])
        intermittent_duids = model_data.df_g.loc[intermittent_duids_mask].index

        # Intermittent dispatch aggregated by node
        intermittent_dispatch =(model_data.df_dispatch.reindex(columns=intermittent_duids, fill_value=0)
                                .T
                                .join(model_data.df_g[['NODE']])
                                .groupby('NODE').sum()
                                .reindex(index=model_data.df_n.index, fill_value=0)
                                .T)
        
        # Make sure columns are of type datetime
        intermittent_dispatch.index = intermittent_dispatch.index.astype('datetime64[ns]')
        
        return intermittent_dispatch
    
    
    def get_hydro_dispatch(self):
        "Dispatch from hydro plant"
        
        # Dispatch from hydro plant
        hydro_duids_mask = self.df_g['FUEL_CAT'].isin(['Hydro'])
        hydro_duids = self.df_g.loc[hydro_duids_mask].index

        # Hydro plant dispatch aggregated by node
        hydro_dispatch = (self.df_dispatch.reindex(columns=hydro_duids, fill_value=0)
                          .T
                          .join(model_data.df_g[['NODE']])
                          .groupby('NODE').sum()
                          .reindex(index=self.df_n.index, fill_value=0)
                          .T)
        
        # Make sure columns are of type datetime
        hydro_dispatch.index = hydro_dispatch.index.astype('datetime64[ns]')
        
        return hydro_dispatch
    
    
    def get_reference_nodes(self):
        "Get reference node IDs"
        
        # Filter Regional Reference Nodes (RRNs) in Tasmania and Victoria.
        mask = (model_data.df_n['RRN'] == 1) & (model_data.df_n['NEM_REGION'].isin(['TAS1', 'VIC1']))
        reference_node_ids = model_data.df_n[mask].index
        
        return reference_node_ids
    
    
    def get_node_demand(self):   
        "Compute demand at each node for a given time period, t"

        def _node_demand(row):
            # NEM region for a given node
            region = row['NEM_REGION']

            # Load at node
            demand = self.df_load.loc[:, region] * row['PROP_REG_D']

            return demand
        node_demand = self.df_n.apply(_node_demand, axis=1).T
        
        return node_demand
    
    
    def get_generator_node_map(self, generators):
        "Get set of generators connected to each node"
        generator_node_map = (self.df_g.reindex(index=generators)
                              .reset_index()
                              .rename(columns={'OMEGA_G': 'DUID'})
                              .groupby('NODE').agg(lambda x: set(x))['DUID']
                              .reindex(self.df_n.index, fill_value=set()))
        
        return generator_node_map
    
    
    def get_ac_interconnector_branches(self):
        "Get all AC interconnector branches - check that flow directions for each branch are correct"

        # Check that from and to regions conform with regional power flow limit directions
        def check_flow_direction(row):
            if (row['FROM_REGION'] == self.df_ac_i_limits.loc[row.name, 'FROM_REGION']) & (row['TO_REGION'] == model_data.df_ac_i_limits.loc[row.name, 'TO_REGION']):
                return True
            else:
                return False
        # Flow directions are consistent between link and limit DataFrames if True
        flow_directions_conform = self.df_ac_i_links.apply(check_flow_direction, axis=1).all()
        if flow_directions_conform:
            print('Flow directions conform with regional flow limit directions: {0}'.format(flow_directions_conform))
        else:
            raise(Exception('Link flow directions inconsitent with regional flow forward limit definition'))

        # Forward links
        forward_links = self.df_ac_i_links.apply(lambda x: pd.Series({'INTERCONNECTOR_ID': '-'.join([x.name, 'FORWARD']), 'BRANCH': (x['FROM_NODE'], x['TO_NODE'])}), axis=1).set_index('INTERCONNECTOR_ID')
        
        # Reverse links
        reverse_links = self.df_ac_i_links.apply(lambda x: pd.Series({'INTERCONNECTOR_ID': '-'.join([x.name, 'REVERSE']), 'BRANCH': (x['TO_NODE'], x['FROM_NODE'])}), axis=1).set_index('INTERCONNECTOR_ID')
        
        # Combine forward and reverse links
        df = pd.concat([forward_links, reverse_links]).reset_index()
        
        # Construct branch ID
        df['BRANCH_ID'] = df.apply(lambda x: '_'.join(['L', str(x.name + 1)]), axis=1)
        df.set_index('BRANCH_ID', inplace=True)
        
        return df
    
    
    def get_ac_interconnector_flow_limits(self):
        "Get aggregate flow limits for each interconnector direction (both forward and reverse)"
        
        # Forward limits
        forward_limits = self.df_ac_i_limits.apply(lambda x: pd.Series({'LIMIT': x['FORWARD_LIMIT_MW'], 'INTERCONNECTOR_ID': '-'.join([x.name, 'FORWARD'])}), axis=1).set_index('INTERCONNECTOR_ID')

        # Reverse limits
        reverse_limits = self.df_ac_i_limits.apply(lambda x: pd.Series({'LIMIT': x['REVERSE_LIMIT_MW'], 'INTERCONNECTOR_ID': '-'.join([x.name, 'REVERSE'])}), axis=1).set_index('INTERCONNECTOR_ID')

        # Combine forward and reverse limits
        interconnector_limits = pd.concat([forward_limits, reverse_limits])
        
        return interconnector_limits
    
    
    def get_ac_interconnector_branch_ids(self):
        "Get branch IDs that consitute each interconnector"
        
        # Branch IDs for each interconnector
        df = self.get_ac_interconnector_branches().reset_index().groupby('INTERCONNECTOR_ID').apply(lambda x: list(x['BRANCH_ID']))
        
        return df
    
    
    def get_ac_interconnector_branch_node_incidence_matrix(self):
        "Incidence matrix showing if AC interconnector branch is defined as (+) or (-) flow for each node"
        
        # Branches constituting AC interconnectors
        interconnector_branches = self.get_ac_interconnector_branches()

        # Initialise interconnector branch - node incidence matrix
        df = pd.DataFrame(index=interconnector_branches.index, columns=self.df_n.index, data=0)

        for index, row in df.iterrows():
            # Branch from node
            from_node = interconnector_branches.loc[index, 'BRANCH'][0]

            # Branch to node
            to_node = interconnector_branches.loc[index, 'BRANCH'][1]

            # Update values in matrix
            df.loc[index, from_node] = 1
            df.loc[index, to_node] = -1

        return df.T

# Create object containing organised model data
model_data = OrganiseData()


# ## Model

# In[5]:


def create_model(use_pu=None, variable_baseline=None, objective_type=None):
    """Create TPS baseline selection model
    
    Parameters
    ----------
    use_pu : bool
        Define if per-unit normalisation should be used. Re-scales parameters by system base power.
    
    variable_baseline : bool
        Specify if the baseline should be treated as a variable. E.g. find baseline that delivers given objective.
        
    objective_type : str
        Options:
            'feasibility' - find a feasible solution for given baseline
            'permit_price_target' - find baseline that delivers given permit price
            'weighted_rrn_price_target' - find baseline that targets weighted regional reference node (RRN) prices
            'nodal_electricity_price_target' - find baseline that targets nodal prices
            'minimise_electricity_price' - find baseline that minimises average wholesale electricity price
            
    Returns
    -------
    model : Pyomo model object
        Model object contains constraints and objectives corresponding to the given objective type
    """
    
    # Check parameters correctly specified
    if (use_pu is None) or (variable_baseline is None):
        raise(Exception('Must specify if baseline is variable, if per-unit system should be used, and type of objective for which model should be optimised'))
    
    
    # Mapping functions
    # -----------------
    def f_1(g):
        "Given generator, g, return the index of the node to which it is connected"
        return int(model_data.df_g.loc[g, 'NODE'])

    def f_2(h):
        "Given HVDC link, h, return the index of the link's 'from' node"
        return int(model_data.df_hvdc_links.loc[h, 'FROM_NODE'])

    def f_3(h):
        "Given HVDC link, h, return the index of the link's 'to' node"
        return int(model_data.df_hvdc_links.loc[h, 'TO_NODE'])
    
    def f_4(r):
        "Given NEM region, r, return index of region's Regional Reference Node (RRN)"
        return int(model_data.df_n[model_data.df_n['RRN'] == 1].reset_index().set_index('NEM_REGION').loc[r, 'NODE_ID'])


    # Construct model
    # ---------------
    # Initialise model
    model = ConcreteModel()


    # Sets
    # ----   
    # Nodes
    model.OMEGA_N = Set(initialize=model_data.df_n.index)

    # Generators
    model.OMEGA_G = Set(initialize=model_data.get_all_dispatchable_fossil_generator_duids())

    # AC edges
    ac_edges = model_data.get_all_ac_edges()
    model.OMEGA_NM = Set(initialize=ac_edges)

    # Sets of branches for which aggregate AC interconnector limits are defined
    ac_interconnector_flow_limits = model_data.get_ac_interconnector_flow_limits()
    model.OMEGA_J = Set(initialize=ac_interconnector_flow_limits.index)

    # HVDC links
    model.OMEGA_H = Set(initialize=model_data.df_hvdc_links.index)

    # Operating scenarios
    model.OMEGA_S = Set(initialize=model_data.df_scenarios.columns)
    
    # NEM regions
    model.OMEGA_R = Set(initialize=model_data.df_n['NEM_REGION'].unique())
    
    # Branches which constitute AC interconnectors in the network
    ac_interconnector_branch_node_incidence_matrix = model_data.get_ac_interconnector_branch_node_incidence_matrix()
    model.OMEGA_L = Set(initialize=ac_interconnector_branch_node_incidence_matrix.columns)


    # Maps
    # ----
    # Generator-node map
    generator_node_map = model_data.get_generator_node_map(model.OMEGA_G)

    # Network graph
    network_graph = model_data.get_network_graph()

    # Interconnectors and the branches to from which they are constituted
    ac_interconnector_branch_ids = model_data.get_ac_interconnector_branch_ids()
    
    # From and to nodes for each interconnector branch
    ac_interconnector_branches = model_data.get_ac_interconnector_branches()


    # Parameters
    # ----------
    # System base power
    model.BASE_POWER = Param(initialize=100)
    
    # Emissions intensity baseline (fixed)
    model.PHI = Param(initialize=0.95, mutable=True)

    # Admittance matrix
    admittance_matrix = model_data.get_admittance_matrix()
    def B_RULE(model, n, m):
        admittance_matrix_element = float(np.imag(admittance_matrix.loc[n, m]))
        if use_pu:
            return admittance_matrix_element
        else:
            return model.BASE_POWER * admittance_matrix_element
    model.B = Param(model.OMEGA_NM, rule=B_RULE)

    def P_H_MAX_RULE(s, h):
        forward_flow_limit = float(model_data.df_hvdc_links.loc[h, 'FORWARD_LIMIT_MW'])
        if use_pu:
            return forward_flow_limit / model.BASE_POWER
        else:
            return forward_flow_limit
    model.P_H_MAX = Param(model.OMEGA_H, rule=P_H_MAX_RULE)

    def P_H_MIN_RULE(s, h):
        reverse_flow_limit = float(model_data.df_hvdc_links.loc[h, 'REVERSE_LIMIT_MW'])
        if use_pu:
            return - reverse_flow_limit / model.BASE_POWER
        else:
            return - reverse_flow_limit
    model.P_H_MIN = Param(model.OMEGA_H, rule=P_H_MIN_RULE)

    # Reference nodes
    reference_nodes = model_data.get_reference_nodes()
    def S_R_RULE(model, n):
        if n in reference_nodes:
            return 1
        else:
            return 0
    model.S_R = Param(model.OMEGA_N, rule=S_R_RULE)
    
    # Maximum generator output
    def P_MAX_RULE(model, g):
        registered_capacity = float(model_data.df_g.loc[g, 'REG_CAP'])
        if use_pu:
            return registered_capacity / model.BASE_POWER
        else:
            return registered_capacity
    model.P_MAX = Param(model.OMEGA_G, rule=P_MAX_RULE)

    # Minimum generator output (set to 0)
    def P_MIN_RULE(model, g):
        minimum_output = 0
        if use_pu:
            return minimum_output / model.BASE_POWER
        else:
            return minimum_output
    model.P_MIN = Param(model.OMEGA_G, rule=P_MIN_RULE)

    # Generator short-run marginal costs
    def C_RULE(model, g):
        marginal_cost = float(model_data.df_g.loc[g, 'SRMC_2016-17'])
        if use_pu:
            return marginal_cost / model.BASE_POWER
        else:
            return marginal_cost
    model.C = Param(model.OMEGA_G, rule=C_RULE)

    # Generator emissions intensities
    def E_RULE(model, g):
        return float(model_data.df_g.loc[g, 'EMISSIONS'])
    model.E = Param(model.OMEGA_G, rule=E_RULE)

    # Max voltage angle difference between connected nodes
    model.THETA_DELTA = Param(initialize=float(pi / 2), mutable=True)

    # HVDC incidence matrix
    hvdc_incidence_matrix = model_data.get_HVDC_incidence_matrix()
    def K_RULE(model, n, h):
        return float(hvdc_incidence_matrix.loc[n, h])
    model.K = Param(model.OMEGA_N, model.OMEGA_H, rule=K_RULE)    

    # AC interconnector incidence matrix
    def S_L_RULE(model, n, l):
        return float(ac_interconnector_branch_node_incidence_matrix.loc[n, l])
    model.S_L = Param(model.OMEGA_N, model.OMEGA_L, rule=S_L_RULE)

    # Aggregate AC interconnector flow limits
    ac_interconnector_flow_limits = model_data.get_ac_interconnector_flow_limits()
    def F_RULE(model, j):
        power_flow_limit = float(ac_interconnector_flow_limits.loc[j, 'LIMIT'])
        if use_pu:
            return power_flow_limit / model.BASE_POWER
        else:
            return power_flow_limit
    model.F = Param(model.OMEGA_J, rule=F_RULE)

    # Big-M parameters
    def M_11_RULE(model, g):
        return model.P_MAX[g] - model.P_MIN[g]
    model.M_11 = Param(model.OMEGA_G, rule=M_11_RULE)

    def M_12_RULE(model, g):
        bound = 1e3
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_12 = Param(model.OMEGA_G, rule=M_12_RULE)

    def M_21_RULE(model, g):
        return model.P_MAX[g] - model.P_MIN[g]
    model.M_21 = Param(model.OMEGA_G, rule=M_21_RULE)

    def M_22_RULE(model, g):
        bound = 1e3
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_22 = Param(model.OMEGA_G, rule=M_22_RULE)

    def M_31_RULE(model, n, m):
        return float(pi)
    model.M_31 = Param(model.OMEGA_NM, rule=M_31_RULE)

    def M_32_RULE(model, n, m):
        bound = 1e3
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_32 = Param(model.OMEGA_NM, rule=M_32_RULE)

    def M_41_RULE(model, j):
        if 'REVERSE' in j:
            new_index = j.replace('REVERSE', 'FORWARD')
        elif 'FORWARD' in j:
            new_index = j.replace('FORWARD', 'REVERSE')
        else:
            raise(Exception('REVERSE / FORWARD not in index name'))
        return model.F[j] + model.F[new_index]
    model.M_41 = Param(model.OMEGA_J, rule=M_41_RULE)

    def M_42_RULE(model, j):
        bound = 1e3
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_42 = Param(model.OMEGA_J, rule=M_42_RULE)

    def M_51_RULE(model, h):
        return model.P_H_MAX[h] - model.P_H_MIN[h]
    model.M_51 = Param(model.OMEGA_H, rule=M_51_RULE)

    def M_52_RULE(model, h):
        bound = 1e3
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_52 = Param(model.OMEGA_H, rule=M_52_RULE)

    def M_61_RULE(model, h):
        return model.P_H_MAX[h] - model.P_H_MIN[h]
    model.M_61 = Param(model.OMEGA_H, rule=M_61_RULE)

    def M_62_RULE(model, h):
        bound = 1e3
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_62 = Param(model.OMEGA_H, rule=M_62_RULE)

    def M_71_RULE(model):
        bound = 1e4
        if use_pu:
            return 1e4 / model.BASE_POWER
        else:
            return bound
    model.M_71 = Param(rule=M_71_RULE)

    def M_72_RULE(model):
        bound = 1e4
        if use_pu:
            return bound / model.BASE_POWER
        else:
            return bound
    model.M_72 = Param(rule=M_72_RULE)


    # Variables
    # ---------
    # Permit market constraint dual variable
    model.tau = Var()

    # Permit market constraint binary variable
    model.GAMMA_7 = Var(within=Binary)

    # Additional sets, parameters, variable, and constraints if baseline is variable
    if variable_baseline:
        # Largest integer for baseline selection parameter index
        model.U = 10

        # Set of integers used to select discretized baseline
        model.OMEGA_U = Set(initialize=range(0, model.U + 1))

        # Minimum emissions intensity baseline
        def PHI_MIN_RULE(model):
            return float(0.8)
        model.PHI_MIN = Param(rule=PHI_MIN_RULE)

        # Maximum emissions intensity baseline
        def PHI_MAX_RULE(model):
            return float(1.3)
        model.PHI_MAX = Param(rule=PHI_MAX_RULE)

        # Emissions intensity baseline increment
        model.PHI_DELTA = Param(initialize=float((model.PHI_MAX - model.PHI_MIN) / (2**model.U)))

        # Parameter equal 2^u used when selecting discrete emissions intensity baseline
        def TWO_U_RULE(model, u):
            return float(2**u)
        model.TWO_U = Param(model.OMEGA_U, rule=TWO_U_RULE)

        # Binary variables used to determine discretized emissions intensity baseline choice
        model.PSI = Var(model.OMEGA_U, within=Binary)

        # Composite variable - PSI x tau - used to linearise bi-linear term
        model.z_1 = Var(model.OMEGA_U)

        # Big-M parameter used used to make z_1=0 when PSI=0, and z_1=tau when PSI=1
        def L_1_RULE(model):
            bound = 1e3
            if use_pu:
                return bound / model.BASE_POWER
            else:
                return bound
        model.L_1 = Param(rule=L_1_RULE)

        # Big-M paramter used to make z_2=0 when PSI=0, and z_2=p when PSI=1
        def L_2_RULE(model, g):
            return model.P_MAX[g] - model.P_MIN[g]
        model.L_2 = Param(model.OMEGA_G, rule=L_2_RULE)

        # Constraints such that z_1=0 when PSI=0 and z_1 = tau when PSI=1
        def Z_1_CONSTRAINT_1_RULE(model, u):
            return 0 <= model.tau - model.z_1[u]
        model.Z_1_CONSTRAINT_1 = Constraint(model.OMEGA_U, rule=Z_1_CONSTRAINT_1_RULE)

        def Z_1_CONSTRAINT_2_RULE(model, u):
            return model.tau - model.z_1[u] <= model.L_1 * (1 - model.PSI[u])
        model.Z_1_CONSTRAINT_2 = Constraint(model.OMEGA_U, rule=Z_1_CONSTRAINT_2_RULE)

        def Z_1_CONSTRAINT_3_RULE(model, u):
            return 0 <= model.z_1[u]
        model.Z_1_CONSTRAINT_3 = Constraint(model.OMEGA_U, rule=Z_1_CONSTRAINT_3_RULE)

        def Z_1_CONSTRAINT_4_RULE(model, u):
            return model.z_1[u] <= model.L_1 * model.PSI[u]
        model.Z_1_CONSTRAINT_4 = Constraint(model.OMEGA_U, rule=Z_1_CONSTRAINT_4_RULE)

        # Discretised emissions intensity baseline value
        model.PHI_DISCRETE = Expression(expr=model.PHI_MIN + (model.PHI_DELTA * sum(model.TWO_U[u] * model.PSI[u] for u in model.OMEGA_U)))


    def SCENARIO_RULE(b, s):
        "Block of constraints describing optimality conditions for each scenario"

        # Parameters
        # ----------       
        # Fixed power injections
        def R_RULE(b, n):
            fixed_injection = float(model_data.df_scenarios.loc[('intermittent', n), s] + model_data.df_scenarios.loc[('hydro', n), s])
            
            # Remove very small fixed power injections to improve numerical conditioning
            if fixed_injection < 1:
                fixed_injection = 0
            
            if use_pu:
                return fixed_injection / model.BASE_POWER
            else:
                return fixed_injection
        b.R = Param(model.OMEGA_N, rule=R_RULE)

        # Demand
        def D_RULE(b, n):
            demand = float(model_data.df_scenarios.loc[('demand', n), s])
            
            # Remove small demand to improve numerical conditioning
            if demand < 1:
                demand = 0
                        
            if use_pu:
                return demand / model.BASE_POWER
            else:
                return demand
        b.D = Param(model.OMEGA_N, rule=D_RULE)
        
        # Proportion of total demand consumed in each region
        def ZETA_RULE(b, r):            
            # Region demand
            region_demand = float((model_data.df_scenarios.rename_axis(['level_1', 'NODE_ID'])
                                   .join(model_data.df_n[['NEM_REGION']], how='left')
                                   .reset_index()
                                   .groupby(['NEM_REGION','level_1'])
                                   .sum()
                                   .loc[(r, 'demand'), s]))

            # Total demand
            total_demand = float(model_data.df_scenarios.reset_index().groupby('level_1').sum().loc['demand', s])
            
            # Proportion of demand consumed in region
            demand_proportion = float(region_demand / total_demand)
            
            return demand_proportion
        b.ZETA = Param(model.OMEGA_R, rule=ZETA_RULE)           

        # Scenario duration
        def RHO_RULE(b):
            return float(model_data.df_scenarios.loc[('hours', 'duration'), s] / 8760)
        b.RHO = Param(rule=RHO_RULE)


        # Primal variables
        # ----------------
        # Generator output
        b.p = Var(model.OMEGA_G)

        # HVDC link flow
        b.p_H = Var(model.OMEGA_H)

        # Node voltage angle
        b.theta = Var(model.OMEGA_N)


        # Dual variables
        # --------------
        # Min power output constraint dual varaible
        b.mu_1 = Var(model.OMEGA_G)

        # Max power output constraint dual variable
        b.mu_2 = Var(model.OMEGA_G)

        # Max voltage angle difference constraint dual variable
        b.mu_3 = Var(model.OMEGA_NM)

        # AC link power flow constraint dual variable
        b.mu_4 = Var(model.OMEGA_J)

        # Min HVDC flow constraint dual variable
        b.mu_5 = Var(model.OMEGA_H)

        # Max HVDC flow constraint dual variable
        b.mu_6 = Var(model.OMEGA_H)

        # Reference node voltage angle constraint dual variable
        b.nu_1 = Var(model.OMEGA_N)

        # Node power balance constraint dual variable
        b.lamb = Var(model.OMEGA_N)


        # Binary variables
        # ----------------
        # Min power output binary variable
        b.GAMMA_1 = Var(model.OMEGA_G, within=Binary)

        # Max power output binary variable
        b.GAMMA_2 = Var(model.OMEGA_G, within=Binary)

        # Max voltage angle difference binary variable
        b.GAMMA_3 = Var(model.OMEGA_NM, within=Binary)

        # AC link power flow dual variable
        b.GAMMA_4 = Var(model.OMEGA_J, within=Binary)

        # Min HVDC flow binary variable
        b.GAMMA_5 = Var(model.OMEGA_H, within=Binary)

        # Max HVDC flow binary variable
        b.GAMMA_6 = Var(model.OMEGA_H, within=Binary)

        # Parameters and variables if emissions intensity baseline is variable
        if variable_baseline:
            # Composite variable - PSI x p - used to linearise bi-linear term
            b.z_2 = Var(model.OMEGA_U * model.OMEGA_G)

            # Constraints such that z_2=0 when PSI=0, and z_2=p when PSI=1
            def Z_2_CONSTRAINT_1_RULE(b, u, g):
                return 0 <= b.p[g] - b.z_2[u, g]
            b.Z_2_CONSTRAINT_1 = Constraint(model.OMEGA_U * model.OMEGA_G, rule=Z_2_CONSTRAINT_1_RULE)

            def Z_2_CONSTRAINT_2_RULE(b, u, g):
                return b.p[g] - b.z_2[u, g] <= model.L_2[g] * (1 - model.PSI[u])
            b.Z_2_CONSTRAINT_2 = Constraint(model.OMEGA_U * model.OMEGA_G, rule=Z_2_CONSTRAINT_2_RULE)

            def Z_2_CONSTRAINT_3_RULE(b, u, g):
                return 0 <= b.z_2[u, g]
            b.Z_2_CONSTRAINT_3 = Constraint(model.OMEGA_U * model.OMEGA_G, rule=Z_2_CONSTRAINT_3_RULE)

            def Z_2_CONSTRAINT_4_RULE(b, u, g):
                return b.z_2[u, g] <= model.L_2[g] * model.PSI[u]
            b.Z_2_CONSTRAINT_4 = Constraint(model.OMEGA_U * model.OMEGA_G, rule=Z_2_CONSTRAINT_4_RULE)


        # Constraints
        # -----------
        # First order conditions
        # If baseline is fixed
        def FOC_1_RULE(b, g):
            return (model.C[g] 
                    + ((model.E[g] - model.PHI) * model.tau) 
                    - b.mu_1[g] 
                    + b.mu_2[g] 
                    - b.lamb[f_1(g)] == 0)

        # Linearised first order condition if baseline is variable
        def FOC_1_LIN_RULE(b, g):
            return (model.C[g] 
                    + ((model.E[g] - model.PHI_MIN) * model.tau) 
                    - (model.PHI_DELTA * sum(model.TWO_U[u] * model.z_1[u] for u in model.OMEGA_U)) 
                    - b.mu_1[g] + b.mu_2[g] - b.lamb[f_1(g)] == 0)

        # Activate appropriate constraint depending on whether baseline is fixed or variable
        if variable_baseline:
            # Activate if variable
            b.FOC_1_LIN = Constraint(model.OMEGA_G, rule=FOC_1_LIN_RULE)
        else:
            # Activate if fixed
            b.FOC_1 = Constraint(model.OMEGA_G, rule=FOC_1_RULE)

        def FOC_2_RULE(b, n):
            return (sum(b.mu_3[n, m] 
                        - b.mu_3[m, n] 
                        + (b.lamb[n] * model.B[n, m]) 
                        - (b.lamb[m] * model.B[m, n]) for m in network_graph[n]) 
                    + (b.nu_1[n] * model.S_R[n]) 
                    + sum(model.B[ac_interconnector_branches.loc[l, 'BRANCH']] * b.mu_4[j] * model.S_L[n, l]
                          for j in model.OMEGA_J for l in ac_interconnector_branch_ids.loc[j]
                         ) == 0)
        b.FOC_2 = Constraint(model.OMEGA_N, rule=FOC_2_RULE)

        def FOC_3_RULE(b, h):
            return ((model.K[f_2(h), h] * b.lamb[f_2(h)]) 
                    + (model.K[f_3(h), h] * b.lamb[f_3(h)]) 
                    - b.mu_5[h] 
                    + b.mu_6[h] == 0)
        b.FOC_3 = Constraint(model.OMEGA_H, rule=FOC_3_RULE)

        def EQUALITY_CONSTRAINT_1_RULE(b, n):
            if model.S_R[n] == 1:
                return b.theta[n] == 0
            else:
                return Constraint.Skip
        b.EQUALITY_CONSTRAINT_1 = Constraint(model.OMEGA_N, rule=EQUALITY_CONSTRAINT_1_RULE)

        def EQUALITY_CONSTRAINT_2_RULE(b, n):
            return (b.D[n] 
                    - b.R[n] 
                    - sum(b.p[g] for g in generator_node_map[n]) 
                    + sum(model.B[n, m] * (b.theta[n] - b.theta[m]) for m in network_graph[n]) 
                    + sum(model.K[n, h] * b.p_H[h] for h in model.OMEGA_H) == 0)
        b.EQUALITY_CONSTRAINT_2 = Constraint(model.OMEGA_N, rule=EQUALITY_CONSTRAINT_2_RULE)

        # Linearised complementarity constraints
        # --------------------------------------
        # Min power output
        def LIN_COMP_1_1_RULE(b, g):
            return model.P_MIN[g] - b.p[g] <= 0
        b.LIN_COMP_1_1 = Constraint(model.OMEGA_G, rule=LIN_COMP_1_1_RULE)

        def LIN_COMP_1_2_RULE(b, g):
            return b.mu_1[g] >= 0
        b.LIN_COMP_1_2 = Constraint(model.OMEGA_G, rule=LIN_COMP_1_2_RULE)

        def LIN_COMP_1_3_RULE(b, g):
            return b.p[g] - model.P_MIN[g] <= b.GAMMA_1[g] * model.M_11[g]
        b.LIN_COMP_1_3 = Constraint(model.OMEGA_G, rule=LIN_COMP_1_3_RULE)

        def LIN_COMP_1_4_RULE(b, g):
            return b.mu_1[g] <= (1 - b.GAMMA_1[g]) * model.M_12[g]
        b.LIN_COMP_1_4 = Constraint(model.OMEGA_G, rule=LIN_COMP_1_4_RULE)

        # Max power output
        def LIN_COMP_2_1_RULE(b, g):
            return b.p[g] - model.P_MAX[g] <= 0
        b.LIN_COMP_2_1 = Constraint(model.OMEGA_G, rule=LIN_COMP_2_1_RULE)

        def LIN_COMP_2_2_RULE(b, g):
            return b.mu_2[g] >= 0
        b.LIN_COMP_2_2 = Constraint(model.OMEGA_G, rule=LIN_COMP_2_2_RULE)

        def LIN_COMP_2_3_RULE(b, g):
            return model.P_MAX[g] - b.p[g] <= b.GAMMA_2[g] * model.M_21[g]
        b.LIN_COMP_2_3 = Constraint(model.OMEGA_G, rule=LIN_COMP_2_3_RULE)

        def LIN_COMP_2_4_RULE(b, g):
            return b.mu_2[g] <= (1 - b.GAMMA_2[g]) * model.M_22[g]
        b.LIN_COMP_2_4 = Constraint(model.OMEGA_G, rule=LIN_COMP_2_4_RULE)

        # Max voltage angle difference between connected nodes
        def LIN_COMP_3_1_RULE(b, n, m):
            return b.theta[n] - b.theta[m] - model.THETA_DELTA <= 0
        b.LIN_COMP_3_1 = Constraint(model.OMEGA_NM, rule=LIN_COMP_3_1_RULE)

        def LIN_COMP_3_2_RULE(b, n, m):
            return b.mu_3[n, m] >= 0
        b.LIN_COMP_3_2 = Constraint(model.OMEGA_NM, rule=LIN_COMP_3_2_RULE)

        def LIN_COMP_3_3_RULE(b, n, m):
            return model.THETA_DELTA + b.theta[m] - b.theta[n] <= b.GAMMA_3[n, m] * model.M_31[n, m]
        b.LIN_COMP_3_3 = Constraint(model.OMEGA_NM, rule=LIN_COMP_3_3_RULE)

        def LIN_COMP_3_4_RULE(b, n, m):
            return b.mu_3[n, m] <= (1 - b.GAMMA_3[n, m]) * model.M_32[n, m]
        b.LIN_COMP_3_4 = Constraint(model.OMEGA_NM, rule=LIN_COMP_3_4_RULE)

        # Interconnector flow limits
        def LIN_COMP_4_1_RULE(b, j):
            branches = [ac_interconnector_branches.loc[branch_id, 'BRANCH'] for branch_id in ac_interconnector_branch_ids.loc[j]]
            return sum(model.B[n, m] * (b.theta[n] - b.theta[m]) for n, m in branches) - model.F[j] <= 0
        b.LIN_COMP_4_1 = Constraint(model.OMEGA_J, rule=LIN_COMP_4_1_RULE)

        def LIN_COMP_4_2_RULE(b, j):
            return b.mu_4[j] >= 0
        b.LIN_COMP_4_2 = Constraint(model.OMEGA_J, rule=LIN_COMP_4_2_RULE)

        def LIN_COMP_4_3_RULE(b, j):
            branches = [ac_interconnector_branches.loc[branch_id, 'BRANCH'] for branch_id in ac_interconnector_branch_ids.loc[j]]
            return model.F[j] - sum(model.B[n, m] * (b.theta[n] - b.theta[m]) for n, m in branches) <= b.GAMMA_4[j] * model.M_41[j]
        b.LIN_COMP_4_3 = Constraint(model.OMEGA_J, rule=LIN_COMP_4_3_RULE)

        def LIN_COMP_4_4_RULE(b, j):
            return b.mu_4[j] <= (1 - b.GAMMA_4[j]) * model.M_42[j]
        b.LIN_COMP_4_4 = Constraint(model.OMEGA_J, rule=LIN_COMP_4_4_RULE)

        # HVDC link flow limits
        def LIN_COMP_5_1_RULE(b, h):
            return model.P_H_MIN[h] - b.p_H[h] <= 0
        b.LIN_COMP_5_1 = Constraint(model.OMEGA_H, rule=LIN_COMP_5_1_RULE)

        def LIN_COMP_5_2_RULE(b, h):
            return b.mu_5[h] >= 0
        b.LIN_COMP_5_2 = Constraint(model.OMEGA_H, rule=LIN_COMP_5_2_RULE)

        def LIN_COMP_5_3_RULE(b, h):
            return b.p_H[h] - model.P_H_MIN[h] <= b.GAMMA_5[h] * model.M_51[h]
        b.LIN_COMP_5_3 = Constraint(model.OMEGA_H, rule=LIN_COMP_5_3_RULE)

        def LIN_COMP_5_4_RULE(b, h):
            return b.mu_5[h] <= (1 - b.GAMMA_5[h]) * model.M_52[h]
        b.LIN_COMP_5_4 = Constraint(model.OMEGA_H, rule=LIN_COMP_5_4_RULE)

        def LIN_COMP_6_1_RULE(b, h):
            return b.p_H[h] - model.P_H_MAX[h] <= 0
        b.LIN_COMP_6_1 = Constraint(model.OMEGA_H, rule=LIN_COMP_6_1_RULE)

        def LIN_COMP_6_2_RULE(b, h):
            return b.mu_6[h] >= 0
        b.LIN_COMP_6_2 = Constraint(model.OMEGA_H, rule=LIN_COMP_6_2_RULE)

        def LIN_COMP_6_3_RULE(b, h):
            return model.P_H_MAX[h] - b.p_H[h] <= b.GAMMA_6[h] * model.M_61[h]
        b.LIN_COMP_6_3 = Constraint(model.OMEGA_H, rule=LIN_COMP_6_3_RULE)

        def LIN_COMP_6_4_RULE(b, h):
            return b.mu_6[h] <= (1 - b.GAMMA_6[h]) * model.M_62[h]
        b.LIN_COMP_6_4 = Constraint(model.OMEGA_H, rule=LIN_COMP_6_4_RULE)
    model.SCENARIO = Block(model.OMEGA_S, rule=SCENARIO_RULE)

    # Permit market complementarity constraints
    def PERMIT_MARKET_1_RULE(model):
        return sum(model.SCENARIO[s].RHO * ((model.E[g] - model.PHI) * model.SCENARIO[s].p[g]) for g in model.OMEGA_G for s in model.OMEGA_S) <= 0

    def PERMIT_MARKET_1_LIN_RULE(model):
        return (sum(model.SCENARIO[s].RHO * (((model.E[g] - model.PHI_MIN) * model.SCENARIO[s].p[g]) 
                                            - (model.PHI_DELTA * sum(model.TWO_U[u] * model.SCENARIO[s].z_2[u, g] for u in model.OMEGA_U))) for g in model.OMEGA_G for s in model.OMEGA_S) <= 0)

    def PERMIT_MARKET_2_RULE(model):
        return model.tau >= 0
    model.PERMIT_MARKET_2 = Constraint(rule=PERMIT_MARKET_2_RULE)

    def PERMIT_MARKET_3_RULE(model):
        return sum(model.SCENARIO[s].RHO * ((model.PHI - model.E[g]) * model.SCENARIO[s].p[g]) for g in model.OMEGA_G for s in model.OMEGA_S) <= model.GAMMA_7 * model.M_71

    def PERMIT_MARKET_3_LIN_RULE(model):
        return (sum(model.SCENARIO[s].RHO * (((model.PHI_MIN - model.E[g]) * model.SCENARIO[s].p[g]) 
                                            + (model.PHI_DELTA * sum(model.TWO_U[u] * model.SCENARIO[s].z_2[u, g] for u in model.OMEGA_U))) for g in model.OMEGA_G for s in model.OMEGA_S) <= model.GAMMA_7 * model.M_71)

    def PERMIT_MARKET_4_RULE(model):
        return model.tau <= (1 - model.GAMMA_7) * model.M_72
    model.PERMIT_MARKET_4 = Constraint(rule=PERMIT_MARKET_4_RULE)

    # Use appropriate constraint formulations depending on whether emissions intensity baseline is variable or not
    if variable_baseline:
        model.PERMIT_MARKET_1_LIN = Constraint(rule=PERMIT_MARKET_1_LIN_RULE)
        model.PERMIT_MARKET_3_LIN = Constraint(rule=PERMIT_MARKET_3_LIN_RULE)
    else:
        model.PERMIT_MARKET_1 = Constraint(rule=PERMIT_MARKET_1_RULE)
        model.PERMIT_MARKET_3 = Constraint(rule=PERMIT_MARKET_3_RULE)


    # Expressions
    # -----------
    # Total revenue from electricity sales
    model.TOTAL_REVENUE = Expression(expr=sum(model.SCENARIO[s].RHO * model.SCENARIO[s].lamb[n] * model.SCENARIO[s].D[n] for s in model.OMEGA_S for n in model.OMEGA_N))

    # Total demand
    model.TOTAL_DEMAND = Expression(expr=sum(model.SCENARIO[s].RHO * model.SCENARIO[s].D[n] for s in model.OMEGA_S for n in model.OMEGA_N))

    # Average price
    model.AVERAGE_ELECTRICITY_PRICE = Expression(expr=model.TOTAL_REVENUE / model.TOTAL_DEMAND)

    # Weighted RRN price
    model.WEIGHTED_RRN_PRICE = Expression(expr=sum(model.SCENARIO[s].RHO * model.SCENARIO[s].ZETA[r] * model.SCENARIO[s].lamb[f_4(r)] for s in model.OMEGA_S for r in model.OMEGA_R))


    # Objective functions
    # -------------------
    # Feasibility
    if objective_type == 'feasibility':
        model.dummy = Var(bounds=(0, 1))
        model.OBJECTIVE = Objective(expr=model.dummy, sense=minimize)
        
        
    # Permit price target
    elif objective_type == 'permit_price_target':
        
        def PERMIT_PRICE_TARGET_RULE(model):
            # Permit price target [$/tCO2]
            permit_price_target = 30
            
            if use_pu:
                return permit_price_target / model.BASE_POWER
            else:
                return permit_price_target
        model.PERMIT_PRICE_TARGET = Param(initialize=PERMIT_PRICE_TARGET_RULE, mutable=True)
        
        # Dummy variables used to target given permit price
        model.PERMIT_PRICE_TARGET_X_1 = Var(within=NonNegativeReals)
        model.PERMIT_PRICE_TARGET_X_2 = Var(within=NonNegativeReals)
        
        # Constraints to minimise difference between permit price and target
        model.PERMIT_PRICE_TARGET_CONSTRAINT_1 = Constraint(expr=model.PERMIT_PRICE_TARGET_X_1 >= model.PERMIT_PRICE_TARGET - model.tau)
        model.PERMIT_PRICE_TARGET_CONSTRAINT_2 = Constraint(expr=model.PERMIT_PRICE_TARGET_X_2 >= model.tau - model.PERMIT_PRICE_TARGET)
        
        # Objective function
        model.OBJECTIVE = Objective(expr=model.PERMIT_PRICE_TARGET_X_1 + model.PERMIT_PRICE_TARGET_X_2)
        
        
    # Weighted RRN price target        
    elif objective_type == 'weighted_rrn_price_target':
        # Weighted RRN price target
        def WEIGHTED_RRN_PRICE_TARGET_RULE(model):
            # Price target [$/MWh]
            weighted_rrn_price_target = 36

            if use_pu:
                return weighted_rrn_price_target / model.BASE_POWER
            else:
                return weighted_rrn_price_target
        model.WEIGHTED_RRN_PRICE_TARGET = Param(initialize=WEIGHTED_RRN_PRICE_TARGET_RULE, mutable=True)
        
        # Dummy variables used to minimise difference between average price and price target
        model.WEIGHTED_RRN_PRICE_X_1 = Var(within=NonNegativeReals)
        model.WEIGHTED_RRN_PRICE_X_2 = Var(within=NonNegativeReals)
        
        # Constraints used to minimise difference between RRN price target and RRN price
        model.WEIGHTED_RRN_PRICE_TARGET_CONSTRAINT_1 = Constraint(expr=model.WEIGHTED_RRN_PRICE_X_1 >= model.WEIGHTED_RRN_PRICE - model.WEIGHTED_RRN_PRICE_TARGET)
        model.WEIGHTED_RRN_PRICE_TARGET_CONSTRAINT_2 = Constraint(expr=model.WEIGHTED_RRN_PRICE_X_2 >= model.WEIGHTED_RRN_PRICE_TARGET - model.WEIGHTED_RRN_PRICE)
        
        # Weighted RRN price targeting objective function
        model.OBJECTIVE = Objective(expr=model.WEIGHTED_RRN_PRICE_X_1 + model.WEIGHTED_RRN_PRICE_X_2)
        
        
    # Nodal electricity price target
    elif objective_type == 'nodal_electricity_price_target':
        def NODAL_ELECTRICITY_PRICE_TARGET_RULE(model, s, n):
            # Price target [$/MWh]
            target_nodal_price = 30
            
            if use_pu:
                return target_nodal_price / model.BASE_POWER
            else:
                return target_nodal_price
        model.NODAL_ELECTRICITY_PRICE_TARGET = Param(model.OMEGA_S, model.OMEGA_N, initialize=NODAL_ELECTRICITY_PRICE_TARGET_RULE, mutable=True)
        
        # Dummy variables used to minimise difference between nodal price and target
        model.NODAL_ELECTRICITY_PRICE_X_1 = Var(model.OMEGA_S, model.OMEGA_N, within=NonNegativeReals)
        model.NODAL_ELECTRICITY_PRICE_X_2 = Var(model.OMEGA_S, model.OMEGA_N, within=NonNegativeReals)
        
        # Constraints to minimise difference between nodal price and target
        def NODAL_ELECTRICTY_PRICE_TARGET_CONSTRAINT_1_RULE(model, s, n):
            return model.NODAL_ELECTRICITY_PRICE_X_1[s, n] >= model.SCENARIO[s].lamb[n] - model.NODAL_ELECTRICITY_PRICE_TARGET[s, n]
        model.NODAL_ELECTRICTY_PRICE_TARGET_CONSTRAINT_1 = Constraint(model.OMEGA_S, model.OMEGA_N, rule=NODAL_ELECTRICTY_PRICE_TARGET_CONSTRAINT_1_RULE)

        def NODAL_ELECTRICTY_PRICE_TARGET_CONSTRAINT_2_RULE(model, s, n):
            return model.NODAL_ELECTRICITY_PRICE_X_2[s, n] >= model.NODAL_ELECTRICITY_PRICE_TARGET[s, n] -  model.SCENARIO[s].lamb[n]
        model.NODAL_ELECTRICTY_PRICE_TARGET_CONSTRAINT_2 = Constraint(model.OMEGA_S, model.OMEGA_N, rule=NODAL_ELECTRICTY_PRICE_TARGET_CONSTRAINT_2_RULE)

        # Nodal price objective
        model.OBJECTIVE = Objective(expr=sum(model.SCENARIO[s].RHO * model.SCENARIO[s].D[n] * (model.NODAL_ELECTRICITY_PRICE_X_1[s, n] + model.NODAL_ELECTRICITY_PRICE_X_2[s, n]) for s in model.OMEGA_S for n in model.OMEGA_N))
        
        
    # Minimise average electricity price  
    elif objective_type == 'minimise_electricity_price':
        model.OBJECTIVE = Objective(expr=model.AVERAGE_ELECTRICITY_PRICE, sense=minimize)
        
        # Add constraint to put a lower-bound on the average electricity price (help solver)
        model.AVERAGE_PRICE_MIN = Param(initialize=0.3, mutable=True)
        model.AVERAGE_PRICE_LOWER_BOUND = Constraint(expr=model.AVERAGE_ELECTRICITY_PRICE >= model.AVERAGE_PRICE_MIN)

    else:
        raise(Exception('Invalid objective type'))

    return model


# Setup solver.

# In[6]:


# Setup solver
# ------------
solver = 'cplex'
solver_io = 'mps'
opt = SolverFactory(solver, solver_io=solver_io)


# Check model runs

# In[7]:


# Run BAU model (very high baseline)
model_bau = create_model(use_pu=True, variable_baseline=False, objective_type='feasibility')
model_bau.PHI = 1.5
res_bau = opt.solve(model_bau, keepfiles=False, tee=True, warmstart=True)


# Function to check if case should be processed.

# In[ ]:


def check_processed(filename, overwrite=True):
    """Check if file has been processed"""
    
    # Files that have already been processed
    files = os.listdir(paths.output_dir)
    
    if (filename in files) and not overwrite:
        # No need to process file
        return False
    else:
        # Need to process the file
        return True


# Solve model for different emissions intensity baselines.

# In[ ]:


def run_emissions_intensity_baseline_scenarios():
    "Run model for different emissions intensity baseline scenarios"
    
    # Instantiate model object
    model = create_model(use_pu=True, variable_baseline=False, objective_type='feasibility')
    opt.options['mip tolerances absmipgap'] = 1e-6
    opt.options['emphasis mip'] = 1 # Emphasise feasibility

    # Voltage angle difference limits
    for angle_limit in [pi/2, pi/3]:
#     for angle_limit in [pi/2]:
        
        # Update voltage angle difference limit
        model.THETA_DELTA = angle_limit
        
        # Solve model for different PHI scenarios
        for baseline in np.linspace(1.1, 0.9, 41):
#         for baseline in [1.1]:        

            # Dictionary in which to store results
            fixed_baseline_results = dict()

            # Update baseline
            print('Emissions intensity baseline: {0} tCO2/MWh'.format(baseline))
            model.PHI = baseline
            
            # Filename corresponding to case
            filename = f'fixed_baseline_results_phi_{model.PHI.value:.3f}_angle_limit_{angle_limit:.3f}.pickle'            
            
            # Check if case should be processed
            process_case = check_processed(filename, overwrite=False)
            if process_case:
                pass
            else:
                print(f'Already processed. Skipping: {filename}')
                continue

            # Model results
            res = opt.solve(model, keepfiles=False, tee=True, warmstart=True)
            model.solutions.store_to(res)

            # Place results in DataFrame
            try:
                df = pd.DataFrame(res['Solution'][0])
                fixed_baseline_results = {'FIXED_BASELINE': model.PHI.value, 'results': df}
            except:
                fixed_baseline_results = {'FIXED_BASELINE': model.PHI.value, 'results': 'infeasible'}
                print('Baseline {0} is infeasible'.format(model.PHI.value))   

            # Try to print results
            try:
                print(model.AVERAGE_ELECTRICITY_PRICE.display())
            except:
                pass

            try:
                print(model.tau.display())
            except:
                pass

            # Save results
            with open(os.path.join(paths.output_dir, filename), 'wb') as f:
                pickle.dump(fixed_baseline_results, f)
            
run_emissions_intensity_baseline_scenarios()


# Identify baseline that targets wholesale electricity price

# In[ ]:


def run_weighted_rrn_price_target_scenarios():
    "Run model that targets weighted RRN electricity prices"
    
    # Run BAU model (very high baseline)
    model_bau = create_model(use_pu=True, variable_baseline=False, objective_type='feasibility')
    model_bau.PHI = 1.5
       
    # Instantiate model object
    model = create_model(use_pu=True, variable_baseline=True, objective_type='weighted_rrn_price_target')
    opt.options['mip tolerances absmipgap'] = 1e-3

    # Voltage angle difference limits
    for angle_limit in [pi/2, pi/3]:
#     for angle_limit in [pi/2]:

        # Update voltage angle difference limit
        model_bau.THETA_DELTA = angle_limit
        model.THETA_DELTA = angle_limit
        
        # BAU model results
        print('Solving BAU scenario')
        res_bau = opt.solve(model_bau, keepfiles=False, tee=True, warmstart=True)    
        model_bau.solutions.store_to(res_bau)
        bau_weighted_rnn_price = model_bau.WEIGHTED_RRN_PRICE.expr()
        print('BAU weighted RNN price: {0}'.format(bau_weighted_rnn_price))
        
        # Update voltage angle difference limit
        model.THETA_DELTA = angle_limit
    
        # Weighted RNN price target as a multiple of BAU weighted RNN price
        for price_multiplier in [1.1, 1.2, 1.3, 1.4, 0.8]:
#         for price_multiplier in [1.1]:
        
            # Filename corresponding to case
            filename = f'weighted_rrn_price_target_{price_multiplier:.3f}_angle_limit_{angle_limit:.3f}.pickle'
        
            # Check if case should be processed
            process_case = check_processed(filename, overwrite=False)
            if process_case:
                pass
            else:
                print(f'Already processed. Skipping: {filename}')
                continue
        
        
            # Update price target
            model.WEIGHTED_RRN_PRICE_TARGET = price_multiplier * bau_weighted_rnn_price
            print('Target price input: {}'.format(price_multiplier * bau_weighted_rnn_price))

            # Model results
            res = opt.solve(model, keepfiles=False, tee=True, warmstart=True)
            model.solutions.store_to(res)

            # Place results in DataFrame
            try:
                df = pd.DataFrame(res['Solution'][0])
                price_target_results = {'WEIGHTED_RRN_PRICE_TARGET': model.WEIGHTED_RRN_PRICE_TARGET.value,
                                        'WEIGHTED_RRN_PRICE_TARGET_BAU_MULTIPLE': price_multiplier,
                                        'results': df,
                                        'PHI_DISCRETE': model.PHI_DISCRETE.expr()}
            except:
                price_target_results = {'WEIGHTED_RRN_PRICE_TARGET': model.WEIGHTED_RRN_PRICE_TARGET.value,
                                        'results': 'infeasible'}
                print('Weighted RNN price target {0} is infeasible'.format(model.WEIGHTED_RRN_PRICE_TARGET.value))   

            # Try to print results
            try:
                print(model.WEIGHTED_RRN_PRICE.display())
            except:
                pass

            try:
                print(model.PHI_DISCRETE.display())
            except:
                pass

            print(model.tau.display())

            # Save results
            with open(os.path.join(paths.output_dir, filename), 'wb') as f:
                pickle.dump(price_target_results, f)
           
run_weighted_rrn_price_target_scenarios()


# Identify baseline that targets equilibrium permit price

# In[ ]:


def run_permit_price_target_scenarios():
    "Run model for different permit price target scenarios"

    # Instantiate model object
    model = create_model(use_pu=True, variable_baseline=True, objective_type='permit_price_target')
    opt.options['mip tolerances absmipgap'] = 1e-3

    # Voltage angle difference limits
    for angle_limit in [pi/2, pi/3]:
#     for angle_limit in [pi/2]:
        
        # Update voltage angle difference limit
        model.THETA_DELTA = angle_limit
        
        for permit_price in [25/100, 50/100, 75/100, 100/100]:
#         for permit_price in [25/100]:
        
            # Filename
            filename = f'permit_price_target_{permit_price:.3f}_angle_limit_{angle_limit:.3f}.pickle'

            # Check if case should be processed
            process_case = check_processed(filename, overwrite=False)
            if process_case:
                pass
            else:
                print(f'Already processed. Skipping: {filename}')
                continue
        
        
            # Set permit price target
            model.PERMIT_PRICE_TARGET = permit_price

            # Model results
            res = opt.solve(model, keepfiles=False, tee=True, warmstart=True)
            model.solutions.store_to(res)

            # Place results in DataFrame
            try:
                df = pd.DataFrame(res['Solution'][0])
                permit_price_target_results = {'PERMIT_PRICE_TARGET': permit_price,
                                               'results': df,
                                               'PHI_DISCRETE': model.PHI_DISCRETE.expr()}
            except:
                permit_price_target_results = {'PERMIT_PRICE_TARGET': permit_price,
                                               'results': 'infeasible'}
                print('Permit price target {0} is infeasible'.format(permit_price))

            # Try to print results
            try:
                print(model.PHI_DISCRETE.display())
            except:
                pass

            print(model.tau.display())

            # Save results

            with open(os.path.join(paths.output_dir, filename), 'wb') as f:
                pickle.dump(permit_price_target_results, f)

run_permit_price_target_scenarios()



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


def create_model(use_pu=None, constraint_mode=None, objective_type=None):
    """Create TPS baseline selection model
    
    Parameters
    ----------
    use_pu : bool
        Define if per-unit normalisation should be used. Re-scales parameters by system base power.
    
    constraint_mode : str
        Options:
            'find_phi_find_tau' - both baseline and permit price are variable
            'fix_tau_find_phi' - find baseline that delivers a given permit price
            'fix_phi_find_tau' - find permit price that results from a given baseline
        
    objective_type : str
        Options:
            'feasibility' - find a feasible solution for given baseline
            'weighted_rrn_price_target' - find baseline that targets weighted regional reference node (RRN) prices
            'nodal_electricity_price_target' - find baseline that targets nodal prices
            'minimise_electricity_price' - find baseline that minimises average wholesale electricity price
 
    Returns
    -------
    model : Pyomo model object
        Model object contains constraints and objectives corresponding to the given objective type
    """
    
    # Check parameters correctly specified
    if (use_pu is None) or (constraint_mode is None):
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
    model.THETA_DELTA = Param(initialize=float(pi / 2))

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


    # Variables
    # ---------
    # Permit market constraint dual variable
    model.tau = Var()
    
    # Emissions intensity baseline
    model.phi = Var()    
    
    # Convex Envelopes
    # -------------------
    # Bounds for emissions intensity baseline tCO2/MWh
    model.PHI_LO = Param(initialize=0.7)
    model.PHI_UP = Param(initialize=1.5)
    
    # Permit prices
    # Note: if pu normalisation will scale tau by BASE_POWER
    model.TAU_LO = Param(initialize=0)
    model.TAU_UP = Param(initialize=1000/model.BASE_POWER)  
    
    
    # Mapping for bi-linear and tri-linear terms
    # ------------------------------------------
    # tau --> x
    model.x = Expression(expr=model.tau)
    model.X_LO = Expression(expr=model.TAU_LO)
    model.X_UP = Expression(expr=model.TAU_UP)
    
    # phi --> y
    model.y = Expression(expr=model.phi)
    model.Y_LO = Expression(expr=model.PHI_LO)
    model.Y_UP = Expression(expr=model.PHI_UP)
    
    # p --> z
    def Z_LO_RULE(model, g):
        return model.P_MIN[g]
    model.Z_LO = Expression(model.OMEGA_G, rule=Z_LO_RULE)
    
    def Z_UP_RULE(model, g):
        return model.P_MAX[g]
    model.Z_UP = Expression(model.OMEGA_G, rule=Z_UP_RULE)
    
    # Convex envelope for tau * phi <---> x * y
    model.w_1 = Var()
    model.W_1_CE_1 = Constraint(expr=model.w_1 >= (model.X_LO * model.y) + (model.x * model.Y_LO) - (model.X_LO * model.Y_LO))
    model.W_1_CE_2 = Constraint(expr=model.w_1 >= (model.X_UP * model.y) + (model.x * model.Y_UP) - (model.X_UP * model.Y_UP))
    model.W_1_CE_3 = Constraint(expr=model.w_1 <= (model.X_UP * model.y) + (model.x * model.Y_LO) - (model.X_UP * model.Y_LO))
    model.W_1_CE_4 = Constraint(expr=model.w_1 <= (model.x * model.Y_UP) + (model.X_LO * model.y) - (model.X_LO * model.Y_UP))


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
            region_demand = float((model_data.df_scenarios
                                   .join(model_data.df_n[['NEM_REGION']], how='left')
                                   .reset_index()
                                   .groupby(['NEM_REGION','level'])
                                   .sum()
                                   .loc[(r, 'demand'), s]))

            # Total demand
            total_demand = float(model_data.df_scenarios.reset_index().groupby('level').sum().loc['demand', s])
            
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
        
        # Mapping for reformulation
        # -------------------------
        def Z_RULE(b, g):
            return b.p[g]
        b.z = Expression(model.OMEGA_G, rule=Z_RULE)
        
        # McCormick Envelopes for bi-linear terms
        # ---------------------------------------        
        # Constraints used to linearise phi * p <---> y * z
        b.w_2 = Var(model.OMEGA_G)
        def W_2_BLOCK_RULE(c, g):
            c.W_2_CE_1 = Constraint(expr=b.w_2[g] >= (model.PHI_LO * b.p[g]) + (model.phi * model.P_MIN[g]) - (model.PHI_LO * model.P_MIN[g]))
            c.W_2_CE_2 = Constraint(expr=b.w_2[g] >= (model.PHI_UP * b.p[g]) + (model.phi * model.P_MAX[g]) - (model.PHI_UP * model.P_MAX[g]))
            c.W_2_CE_3 = Constraint(expr=b.w_2[g] <= (model.PHI_UP * b.p[g]) + (model.phi * model.P_MIN[g]) - (model.PHI_UP * model.P_MIN[g]))
            c.W_2_CE_4 = Constraint(expr=b.w_2[g] <= (model.phi * model.P_MAX[g]) + (model.PHI_LO * b.p[g]) - (model.PHI_LO * model.P_MAX[g]))
        b.W_2_BLOCK = Block(model.OMEGA_G, rule=W_2_BLOCK_RULE)
        
        # Constraints used to linearise tau * p <---> x * z
        b.w_3 = Var(model.OMEGA_G)
        def W_3_BLOCK_RULE(c, g):
            c.W_3_CE_1 = Constraint(expr=b.w_3[g] >= (model.TAU_LO * b.p[g]) + (model.tau * model.P_MIN[g]) - (model.TAU_LO * model.P_MIN[g]))
            c.W_3_CE_2 = Constraint(expr=b.w_3[g] >= (model.TAU_UP * b.p[g]) + (model.tau * model.P_MAX[g]) - (model.TAU_UP * model.P_MAX[g]))
            c.W_3_CE_3 = Constraint(expr=b.w_3[g] <= (model.TAU_UP * b.p[g]) + (model.tau * model.P_MIN[g]) - (model.TAU_UP * model.P_MIN[g]))
            c.W_3_CE_4 = Constraint(expr=b.w_3[g] <= (model.tau * model.P_MAX[g]) + (model.TAU_LO * b.p[g]) - (model.TAU_LO * model.P_MAX[g]))
        b.W_3_BLOCK = Block(model.OMEGA_G, rule=W_3_BLOCK_RULE)

        # Trilinear term inearlisation: phi * tau * p
        # -------------------------------------------        
        b.w_4 = Var(model.OMEGA_G)
        def W_4_BLOCK_RULE(c, g):
            
            if value((model.X_UP * model.Y_LO * model.Z_LO[g]) + (model.X_LO * model.Y_UP * model.Z_UP[g])) > value((model.X_LO * model.Y_UP * model.Z_LO[g]) + (model.X_UP * model.Y_LO * model.Z_UP[g])):
                raise Exception('Epigraph relation 1 not satisfied')
            
            if value((model.X_UP * model.Y_LO * model.Z_LO[g]) + (model.X_LO * model.Y_UP * model.Z_UP[g])) > value((model.X_UP * model.Y_UP * model.Z_LO[g]) + (model.X_LO * model.Y_LO * model.Z_UP[g])):
                raise Exception('Epigraph relation 2 not satisfied')
                            
            # Epigraph of xyx
            c.W_4_CE_1 = Constraint(expr=b.w_4[g] >= (model.Y_LO * model.Z_LO[g] * model.x) + (model.X_LO * model.Z_LO[g] * model.y) + (model.X_LO * model.Y_LO * b.z[g]) - (2 * model.X_LO * model.Y_LO * model.Z_LO[g]))
            c.W_4_CE_2 = Constraint(expr=b.w_4[g] >= (model.Y_UP * model.Z_UP[g] * model.x) + (model.X_UP * model.Z_UP[g] * model.y) + (model.X_UP * model.Y_UP * b.z[g]) - (2 * model.X_UP * model.Y_UP * model.Z_UP[g]))
            c.W_4_CE_3 = Constraint(expr=b.w_4[g] >= (model.Y_LO * model.Z_UP[g] * model.x) + (model.X_LO * model.Z_UP[g] * model.y) + (model.X_UP * model.Y_LO * b.z[g]) - (model.X_LO * model.Y_LO * model.Z_UP[g]) - (model.X_UP * model.Y_LO * model.Z_UP[g]))
            c.W_4_CE_4 = Constraint(expr=b.w_4[g] >= (model.Y_UP * model.Z_LO[g] * model.x) + (model.X_UP * model.Z_LO[g] * model.y) + (model.X_LO * model.Y_UP * b.z[g]) - (model.X_UP * model.Y_UP * model.Z_LO[g]) - (model.X_LO * model.Y_UP * model.Z_LO[g]))
            
            c.W_4_EX_1 = Expression(expr=(model.X_UP * model.Y_UP * model.Z_LO[g]) - (model.X_LO * model.Y_UP * model.Z_UP[g]) - (model.X_UP * model.Y_LO * model.Z_LO[g]) + (model.X_UP * model.Y_LO * model.Z_UP[g]))
            c.W_4_CE_5 = Constraint(expr=b.w_4[g] >= ((c.W_4_EX_1 / (model.X_UP - model.X_LO)) * model.x) + (model.X_UP * model.Z_LO[g] * model.y) + (model.X_UP * model.Y_LO * b.z[g]) + ( - ((c.W_4_EX_1 * model.X_LO) / (model.X_UP - model.X_LO)) - (model.X_UP * model.Y_UP * model.Z_LO[g]) - (model.X_UP * model.Y_LO * model.Z_UP[g]) + (model.X_LO * model.Y_UP * model.Z_UP[g]) ) )
            
            c.W_4_EX_2 = Expression(expr=(model.X_LO * model.Y_LO * model.Z_UP[g]) - (model.X_UP * model.Y_LO * model.Z_LO[g]) - (model.X_LO * model.Y_UP * model.Z_UP[g]) + (model.X_LO * model.Y_UP * model.Z_LO[g]))
            c.W_4_CE_6 = Constraint(expr=b.w_4[g] >= ((c.W_4_EX_2 / (model.X_LO - model.X_UP)) * model.x) + (model.X_LO * model.Z_UP[g] * model.y) + (model.X_LO * model.Y_UP * b.z[g]) + ( - ((c.W_4_EX_1 * model.X_UP) / (model.X_LO - model.X_UP)) - (model.X_LO * model.Y_LO * model.Z_UP[g]) - (model.X_LO * model.Y_UP * model.Z_LO[g]) + (model.X_UP * model.Y_LO * model.Z_LO[g]) ) )        
            
            # Hypograph of xyx
            c.W_4_CE_7 = Constraint(expr=b.w_4[g] <= (model.Y_LO * model.Z_LO[g] * model.x) + (model.X_UP * model.Z_LO[g] * model.y) + (model.X_UP * model.Y_UP * b.z[g]) - (model.X_UP * model.Y_UP * model.Z_LO[g]) - (model.X_UP * model.Y_LO * model.Z_LO[g]))
            c.W_4_CE_8 = Constraint(expr=b.w_4[g] <= (model.Y_UP * model.Z_LO[g] * model.x) + (model.X_LO * model.Z_LO[g] * model.y) + (model.X_UP * model.Y_UP * b.z[g]) - (model.X_UP * model.Y_UP * model.Z_LO[g]) - (model.X_LO * model.Y_UP * model.Z_LO[g]))
            c.W_4_CE_9 = Constraint(expr=b.w_4[g] <= (model.Y_LO * model.Z_LO[g] * model.x) + (model.X_UP * model.Z_UP[g] * model.y) + (model.X_UP * model.Y_LO * b.z[g]) - (model.X_UP * model.Y_LO * model.Z_UP[g]) - (model.X_UP * model.Y_LO * model.Z_LO[g]))
            c.W_4_CE_10 = Constraint(expr=b.w_4[g] <= (model.Y_UP * model.Z_UP[g] * model.x) + (model.X_LO * model.Z_LO[g] * model.y) + (model.X_LO * model.Y_UP * b.z[g]) - (model.X_LO * model.Y_UP * model.Z_UP[g]) - (model.X_LO * model.Y_UP * model.Z_LO[g]))
            c.W_4_CE_11 = Constraint(expr=b.w_4[g] <= (model.Y_LO * model.Z_UP[g] * model.x) + (model.X_UP * model.Z_UP[g] * model.y) + (model.X_LO * model.Y_LO * b.z[g]) - (model.X_UP * model.Y_LO * model.Z_UP[g]) - (model.X_LO * model.Y_LO * model.Z_UP[g]))
            c.W_4_CE_12 = Constraint(expr=b.w_4[g] <= (model.Y_UP * model.Z_UP[g] * model.x) + (model.X_LO * model.Z_UP[g] * model.y) + (model.X_LO * model.Y_LO * b.z[g]) - (model.X_LO * model.Y_UP * model.Z_UP[g]) - (model.X_LO * model.Y_LO * model.Z_UP[g]))
            
        b.W_4_BLOCK = Block(model.OMEGA_G, rule=W_4_BLOCK_RULE)

        
        # Constraints
        # -----------
        # First order conditions
        if (constraint_mode == 'fix_phi_find_tau') or (constraint_mode == 'fix_tau_find_phi'):
            def FOC_1_RULE(b, g):
                return (model.C[g] 
                        + ((model.E[g] - model.phi) * model.tau) 
                        - b.mu_1[g] 
                        + b.mu_2[g] 
                        - b.lamb[f_1(g)] == 0)
        
        elif constraint_mode == 'find_phi_find_tau':
            def FOC_1_RULE(b, g):
                return (model.C[g] 
                        + (model.E[g] * model.tau) 
                        - model.w_1
                        - b.mu_1[g] 
                        + b.mu_2[g] 
                        - b.lamb[f_1(g)] == 0)            
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

        def LIN_COMP_1_1_RULE(b, g):
            return model.P_MIN[g] - b.p[g] <= 0
        b.LIN_COMP_1_1 = Constraint(model.OMEGA_G, rule=LIN_COMP_1_1_RULE)

        def LIN_COMP_1_2_RULE(b, g):
            return b.mu_1[g] >= 0
        b.LIN_COMP_1_2 = Constraint(model.OMEGA_G, rule=LIN_COMP_1_2_RULE)

        def LIN_COMP_2_1_RULE(b, g):
            return b.p[g] - model.P_MAX[g] <= 0
        b.LIN_COMP_2_1 = Constraint(model.OMEGA_G, rule=LIN_COMP_2_1_RULE)

        def LIN_COMP_2_2_RULE(b, g):
            return b.mu_2[g] >= 0
        b.LIN_COMP_2_2 = Constraint(model.OMEGA_G, rule=LIN_COMP_2_2_RULE)

        def LIN_COMP_3_1_RULE(b, n, m):
            return b.theta[n] - b.theta[m] - model.THETA_DELTA <= 0
        b.LIN_COMP_3_1 = Constraint(model.OMEGA_NM, rule=LIN_COMP_3_1_RULE)

        def LIN_COMP_3_2_RULE(b, n, m):
            return b.mu_3[n, m] >= 0
        b.LIN_COMP_3_2 = Constraint(model.OMEGA_NM, rule=LIN_COMP_3_2_RULE)

        def LIN_COMP_4_1_RULE(b, j):
            branches = [ac_interconnector_branches.loc[branch_id, 'BRANCH'] for branch_id in ac_interconnector_branch_ids.loc[j]]
            return sum(model.B[n, m] * (b.theta[n] - b.theta[m]) for n, m in branches) - model.F[j] <= 0
        b.LIN_COMP_4_1 = Constraint(model.OMEGA_J, rule=LIN_COMP_4_1_RULE)

        def LIN_COMP_4_2_RULE(b, j):
            return b.mu_4[j] >= 0
        b.LIN_COMP_4_2 = Constraint(model.OMEGA_J, rule=LIN_COMP_4_2_RULE)

        def LIN_COMP_5_1_RULE(b, h):
            return model.P_H_MIN[h] - b.p_H[h] <= 0
        b.LIN_COMP_5_1 = Constraint(model.OMEGA_H, rule=LIN_COMP_5_1_RULE)

        def LIN_COMP_5_2_RULE(b, h):
            return b.mu_5[h] >= 0
        b.LIN_COMP_5_2 = Constraint(model.OMEGA_H, rule=LIN_COMP_5_2_RULE)

        def LIN_COMP_6_1_RULE(b, h):
            return b.p_H[h] - model.P_H_MAX[h] <= 0
        b.LIN_COMP_6_1 = Constraint(model.OMEGA_H, rule=LIN_COMP_6_1_RULE)

        def LIN_COMP_6_2_RULE(b, h):
            return b.mu_6[h] >= 0
        b.LIN_COMP_6_2 = Constraint(model.OMEGA_H, rule=LIN_COMP_6_2_RULE)
        
        # Strong duality constraint
        # -------------------------
        # Primal objective
        if constraint_mode == 'fix_phi_find_tau':
            b.PRIMAL_OBJECTIVE = Expression(expr=sum((model.C[g] * b.p[g]) + ((model.E[g] - model.phi) * b.w_3[g]) for g in model.OMEGA_G))
        
        elif constraint_mode == 'fix_tau_find_phi':
            b.PRIMAL_OBJECTIVE = Expression(expr=sum((model.C[g] * b.p[g]) + (model.E[g] * model.tau * b.p[g]) - (model.tau * b.w_2[g])  for g in model.OMEGA_G))
        
        elif constraint_mode == 'find_phi_find_tau':
            b.PRIMAL_OBJECTIVE = Expression(expr=sum((model.C[g] * b.p[g]) + (model.E[g] * b.w_3[g]) - b.w_4[g]  for g in model.OMEGA_G))
        
        # Dual objective
        b.DUAL_OBJECTIVE = Expression(expr=sum((b.mu_1[g] * model.P_MIN[g]) - (b.mu_2[g] * model.P_MAX[g]) for g in model.OMEGA_G)
                                     - sum(b.mu_4[j] * model.F[j] for j in model.OMEGA_J)
                                     + sum((b.mu_5[h] * model.P_H_MIN[h]) - (b.mu_6[h] * model.P_H_MAX[h]) for h in model.OMEGA_H)
                                     + sum((b.lamb[n] * (b.D[n] - b.R[n])) - model.THETA_DELTA * sum(b.mu_3[n, m] for m in network_graph[n]) for n in model.OMEGA_N)
                                     )
        
        # Strong duality constraint
        b.STRONG_DUALITY = Constraint(expr=b.PRIMAL_OBJECTIVE==b.DUAL_OBJECTIVE)                                 
    model.SCENARIO = Block(model.OMEGA_S, rule=SCENARIO_RULE)
    
    # Permit market
    model.PERMIT_1 = Constraint(expr=model.tau >= 0)
    
    if constraint_mode == 'fix_phi_find_tau':
        model.PERMIT_2 = Constraint(expr=sum(model.SCENARIO[s].RHO * ((model.E[g] - model.phi) * model.SCENARIO[s].p[g]) for s in model.OMEGA_S for g in model.OMEGA_G) <= 0)
        model.PERMIT_3 = Constraint(expr=sum(model.SCENARIO[s].RHO * ((model.E[g] - model.phi) * model.SCENARIO[s].w_3[g]) for s in model.OMEGA_S for g in model.OMEGA_G) == 0)
        
    elif constraint_mode == 'fix_tau_find_phi':
        model.PERMIT_2 = Constraint(expr=sum(model.SCENARIO[s].RHO * ((model.E[g] * model.SCENARIO[s].p[g]) - model.SCENARIO[s].w_2[g]) for s in model.OMEGA_S for g in model.OMEGA_G) <= 0)
        model.PERMIT_3 = Constraint(expr=sum(model.SCENARIO[s].RHO * ((model.E[g] * model.SCENARIO[s].p[g] * model.tau) - (model.tau * model.SCENARIO[s].w_2[g])) for s in model.OMEGA_S for g in model.OMEGA_G) == 0)
    
    elif constraint_mode == 'find_phi_find_tau':
        model.SLACK_1 = Var()
        model.PERMIT_2 = Constraint(expr=sum(model.SCENARIO[s].RHO * ((model.E[g] * model.SCENARIO[s].p[g]) - model.SCENARIO[s].w_2[g]) for s in model.OMEGA_S for g in model.OMEGA_G) <= 0)
        model.PERMIT_3 = Constraint(expr=sum(model.SCENARIO[s].RHO * ((model.E[g] * model.SCENARIO[s].w_3[g]) - model.SCENARIO[s].w_4[g]) for s in model.OMEGA_S for g in model.OMEGA_G) + model.SLACK_1  == 0)


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
    
    # Minimise average electricity price  
    elif objective_type == 'minimise_average_electricity_price':
        model.OBJECTIVE = Objective(expr=model.AVERAGE_ELECTRICITY_PRICE, sense=minimize)
        
        
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
        
        # Dummy variables to minimise slack variable absolute value
        model.SLACK_1_X_1 = Var(within=NonNegativeReals)
        model.SLACK_1_X_2 = Var(within=NonNegativeReals)
        
        # Constraints used to minimise difference between RRN price target and RRN price
        model.WEIGHTED_RRN_PRICE_TARGET_CONSTRAINT_1 = Constraint(expr=model.WEIGHTED_RRN_PRICE_X_1 >= model.WEIGHTED_RRN_PRICE - model.WEIGHTED_RRN_PRICE_TARGET)
        model.WEIGHTED_RRN_PRICE_TARGET_CONSTRAINT_2 = Constraint(expr=model.WEIGHTED_RRN_PRICE_X_2 >= model.WEIGHTED_RRN_PRICE_TARGET - model.WEIGHTED_RRN_PRICE)
        
        # Constraints used to get absolute value of slack variable
        model.SLACK_VARIABLE_CONSTRAINT_1 = Constraint(expr=model.SLACK_1_X_1 >= model.SLACK_1)
        model.SLACK_VARIABLE_CONSTRAINT_2 = Constraint(expr=model.SLACK_1_X_2 >= - model.SLACK_1)
        
        # Weighted RRN price targeting objective function with slack variable
        model.OBJECTIVE = Objective(expr=model.SLACK_1_X_1 + model.SLACK_1_X_2, sense=minimize)
#         model.WEIGHTED_RRN_PRICE_X_1 + model.WEIGHTED_RRN_PRICE_X_2
        
    elif objective_type == 'minimise_baseline':
        model.OBJECTIVE = Objective(expr=model.phi, sense=minimize)
        
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


# Solve model for different emissions intensity baselines.

# In[7]:


model = create_model(use_pu=True, objective_type='weighted_rrn_price_target', constraint_mode='find_phi_find_tau')
# model.dual = Suffix(direction=Suffix.IMPORT)

# Fix baseline
# model.phi.fix(0.98)

# Fix permit price
# model.tau.fix(0.5)

# Solve model
# opt.options['BarHomogeneous'] = 1
# opt.options['DualReductions'] = 0
# opt.options['InfUnbdInfo'] = 1

res = opt.solve(model, keepfiles=True, tee=True)

# Store results
model.solutions.store_to(res)


# Place results in DataFrame
try:
    df = pd.DataFrame(res['Solution'][0])
    fixed_baseline_results = {'baseline': model.phi.value, 'results': df}
except:
    fixed_baseline_results = {'baseline': model.phi.value, 'results': 'infeasible'}
    print('Baseline {0} is infeasible'.format(model.phi.value))


# In[8]:


# res
df


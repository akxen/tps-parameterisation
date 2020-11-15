"""Parameter selector variables"""

import os
import time
import math

import pyomo.environ as pyo

from utils.data import get_case_data


def define_sets(m, data):
    """Define sets"""

    # Nodes
    m.OMEGA_N = pyo.Set(initialize=data['S_NODES'])

    # Generators
    m.OMEGA_G = pyo.Set(initialize=data['S_GENERATORS'])

    # AC edges
    m.OMEGA_NM = pyo.Set(initialize=data['S_AC_EDGES'])

    # Sets of branches for which aggregate AC interconnector limits are defined
    m.OMEGA_J = pyo.Set(initialize=data['S_AC_AGGREGATED_EDGES'])

    # HVDC links
    m.OMEGA_H = pyo.Set(initialize=data['S_HVDC_EDGES'])

    # Operating scenarios
    m.OMEGA_S = pyo.Set(initialize=data['S_SCENARIOS'])

    # NEM regions
    m.OMEGA_R = pyo.Set(initialize=data['S_REGIONS'])

    # Branches which constitute AC interconnectors in the network
    m.OMEGA_L = pyo.Set(initialize=data['S_INTERCONNECTORS'])

    # Set of integers used to select discretised baseline
    m.OMEGA_U = pyo.Set(initialize=data['S_BINARY_EXPANSION_INTEGERS'])

    return m


def define_network_parameters(m, data):
    """Define network parameters"""

    # Max voltage angle difference between connected nodes
    m.P_NETWORK_VOLTAGE_ANGLE_MAX_DIFFERENCE = pyo.Param(initialize=float(math.pi / 2), mutable=True)

    # Branch susceptance matrix elements
    m.P_NETWORK_BRANCH_SUSCEPTANCE = pyo.Param(m.OMEGA_NM, initialize=data['P_NETWORK_BRANCH_SUSCEPTANCE'],
                                               mutable=True)

    # HVDC power flow limits
    m.P_NETWORK_HVDC_FORWARD_LIMIT = pyo.Param(m.OMEGA_H, initialize=data['P_NETWORK_HVDC_FORWARD_LIMIT'], mutable=True)
    m.P_NETWORK_HVDC_REVERSE_LIMIT = pyo.Param(m.OMEGA_H, initialize=data['P_NETWORK_HVDC_REVERSE_LIMIT'], mutable=True)

    # Reference node indicator
    m.P_NETWORK_REFERENCE_NODE_INDICATOR = pyo.Param(m.OMEGA_N, initialize=data['P_NETWORK_REFERENCE_NODE_INDICATOR'])

    # HVDC incidence matrix
    m.P_NETWORK_HVDC_INCIDENCE_MAT = pyo.Param(m.OMEGA_N, m.OMEGA_H, initialize=data['P_NETWORK_HVDC_INCIDENCE_MAT'])

    # AC interconnector incidence matrix
    m.P_NETWORK_INTERCONNECTOR_INCIDENCE_MAT = pyo.Param(m.OMEGA_N, m.OMEGA_L,
                                                         initialize=data['P_NETWORK_INTERCONNECTOR_INCIDENCE_MAT'])

    # Aggregate AC interconnector flow limits
    m.P_NETWORK_INTERCONNECTOR_LIMIT = pyo.Param(m.OMEGA_J, initialize=data['P_NETWORK_INTERCONNECTOR_LIMIT'],
                                                 mutable=True)

    # Fixed nodal power injections
    m.P_NETWORK_FIXED_INJECTION = pyo.Param(m.OMEGA_S, m.OMEGA_N, initialize=data['P_NETWORK_FIXED_INJECTION'],
                                            mutable=True)

    # Network demand
    m.P_NETWORK_DEMAND = pyo.Param(m.OMEGA_S, m.OMEGA_N, initialize=data['P_NETWORK_DEMAND'], mutable=True)

    # Proportion of total demand consumed in each region
    m.P_NETWORK_REGION_DEMAND_PROPORTION = pyo.Param(m.OMEGA_S, m.OMEGA_R,
                                                     initialize=data['P_NETWORK_REGION_DEMAND_PROPORTION'])

    # Network graph describing connections between nodes
    m.P_NETWORK_GRAPH = pyo.Param(m.OMEGA_N, initialize=data['P_NETWORK_GRAPH'], within=pyo.Any)

    # Branches associated with each interconnector
    m.P_NETWORK_INTERCONNECTOR_BRANCH_ID_MAP = pyo.Param(m.OMEGA_J,
                                                         initialize=data['P_NETWORK_INTERCONNECTOR_BRANCH_ID_MAP'],
                                                         within=pyo.Any)

    # Mapping between branch IDs and nodes
    m.P_NETWORK_INTERCONNECTOR_BRANCH_NODES_MAP = pyo.Param(
        m.OMEGA_L, initialize=data['P_NETWORK_INTERCONNECTOR_BRANCH_NODES_MAP'], within=pyo.Any)

    # Mapping between HVDC link ID and its respective 'from' and 'to' nodes
    m.P_NETWORK_HVDC_FROM_NODE = pyo.Param(m.OMEGA_H, initialize=data['P_NETWORK_HVDC_FROM_NODE'])
    m.P_NETWORK_HVDC_TO_NODE = pyo.Param(m.OMEGA_H, initialize=data['P_NETWORK_HVDC_TO_NODE'])

    # List of generators assigned to each node
    m.P_NETWORK_NODE_GENERATOR_MAP = pyo.Param(m.OMEGA_N, initialize=data['P_NETWORK_NODE_GENERATOR_MAP'],
                                               within=pyo.Any)

    # Base power
    m.P_NETWORK_BASE_POWER = pyo.Param(initialize=100)

    # Regional reference node IDs for each region
    m.P_REGION_RRN_NODE_MAP = pyo.Param(m.OMEGA_R, initialize=data['P_REGION_RRN_NODE_MAP'])

    return m


def define_scenario_parameters(m, data):
    """Define scenario parameters"""

    # Relative scenario duration
    m.P_SCENARIO_DURATION = pyo.Param(m.OMEGA_S, initialize=data['P_SCENARIO_DURATION'])

    return m


def define_generator_parameters(m, data):
    """Define generator parameters"""

    # Maximum generator output
    m.P_GENERATOR_MAX_OUTPUT = pyo.Param(m.OMEGA_G, initialize=data['P_GENERATOR_MAX_OUTPUT'], mutable=True)

    # Minimum generator output (set to 0)
    m.P_GENERATOR_MIN_OUTPUT = pyo.Param(m.OMEGA_G, initialize=data['P_GENERATOR_MIN_OUTPUT'])

    # Generator short-run marginal costs
    m.P_GENERATOR_SRMC = pyo.Param(m.OMEGA_G, initialize=data['P_GENERATOR_SRMC'], mutable=True)

    # Generator emissions intensities
    m.P_GENERATOR_EMISSIONS_INTENSITY = pyo.Param(m.OMEGA_G, initialize=data['P_GENERATOR_EMISSIONS_INTENSITY'])

    # Node to which generator is assigned
    m.P_GENERATOR_NODE = pyo.Param(m.OMEGA_G, initialize=data['P_GENERATOR_NODE'])

    return m


def define_policy_parameters(m, data):
    """Define policy parameters"""

    # Emissions intensity baseline (fixed)
    m.P_POLICY_FIXED_BASELINE = pyo.Param(initialize=data['P_POLICY_FIXED_BASELINE'], mutable=True)

    # Permit price target
    m.P_POLICY_PERMIT_PRICE_TARGET = pyo.Param(initialize=data['P_POLICY_PERMIT_PRICE_TARGET'], mutable=True)

    # Weighted RRN price target
    m.P_POLICY_WEIGHTED_RRN_PRICE_TARGET = pyo.Param(initialize=data['P_POLICY_WEIGHTED_RRN_PRICE_TARGET'],
                                                     mutable=True)

    return m


def define_big_m_parameters(m):
    """Define Big-M parameters"""

    def m_11_rule(m, g):
        return m.P_GENERATOR_MAX_OUTPUT[g] - m.P_GENERATOR_MIN_OUTPUT[g]

    m.P_M_11 = pyo.Param(m.OMEGA_G, rule=m_11_rule, mutable=True)
    m.P_M_12 = pyo.Param(m.OMEGA_G, initialize=1e3, mutable=True)

    def m_21_rule(m, g):
        return m.P_GENERATOR_MAX_OUTPUT[g] - m.P_GENERATOR_MIN_OUTPUT[g]

    m.P_M_21 = pyo.Param(m.OMEGA_G, rule=m_21_rule, mutable=True)
    m.P_M_22 = pyo.Param(m.OMEGA_G, initialize=1e3, mutable=True)

    m.P_M_31 = pyo.Param(m.OMEGA_NM, initialize=float(math.pi), mutable=True)
    m.P_M_32 = pyo.Param(m.OMEGA_NM, initialize=1e3, mutable=True)

    def m_41_rule(m, j):
        if 'REVERSE' in j:
            new_index = j.replace('REVERSE', 'FORWARD')
        elif 'FORWARD' in j:
            new_index = j.replace('FORWARD', 'REVERSE')
        else:
            raise (Exception('REVERSE / FORWARD not in index name'))
        return m.P_NETWORK_INTERCONNECTOR_LIMIT[j] + m.P_NETWORK_INTERCONNECTOR_LIMIT[new_index]

    m.P_M_41 = pyo.Param(m.OMEGA_J, rule=m_41_rule, mutable=True)
    m.P_M_42 = pyo.Param(m.OMEGA_J, initialize=1e3, mutable=True)

    def m_51_rule(m, h):
        return m.P_NETWORK_HVDC_FORWARD_LIMIT[h] - m.P_NETWORK_HVDC_REVERSE_LIMIT[h]

    m.P_M_51 = pyo.Param(m.OMEGA_H, rule=m_51_rule, mutable=True)
    m.P_M_52 = pyo.Param(m.OMEGA_H, initialize=1e3, mutable=True)

    def m_61_rule(m, h):
        return m.P_NETWORK_HVDC_FORWARD_LIMIT[h] - m.P_NETWORK_HVDC_REVERSE_LIMIT[h]

    m.P_M_61 = pyo.Param(m.OMEGA_H, rule=m_61_rule, mutable=True)
    m.P_M_62 = pyo.Param(m.OMEGA_H, initialize=1e3, mutable=True)

    m.P_M_71 = pyo.Param(initialize=1e4, mutable=True)
    m.P_M_72 = pyo.Param(rule=1e4, mutable=True)

    return m


def define_binary_expansion_parameters(m, data):
    """Parameters used when implementing binary expansion discretisation"""

    # Largest integer for baseline selection parameter index
    m.P_BIN_EXP_U = pyo.Param(initialize=data['P_BINARY_EXPANSION_LARGEST_INTEGER'])

    # Minimum emissions intensity baseline
    m.P_BIN_EXP_MIN_BASELINE = pyo.Param(initialize=float(0.8))

    # Maximum emissions intensity baseline
    m.P_BIN_EXP_MAX_BASELINE = pyo.Param(initialize=float(1.3))

    # Emissions intensity baseline increment
    m.P_BIN_EXP_BASELINE_DELTA = pyo.Param(
        initialize=float((m.P_BIN_EXP_MAX_BASELINE - m.P_BIN_EXP_MIN_BASELINE) / (2 ** m.P_BIN_EXP_U)))

    def two_pow_u_rule(m, u):
        """Parameter equal 2^u used when selecting discretised emissions intensity baseline"""
        return float(2 ** u)

    m.P_BIN_EXP_TWO_POW_U = pyo.Param(m.OMEGA_U, rule=two_pow_u_rule)

    # Big-M parameter used used to ensure Z_1=0 when PSI=0, and Z_1=V_DUAL_PERMIT_MARKET when PSI=1
    m.P_BIN_EXP_L_1 = pyo.Param(initialize=1e3)

    def l_2_rule(m, g):
        """ Big-M parameter used to make Z_2=0 when PSI=0, and Z_2=p when PSI=1"""
        return m.P_GENERATOR_MAX_OUTPUT[g] - m.P_GENERATOR_MIN_OUTPUT[g]

    m.P_BIN_EXP_L_2 = pyo.Param(m.OMEGA_G, rule=l_2_rule)

    return m


def define_parameters(m, data):
    """Define parameters"""

    # Policy, network, and generator parameters
    m = define_policy_parameters(m, data)
    m = define_network_parameters(m, data)
    m = define_generator_parameters(m, data)
    m = define_scenario_parameters(m, data)
    m = define_big_m_parameters(m)

    # Binary expansion parameters
    m = define_binary_expansion_parameters(m, data)

    return m


def define_binary_expansion_variables(m):
    """Binary expansion variables"""

    # Binary variables used to determine discretised emissions intensity baseline choice
    m.V_BIN_EXP_PSI = pyo.Var(m.OMEGA_U, within=pyo.Binary)

    # Composite variable - PSI x tau - used to linearise bi-linear term
    m.V_BIN_EXP_Z_1 = pyo.Var(m.OMEGA_U)

    # Composite variable - PSI x p - used to linearise bi-linear term
    m.V_BIN_EXP_Z_2 = pyo.Var(m.OMEGA_S, m.OMEGA_U * m.OMEGA_G)

    return m


def define_primal_variables(m):
    """Define primal variables"""

    # Generator output
    m.V_PRIMAL_GENERATOR_POWER = pyo.Var(m.OMEGA_S, m.OMEGA_G)

    # HVDC link flow
    m.V_PRIMAL_HVDC_FLOW = pyo.Var(m.OMEGA_S, m.OMEGA_H)

    # Node voltage angle
    m.V_PRIMAL_VOLTAGE_ANGLE = pyo.Var(m.OMEGA_S, m.OMEGA_N)

    return m


def define_dual_variables(m):
    """Define dual variables"""

    # Permit market constraint dual variable
    m.V_DUAL_PERMIT_MARKET = pyo.Var()

    # Min power output constraint dual variable
    m.V_DUAL_MU_1 = pyo.Var(m.OMEGA_S, m.OMEGA_G)

    # Max power output constraint dual variable
    m.V_DUAL_MU_2 = pyo.Var(m.OMEGA_S, m.OMEGA_G)

    # Max voltage angle difference constraint dual variable
    m.V_DUAL_MU_3 = pyo.Var(m.OMEGA_S, m.OMEGA_NM)

    # AC link power flow constraint dual variable
    m.V_DUAL_MU_4 = pyo.Var(m.OMEGA_S, m.OMEGA_J)

    # Min HVDC flow constraint dual variable
    m.V_DUAL_MU_5 = pyo.Var(m.OMEGA_S, m.OMEGA_H)

    # Max HVDC flow constraint dual variable
    m.V_DUAL_MU_6 = pyo.Var(m.OMEGA_S, m.OMEGA_H)

    # Reference node voltage angle constraint dual variable
    m.V_DUAL_NU_1 = pyo.Var(m.OMEGA_S, m.OMEGA_N)

    # Node power balance constraint dual variable
    m.V_DUAL_LAMBDA = pyo.Var(m.OMEGA_S, m.OMEGA_N)

    return m


def define_big_m_binary_variables(m):
    """Binary variables used in Big-M linearisation"""

    # Permit market constraint binary variable
    # m.V_GAMMA_7 = pyo.Var(within=pyo.Binary)
    m.V_BINARY_PERMIT_MARKET = pyo.Var(within=pyo.Binary)

    # Min power output binary variable
    m.V_BINARY_GAMMA_1 = pyo.Var(m.OMEGA_S, m.OMEGA_G, within=pyo.Binary)

    # Max power output binary variable
    m.V_BINARY_GAMMA_2 = pyo.Var(m.OMEGA_S, m.OMEGA_G, within=pyo.Binary)

    # Max voltage angle difference binary variable
    m.V_BINARY_GAMMA_3 = pyo.Var(m.OMEGA_S, m.OMEGA_NM, within=pyo.Binary)

    # AC link power flow dual variable
    m.V_BINARY_GAMMA_4 = pyo.Var(m.OMEGA_S, m.OMEGA_J, within=pyo.Binary)

    # Min HVDC flow binary variable
    m.V_BINARY_GAMMA_5 = pyo.Var(m.OMEGA_S, m.OMEGA_H, within=pyo.Binary)

    # Max HVDC flow binary variable
    m.V_BINARY_GAMMA_6 = pyo.Var(m.OMEGA_S, m.OMEGA_H, within=pyo.Binary)

    return m


def define_dummy_variables(m):
    """Dummy variables used as a placeholder in objective function"""

    m.V_DUMMY = pyo.Var(bounds=(0, 1))

    # Dummy variables used to target given permit price
    m.V_DUMMY_PERMIT_PRICE_TARGET_X_1 = pyo.Var(within=pyo.NonNegativeReals)
    m.V_DUMMY_PERMIT_PRICE_TARGET_X_2 = pyo.Var(within=pyo.NonNegativeReals)

    # Dummy variables used to minimise difference between average price and price target
    m.V_DUMMY_WEIGHTED_RRN_PRICE_X_1 = pyo.Var(within=pyo.NonNegativeReals)
    m.V_DUMMY_WEIGHTED_RRN_PRICE_X_2 = pyo.Var(within=pyo.NonNegativeReals)

    return m


def define_variables(m):
    """Define model variables"""

    # Define primal and dual variables
    m = define_primal_variables(m)
    m = define_dual_variables(m)

    # Binary variables used in linearised complementarity constraints
    m = define_big_m_binary_variables(m)

    # Variables used in binary expansion formulation
    m = define_binary_expansion_variables(m)

    # Dummy variables that may be used as a placeholder within an objective function
    m = define_dummy_variables(m)

    return m


def define_binary_expansion_expressions(m):
    """Binary expansion expressions"""

    # Discretised emissions intensity baseline value
    m.E_BIN_EXP_DISCRETISED_BASELINE = pyo.Expression(
        expr=m.P_BIN_EXP_MIN_BASELINE + (
                m.P_BIN_EXP_BASELINE_DELTA * sum(m.P_BIN_EXP_TWO_POW_U[u] * m.V_BIN_EXP_PSI[u] for u in m.OMEGA_U))
    )

    return m


def define_price_target_expressions(m):
    """Expressions use to compute price targets"""

    # Total revenue from electricity sales
    m.E_TOTAL_REVENUE = pyo.Expression(
        expr=sum(m.P_SCENARIO_DURATION[s] * m.V_DUAL_LAMBDA[s, n] * m.P_NETWORK_DEMAND[s, n]
                 for s in m.OMEGA_S for n in m.OMEGA_N))

    # Total demand
    m.E_TOTAL_DEMAND = pyo.Expression(expr=sum(m.P_SCENARIO_DURATION[s] * m.P_NETWORK_DEMAND[s, n]
                                               for s in m.OMEGA_S for n in m.OMEGA_N))

    # Average price
    m.E_AVERAGE_ELECTRICITY_PRICE = pyo.Expression(expr=m.E_TOTAL_REVENUE / m.E_TOTAL_DEMAND)

    # Weighted RRN price
    m.E_WEIGHTED_RRN_PRICE = pyo.Expression(
        expr=sum(m.P_SCENARIO_DURATION[s] * m.P_NETWORK_REGION_DEMAND_PROPORTION[s, r]
                 * m.V_DUAL_LAMBDA[s, m.P_REGION_RRN_NODE_MAP[r]] for s in m.OMEGA_S for r in m.OMEGA_R))

    return m


def define_expressions(m):
    """Define expressions"""

    # Binary expansion expressions
    m = define_binary_expansion_expressions(m)

    # Expressions used in price targeting objective functions
    m = define_price_target_expressions(m)

    return m


def define_binary_expansion_constraints(m):
    """Constraints used in binary expansion formulation"""

    # Constraints ensures Z_1=0 when PSI=0 and Z_1=permit_price when PSI=1
    def z_1_constraint_1_rule(m, u):
        return 0 <= m.V_DUAL_PERMIT_MARKET - m.V_BIN_EXP_Z_1[u]

    m.C_Z_1_CONSTRAINT_1 = pyo.Constraint(m.OMEGA_U, rule=z_1_constraint_1_rule)

    def z_1_constraint_2_rule(m, u):
        return m.V_DUAL_PERMIT_MARKET - m.V_BIN_EXP_Z_1[u] <= m.P_BIN_EXP_L_1 * (1 - m.V_BIN_EXP_PSI[u])

    m.C_Z_1_CONSTRAINT_2 = pyo.Constraint(m.OMEGA_U, rule=z_1_constraint_2_rule)

    def z_1_constraint_3_rule(m, u):
        return 0 <= m.V_BIN_EXP_Z_1[u]

    m.C_Z_1_CONSTRAINT_3 = pyo.Constraint(m.OMEGA_U, rule=z_1_constraint_3_rule)

    def z_1_constraint_4_rule(m, u):
        return m.V_BIN_EXP_Z_1[u] <= m.P_BIN_EXP_L_1 * m.V_BIN_EXP_PSI[u]

    m.C_Z_1_CONSTRAINT_4 = pyo.Constraint(m.OMEGA_U, rule=z_1_constraint_4_rule)

    # Constraints ensuring z_2=0 when PSI=0, and z_2=p when PSI=1
    def z_2_constraint_1_rule(m, s, u, g):
        return 0 <= m.V_PRIMAL_GENERATOR_POWER[s, g] - m.V_BIN_EXP_Z_2[s, u, g]

    m.C_Z_2_CONSTRAINT_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_U * m.OMEGA_G, rule=z_2_constraint_1_rule)

    def z_2_constraint_2_rule(m, s, u, g):
        return m.V_PRIMAL_GENERATOR_POWER[s, g] - m.V_BIN_EXP_Z_2[s, u, g] <= m.P_BIN_EXP_L_2[g] * (
                1 - m.V_BIN_EXP_PSI[u])

    m.C_Z_2_CONSTRAINT_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_U * m.OMEGA_G, rule=z_2_constraint_2_rule)

    def z_2_constraint_3_rule(m, s, u, g):
        return 0 <= m.V_BIN_EXP_Z_2[s, u, g]

    m.C_Z_2_CONSTRAINT_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_U * m.OMEGA_G, rule=z_2_constraint_3_rule)

    def z_2_constraint_4_rule(m, s, u, g):
        return m.V_BIN_EXP_Z_2[s, u, g] <= m.P_BIN_EXP_L_2[g] * m.V_BIN_EXP_PSI[u]

    m.C_Z_2_CONSTRAINT_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_U * m.OMEGA_G, rule=z_2_constraint_4_rule)

    return m


def define_first_order_condition_constraints(m):
    """Define constraints constituting the model's first order conditions"""

    def foc_1_rule(m, s, g):
        """To be activated if baseline is fixed"""
        return (m.P_GENERATOR_SRMC[g]
                + ((m.P_GENERATOR_EMISSIONS_INTENSITY[g] - m.P_POLICY_FIXED_BASELINE) * m.V_DUAL_PERMIT_MARKET)
                - m.V_DUAL_MU_1[s, g]
                + m.V_DUAL_MU_2[s, g]
                - m.V_DUAL_LAMBDA[s, m.P_GENERATOR_NODE[g]] == 0)

    m.C_FOC_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=foc_1_rule)

    def foc_1_linearised_rule(m, s, g):
        """To be activated if baseline is discretised"""
        return (m.P_GENERATOR_SRMC[g]
                + ((m.P_GENERATOR_EMISSIONS_INTENSITY[g] - m.P_BIN_EXP_MIN_BASELINE) * m.V_DUAL_PERMIT_MARKET)
                - (m.P_BIN_EXP_BASELINE_DELTA * sum(m.P_BIN_EXP_TWO_POW_U[u] * m.V_BIN_EXP_Z_1[u] for u in m.OMEGA_U))
                - m.V_DUAL_MU_1[s, g]
                + m.V_DUAL_MU_2[s, g]
                - m.V_DUAL_LAMBDA[s, m.P_GENERATOR_NODE[g]]
                == 0)

    m.C_FOC_1_LINEARISED = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=foc_1_linearised_rule)

    def foc_2_rule(m, s, n):
        """Constraint relating to voltage angles"""

        term_1 = sum(m.V_DUAL_MU_3[s, n, j]
                     - m.V_DUAL_MU_3[s, j, n]
                     + (m.V_DUAL_LAMBDA[s, n] * m.P_NETWORK_BRANCH_SUSCEPTANCE[n, j])
                     - (m.V_DUAL_LAMBDA[s, j] * m.P_NETWORK_BRANCH_SUSCEPTANCE[j, n]) for j in m.P_NETWORK_GRAPH[n])

        term_2 = m.V_DUAL_NU_1[s, n] * m.P_NETWORK_REFERENCE_NODE_INDICATOR[n]

        term_3 = sum(
            m.P_NETWORK_BRANCH_SUSCEPTANCE[m.P_NETWORK_INTERCONNECTOR_BRANCH_NODES_MAP[l]]
            * m.V_DUAL_MU_4[s, j]
            * m.P_NETWORK_INTERCONNECTOR_INCIDENCE_MAT[n, l]
            for j in m.OMEGA_J for l in m.P_NETWORK_INTERCONNECTOR_BRANCH_ID_MAP[j])

        return term_1 + term_2 + term_3 == 0

    m.C_FOC_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_N, rule=foc_2_rule)

    def foc_3_rule(m, s, h):
        """Constraint relating to HVDC flows"""

        # From and to nodes for given link, h
        from_node = m.P_NETWORK_HVDC_FROM_NODE[h]
        to_node = m.P_NETWORK_HVDC_TO_NODE[h]

        return ((m.P_NETWORK_HVDC_INCIDENCE_MAT[from_node, h] * m.V_DUAL_LAMBDA[s, from_node])
                + (m.P_NETWORK_HVDC_INCIDENCE_MAT[to_node, h] * m.V_DUAL_LAMBDA[s, to_node])
                - m.V_DUAL_MU_5[s, h]
                + m.V_DUAL_MU_6[s, h] == 0)

    m.C_FOC_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=foc_3_rule)

    return m


def define_network_constraint(m):
    """Define network constraints"""

    def reference_node_voltage_angle_rule(m, s, n):
        if m.P_NETWORK_REFERENCE_NODE_INDICATOR[n] == 1:
            return m.V_PRIMAL_VOLTAGE_ANGLE[s, n] == 0
        else:
            return pyo.Constraint.Skip

    m.C_REFERENCE_NODE_VOLTAGE_ANGLE = pyo.Constraint(m.OMEGA_S, m.OMEGA_N, rule=reference_node_voltage_angle_rule)

    def power_balance_rule(m, s, n):
        """Power balance at each node"""

        # Total injection from generators connected to node
        generator_injection = sum(m.V_PRIMAL_GENERATOR_POWER[s, g] for g in m.P_NETWORK_NODE_GENERATOR_MAP[n])

        # Injection from AC branches
        ac_injection = sum(
            m.P_NETWORK_BRANCH_SUSCEPTANCE[n, j] * (m.V_PRIMAL_VOLTAGE_ANGLE[s, n] - m.V_PRIMAL_VOLTAGE_ANGLE[s, j])
            for j in m.P_NETWORK_GRAPH[n])

        # Injection from HVDC branches
        hvdc_injection = sum(m.P_NETWORK_HVDC_INCIDENCE_MAT[n, h] * m.V_PRIMAL_HVDC_FLOW[s, h] for h in m.OMEGA_H)

        return (m.P_NETWORK_DEMAND[s, n]
                - m.P_NETWORK_FIXED_INJECTION[s, n]
                - generator_injection
                + ac_injection
                + hvdc_injection
                == 0)

    m.C_POWER_BALANCE = pyo.Constraint(m.OMEGA_S, m.OMEGA_N, rule=power_balance_rule)

    return m


def define_linearised_complementarity_constraints(m):
    """Linearised complementarity constraints"""

    def lin_comp_1_1_rule(m, s, g):
        """Min power output constraint 1"""

        return m.P_GENERATOR_MIN_OUTPUT[g] - m.V_PRIMAL_GENERATOR_POWER[s, g] <= 0

    m.C_LIN_COMP_1_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_1_1_rule)

    def lin_comp_1_2_rule(m, s, g):
        """Min power output constraint 2"""

        return m.V_DUAL_MU_1[s, g] >= 0

    m.C_LIN_COMP_1_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_1_2_rule)

    def lin_comp_1_3_rule(m, s, g):
        """Min power output constraint 3"""

        return m.V_PRIMAL_GENERATOR_POWER[s, g] - m.P_GENERATOR_MIN_OUTPUT[g] <= m.V_BINARY_GAMMA_1[s, g] * m.P_M_11[g]

    m.C_LIN_COMP_1_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_1_3_rule)

    def lin_comp_1_4_rule(m, s, g):
        """Min power output constraint 4"""

        return m.V_DUAL_MU_1[s, g] <= (1 - m.V_BINARY_GAMMA_1[s, g]) * m.P_M_12[g]

    m.C_LIN_COMP_1_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_1_4_rule)

    def lin_comp_2_1_rule(m, s, g):
        """Max power output constraint 1"""

        return m.V_PRIMAL_GENERATOR_POWER[s, g] - m.P_GENERATOR_MAX_OUTPUT[g] <= 0

    m.C_LIN_COMP_2_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_2_1_rule)

    def lin_comp_2_2_rule(m, s, g):
        """Max power output constraint 2"""

        return m.V_DUAL_MU_2[s, g] >= 0

    m.C_LIN_COMP_2_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_2_2_rule)

    def lin_comp_2_3_rule(m, s, g):
        """Max power output constraint 3"""

        return (m.P_GENERATOR_MAX_OUTPUT[g] - m.V_PRIMAL_GENERATOR_POWER[s, g]
                <= m.V_BINARY_GAMMA_2[s, g] * m.P_M_21[g])

    m.C_LIN_COMP_2_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_2_3_rule)

    def lin_comp_2_4_rule(m, s, g):
        """Max power output constraint 4"""

        return m.V_DUAL_MU_2[s, g] <= (1 - m.V_BINARY_GAMMA_2[s, g]) * m.P_M_22[g]

    m.C_LIN_COMP_2_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_G, rule=lin_comp_2_4_rule)

    def lin_comp_3_1_rule(m, s, n, j):
        """Max voltage angle difference between connected nodes constraint 1"""

        return (m.V_PRIMAL_VOLTAGE_ANGLE[s, n] - m.V_PRIMAL_VOLTAGE_ANGLE[s, j]
                - m.P_NETWORK_VOLTAGE_ANGLE_MAX_DIFFERENCE
                <= 0)

    m.C_LIN_COMP_3_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_NM, rule=lin_comp_3_1_rule)

    def lin_comp_3_2_rule(m, s, n, j):
        """Max voltage angle difference between connected nodes constraint 2"""

        return m.V_DUAL_MU_3[s, n, j] >= 0

    m.C_LIN_COMP_3_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_NM, rule=lin_comp_3_2_rule)

    def lin_comp_3_3_rule(m, s, n, j):
        """Max voltage angle difference between connected nodes constraint 3"""

        return (m.P_NETWORK_VOLTAGE_ANGLE_MAX_DIFFERENCE + m.V_PRIMAL_VOLTAGE_ANGLE[s, j]
                - m.V_PRIMAL_VOLTAGE_ANGLE[s, n] <= m.V_BINARY_GAMMA_3[s, n, j] * m.P_M_31[n, j])

    m.C_LIN_COMP_3_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_NM, rule=lin_comp_3_3_rule)

    def lin_comp_3_4_rule(m, s, n, j):
        """Max voltage angle difference between connected nodes constraint 4"""

        return m.V_DUAL_MU_3[s, n, j] <= (1 - m.V_BINARY_GAMMA_3[s, n, j]) * m.P_M_32[n, j]

    m.C_LIN_COMP_3_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_NM, rule=lin_comp_3_4_rule)

    def lin_comp_4_1_rule(m, s, j):
        """Interconnector flow limits constraint 1"""

        # Links constituting interconnector, j
        branches = m.P_NETWORK_INTERCONNECTOR_BRANCH_ID_MAP[j]
        branch_links = [m.P_NETWORK_INTERCONNECTOR_BRANCH_NODES_MAP[i] for i in branches]

        return (sum(m.P_NETWORK_BRANCH_SUSCEPTANCE[n, p]
                    * (m.V_PRIMAL_VOLTAGE_ANGLE[s, n] - m.V_PRIMAL_VOLTAGE_ANGLE[s, p])
                    for n, p in branch_links) - m.P_NETWORK_INTERCONNECTOR_LIMIT[j]
                <= 0)

    m.C_LIN_COMP_4_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_J, rule=lin_comp_4_1_rule)

    def lin_comp_4_2_rule(m, s, j):
        """Interconnector flow limits constraint 2"""

        return m.V_DUAL_MU_4[s, j] >= 0

    m.C_LIN_COMP_4_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_J, rule=lin_comp_4_2_rule)

    def lin_comp_4_3_rule(m, s, j):
        """Interconnector flow limits constraint 3"""

        # Links constituting interconnector, j
        branches = m.P_NETWORK_INTERCONNECTOR_BRANCH_ID_MAP[j]
        branch_links = [m.P_NETWORK_INTERCONNECTOR_BRANCH_NODES_MAP[i] for i in branches]

        return (m.P_NETWORK_INTERCONNECTOR_LIMIT[j]
                - sum(m.P_NETWORK_BRANCH_SUSCEPTANCE[n, p]
                      * (m.V_PRIMAL_VOLTAGE_ANGLE[s, n] - m.V_PRIMAL_VOLTAGE_ANGLE[s, p]) for n, p in branch_links)
                <= m.V_BINARY_GAMMA_4[s, j] * m.P_M_41[j])

    m.C_LIN_COMP_4_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_J, rule=lin_comp_4_3_rule)

    def lin_comp_4_4_rule(m, s, j):
        """Interconnector flow limits constraint 4"""

        return m.V_DUAL_MU_4[s, j] <= (1 - m.V_BINARY_GAMMA_4[s, j]) * m.P_M_42[j]

    m.C_LIN_COMP_4_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_J, rule=lin_comp_4_4_rule)

    def lin_comp_5_1_rule(m, s, h):
        """HVDC reverse flow limits constraint 1"""

        return m.P_NETWORK_HVDC_REVERSE_LIMIT[h] - m.V_PRIMAL_HVDC_FLOW[s, h] <= 0

    m.C_LIN_COMP_5_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_5_1_rule)

    def lin_comp_5_2_rule(m, s, h):
        """HVDC reverse flow limits constraint 2"""

        return m.V_DUAL_MU_5[s, h] >= 0

    m.C_LIN_COMP_5_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_5_2_rule)

    def lin_comp_5_3_rule(m, s, h):
        """HVDC reverse flow limits constraint 3"""

        return m.V_PRIMAL_HVDC_FLOW[s, h] - m.P_NETWORK_HVDC_REVERSE_LIMIT[h] <= m.V_BINARY_GAMMA_5[s, h] * m.P_M_51[h]

    m.C_LIN_COMP_5_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_5_3_rule)

    def lin_comp_5_4_rule(m, s, h):
        """HVDC reverse flow limits constraint 4"""

        return m.V_DUAL_MU_5[s, h] <= (1 - m.V_BINARY_GAMMA_5[s, h]) * m.P_M_52[h]

    m.C_LIN_COMP_5_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_5_4_rule)

    def lin_comp_6_1_rule(m, s, h):
        """HVDC forward flow limits constraint 1"""

        return m.V_PRIMAL_HVDC_FLOW[s, h] - m.P_NETWORK_HVDC_FORWARD_LIMIT[h] <= 0

    m.C_LIN_COMP_6_1 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_6_1_rule)

    def lin_comp_6_2_rule(m, s, h):
        """HVDC forward flow limits constraint 2"""

        return m.V_DUAL_MU_6[s, h] >= 0

    m.C_LIN_COMP_6_2 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_6_2_rule)

    def lin_comp_6_3_rule(m, s, h):
        """HVDC forward flow limits constraint 3"""

        return m.P_NETWORK_HVDC_FORWARD_LIMIT[h] - m.V_PRIMAL_HVDC_FLOW[s, h] <= m.V_BINARY_GAMMA_6[s, h] * m.P_M_61[h]

    m.C_LIN_COMP_6_3 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_6_3_rule)

    def lin_comp_6_4_rule(m, s, h):
        """HVDC forward flow limits constraint 4"""

        return m.V_DUAL_MU_6[s, h] <= (1 - m.V_BINARY_GAMMA_6[s, h]) * m.P_M_62[h]

    m.C_LIN_COMP_6_4 = pyo.Constraint(m.OMEGA_S, m.OMEGA_H, rule=lin_comp_6_4_rule)

    return m


def define_permit_market_constraints(m):
    """Define permit market constraints"""

    def permit_market_1_rule(m):
        """Permit market complementarity constraint 1 - use when baseline is fixed"""

        return (sum(m.P_SCENARIO_DURATION[s] * ((m.P_GENERATOR_EMISSIONS_INTENSITY[g] - m.P_POLICY_FIXED_BASELINE)
                                                * m.V_PRIMAL_GENERATOR_POWER[s, g])
                    for g in m.OMEGA_G for s in m.OMEGA_S)
                <= 0)

    m.C_PERMIT_MARKET_1 = pyo.Constraint(rule=permit_market_1_rule)

    def permit_market_1_linearised_rule(m):
        """Permit market complementarity constraint 1 - use when baseline is variable"""

        return (sum(m.P_SCENARIO_DURATION[s]
                    * (((m.P_GENERATOR_EMISSIONS_INTENSITY[g] - m.P_BIN_EXP_MIN_BASELINE)
                        * m.V_PRIMAL_GENERATOR_POWER[s, g])
                       - (m.P_BIN_EXP_BASELINE_DELTA * sum(m.P_BIN_EXP_TWO_POW_U[u] * m.V_BIN_EXP_Z_2[s, u, g]
                                                           for u in m.OMEGA_U))) for g in m.OMEGA_G for s in m.OMEGA_S)
                <= 0)

    m.C_PERMIT_MARKET_1_LINEARISED = pyo.Constraint(rule=permit_market_1_linearised_rule)

    def permit_market_2_rule(m):
        """Permit market complementarity constraint 2"""

        return m.V_DUAL_PERMIT_MARKET >= 0

    m.C_PERMIT_MARKET_2 = pyo.Constraint(rule=permit_market_2_rule)

    def permit_market_3_rule(m):
        """Permit market complementarity constraint 3"""

        return sum(m.P_SCENARIO_DURATION[s]
                   * ((m.P_POLICY_FIXED_BASELINE - m.P_GENERATOR_EMISSIONS_INTENSITY[g]) * m.V_PRIMAL_GENERATOR_POWER[
            s, g])
                   for g in m.OMEGA_G for s in m.OMEGA_S) <= m.V_BINARY_PERMIT_MARKET * m.P_M_71

    m.C_PERMIT_MARKET_3 = pyo.Constraint(rule=permit_market_3_rule)

    def permit_market_3_linearised_rule(m):
        """Permit market complementarity constraint 3 - linearised"""

        return (sum(m.P_SCENARIO_DURATION[s] * (((m.P_BIN_EXP_MIN_BASELINE - m.P_GENERATOR_EMISSIONS_INTENSITY[g])
                                                 * m.V_PRIMAL_GENERATOR_POWER[s, g])
                                                + (m.P_BIN_EXP_BASELINE_DELTA
                                                   * sum(m.P_BIN_EXP_TWO_POW_U[u] * m.V_BIN_EXP_Z_2[s, u, g]
                                                         for u in m.OMEGA_U))) for g in m.OMEGA_G for s in m.OMEGA_S)
                <= m.V_BINARY_PERMIT_MARKET * m.P_M_71)

    m.C_PERMIT_MARKET_3_LINEARISED = pyo.Constraint(rule=permit_market_3_linearised_rule)

    def permit_market_4_rule(m):
        """Permit market complementarity constraint 4"""

        return m.V_DUAL_PERMIT_MARKET <= (1 - m.V_BINARY_PERMIT_MARKET) * m.P_M_72

    m.C_PERMIT_MARKET_4 = pyo.Constraint(rule=permit_market_4_rule)

    return m


def define_permit_price_targeting_constraints(m):
    """Constraints used to get the absolute difference between the permit price and some target"""

    # Constraints to minimise difference between permit price and target
    m.C_PERMIT_PRICE_TARGET_CONSTRAINT_1 = pyo.Constraint(
        expr=m.V_DUMMY_PERMIT_PRICE_TARGET_X_1 >= m.P_POLICY_PERMIT_PRICE_TARGET - m.V_DUAL_PERMIT_MARKET)

    m.C_PERMIT_PRICE_TARGET_CONSTRAINT_2 = pyo.Constraint(
        expr=m.V_DUMMY_PERMIT_PRICE_TARGET_X_2 >= m.V_DUAL_PERMIT_MARKET - m.P_POLICY_PERMIT_PRICE_TARGET)

    return m


def define_weighted_rrn_price_targeting_constraints(m):
    """Constraints used to get absolute difference between weighted RRN price and some target"""

    # Constraints used to minimise difference between RRN price target and RRN price
    m.C_WEIGHTED_RRN_PRICE_TARGET_CONSTRAINT_1 = pyo.Constraint(
        expr=m.V_DUMMY_WEIGHTED_RRN_PRICE_X_1 >= m.E_WEIGHTED_RRN_PRICE - m.P_POLICY_WEIGHTED_RRN_PRICE_TARGET)

    m.C_WEIGHTED_RRN_PRICE_TARGET_CONSTRAINT_2 = pyo.Constraint(
        expr=m.V_DUMMY_WEIGHTED_RRN_PRICE_X_2 >= m.P_POLICY_WEIGHTED_RRN_PRICE_TARGET - m.E_WEIGHTED_RRN_PRICE)

    return m


def define_constraints(m):
    """Define constraints"""

    m = define_binary_expansion_constraints(m)
    m = define_first_order_condition_constraints(m)
    m = define_network_constraint(m)
    m = define_linearised_complementarity_constraints(m)
    m = define_permit_market_constraints(m)

    # Constraints used to obtain absolute values between variables and some target
    m = define_permit_price_targeting_constraints(m)
    m = define_weighted_rrn_price_targeting_constraints(m)

    return m


def define_objective_functions(m):
    """Define objective functions"""

    # Use if trying to find a feasible solution for a fixed baseline
    m.O_FEASIBILITY = pyo.Objective(expr=m.V_DUMMY, sense=pyo.minimize)

    # Objective function
    m.O_PERMIT_PRICE_TARGET = pyo.Objective(expr=m.V_DUMMY_PERMIT_PRICE_TARGET_X_1 + m.V_DUMMY_PERMIT_PRICE_TARGET_X_2)

    # Weighted RRN price targeting objective function
    m.O_WEIGHTED_RRN_PRICE_TARGET = pyo.Objective(
        expr=m.V_DUMMY_WEIGHTED_RRN_PRICE_X_1 + m.V_DUMMY_WEIGHTED_RRN_PRICE_X_2)

    return m


def apply_pu_scaling(m):
    """Attempt to improve numerical stability by scaling selected parameters"""

    # Scale branch susceptance
    for i in m.P_NETWORK_BRANCH_SUSCEPTANCE.keys():
        m.P_NETWORK_BRANCH_SUSCEPTANCE[i] = m.P_NETWORK_BRANCH_SUSCEPTANCE[i] / m.P_NETWORK_BASE_POWER

    # HVDC limits
    for i in m.P_NETWORK_HVDC_FORWARD_LIMIT.keys():
        m.P_NETWORK_HVDC_FORWARD_LIMIT[i] = m.P_NETWORK_HVDC_FORWARD_LIMIT[i] / m.P_NETWORK_BASE_POWER

    for i in m.P_NETWORK_HVDC_REVERSE_LIMIT.keys():
        m.P_NETWORK_HVDC_REVERSE_LIMIT[i] = m.P_NETWORK_HVDC_REVERSE_LIMIT[i] / m.P_NETWORK_BASE_POWER

    # Generator max output
    for i in m.P_GENERATOR_MAX_OUTPUT.keys():
        m.P_GENERATOR_MAX_OUTPUT[i] = m.P_GENERATOR_MAX_OUTPUT[i] / m.P_NETWORK_BASE_POWER

    # Generator short run marginal costs
    for i in m.P_GENERATOR_SRMC.keys():
        m.P_GENERATOR_SRMC[i] = m.P_GENERATOR_SRMC[i] / m.P_NETWORK_BASE_POWER

    # Interconnector flow limits
    for i in m.P_NETWORK_INTERCONNECTOR_LIMIT.keys():
        m.P_NETWORK_INTERCONNECTOR_LIMIT[i] = m.P_NETWORK_INTERCONNECTOR_LIMIT[i] / m.P_NETWORK_BASE_POWER

    # Fixed injections
    for i in m.P_NETWORK_FIXED_INJECTION.keys():
        m.P_NETWORK_FIXED_INJECTION[i] = m.P_NETWORK_FIXED_INJECTION[i] / m.P_NETWORK_BASE_POWER

    # Demand
    for i in m.P_NETWORK_DEMAND.keys():
        m.P_NETWORK_DEMAND[i] = m.P_NETWORK_DEMAND[i] / m.P_NETWORK_BASE_POWER

    # Complementarity constraints
    big_m_parameters = [f'P_M_{i}{j}' for i in range(1, 8) for j in range(1, 3)]
    ignore_scaling = ['P_M_31']
    for i in big_m_parameters:
        if i in ignore_scaling:
            continue
        else:
            for j in m.__getattribute__(i).keys():
                m.__getattribute__(i)[j] = m.__getattribute__(i)[j] / m.P_NETWORK_BASE_POWER

    # Scale price targets
    m.P_POLICY_PERMIT_PRICE_TARGET = m.P_POLICY_PERMIT_PRICE_TARGET.value / m.P_NETWORK_BASE_POWER.value
    m.P_POLICY_WEIGHTED_RRN_PRICE_TARGET = m.P_POLICY_WEIGHTED_RRN_PRICE_TARGET.value / m.P_NETWORK_BASE_POWER.value

    return m


def construct_model(data, use_pu):
    """Construct model"""

    # Initialise model
    m = pyo.ConcreteModel()

    # Define sets and parameters
    m = define_sets(m, data)
    m = define_parameters(m, data)

    # Optionally apply per-unit scaling - attempt to improve numerical stability
    if use_pu:
        m = apply_pu_scaling(m)

    # Define variables, expressions, and constraints
    m = define_variables(m)
    m = define_expressions(m)
    m = define_constraints(m)

    # Define objective functions
    m = define_objective_functions(m)

    return m


def configure_feasibility_model(m):
    """Configure model to accept a fixed emissions intensity baseline"""

    # Deactivate price targeting objective functions
    m.O_PERMIT_PRICE_TARGET.deactivate()
    m.O_WEIGHTED_RRN_PRICE_TARGET.deactivate()

    # Deactivate linearised constraints - those with discretised baseline
    m.C_FOC_1_LINEARISED.deactivate()
    m.C_PERMIT_MARKET_1_LINEARISED.deactivate()
    m.C_PERMIT_MARKET_3_LINEARISED.deactivate()

    # Deactivate constraints used for binary expansion linearisation
    m.C_Z_1_CONSTRAINT_1.deactivate()
    m.C_Z_1_CONSTRAINT_2.deactivate()
    m.C_Z_1_CONSTRAINT_3.deactivate()
    m.C_Z_1_CONSTRAINT_4.deactivate()
    m.C_Z_2_CONSTRAINT_1.deactivate()
    m.C_Z_2_CONSTRAINT_2.deactivate()
    m.C_Z_2_CONSTRAINT_3.deactivate()
    m.C_Z_2_CONSTRAINT_4.deactivate()

    return m


def configure_weighted_rrn_price_targeting_model(m):
    """Configure mode to target weighted RRN prices"""

    # Ensure weighted RRN price targeting objective is the only one active
    m.O_FEASIBILITY.deactivate()
    m.O_PERMIT_PRICE_TARGET.deactivate()

    # Deactivate first order condition and permit market constraints that have not been linearised
    m.C_FOC_1.deactivate()
    m.C_PERMIT_MARKET_1.deactivate()
    m.C_PERMIT_MARKET_3.deactivate()

    return m


def configure_model(m, options):
    """Activate / deactivate constraints and objectives depending on the case to be analysed"""

    if options['mode'] == 'feasibility':
        m = configure_feasibility_model(m)
    if options['mode'] == 'weighted_rrn_price_target':
        m = configure_weighted_rrn_price_targeting_model(m)
    else:
        raise Exception(f"Unexpected mode: {options['mode']}")

    return m


def solve_model(m):
    """Solve model"""

    opt = pyo.SolverFactory('cplex', solver_io='mps')

    # Solver options
    options = {
        'mip tolerances absmipgap': 1e-6,
        'emphasis mip': 1  # Emphasise feasibility
    }

    # Solve model
    opt.solve(m, keepfiles=False, tee=True, options=options)

    return m


def get_solution(m):
    """Return model solution"""
    return m


if __name__ == '__main__':
    # Data directories for scenarios
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'data')
    scenario_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, '1_create_scenarios',
                                      'output')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'utils', 'tmp')

    model_options = {
        'parameters': {
            'P_BINARY_EXPANSION_LARGEST_INTEGER': 10,
            'P_POLICY_FIXED_BASELINE': 1.2,
            'P_POLICY_PERMIT_PRICE_TARGET': 30,
            'P_POLICY_WEIGHTED_RRN_PRICE_TARGET': 20,
        },
        'mode': 'feasibility',
    }

    # Get case data
    case_data = get_case_data(data_directory, scenario_directory, tmp_directory, model_options, use_cache=True)

    # Construct model
    model = construct_model(case_data, use_pu=True)
    model = configure_model(model, model_options)
    model = solve_model(model)

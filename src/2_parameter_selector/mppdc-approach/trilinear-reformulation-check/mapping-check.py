
# coding: utf-8

# # Map variables
# Representation of trilinear monomials using convex envelopes is based on the method presented in [1]. The procedure to construct these envelopes requires that certain conditions must be satisified. For the case where: $\underline{x} \geq 0,\quad \underline{y} \geq 0, \quad \underline{z} \geq 0 $
# 
# Variables from the problem in question must be mapped onto $x, y, z$ such that the following relations apply:
# 
# $\bar{x}\underline{y}\underline{z} + \underline{x}\bar{y}\bar{z} \leq \underline{x}\bar{y}\underline{z} + \bar{x}\underline{y}\bar{z}$
# 
# and
# 
# $\bar{x}\underline{y}\underline{z} + \underline{x}\bar{y}\bar{z} \leq \bar{x}\bar{y}\underline{z} + \underline{x}\underline{y}\bar{x}$
# 
# The following cells first map problem variables $\phi, \tau, p_{g,s}$ onto $x, y, z$. Bounds for these variables are then given, and the mapping checked to ensure the above relations hold.
# 
# ## Import packages

# In[1]:


import os
import pandas as pd


# ## Paths to directories

# In[2]:


# Directory to core data directory
data_dir = os.path.join(os.path.join(os.path.curdir, os.pardir, os.pardir, os.pardir, os.pardir, 'data'))


# ## Load generator data

# In[3]:


# Generator technical parameters
df = pd.read_csv(os.path.join(data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'generators', 'generators.csv'), index_col='DUID')

# Note: MIN_GEN in the database takes on positive values for some generators. For the purposes of this 
# analysis it is assumed all generators have minimum output of 0. Update values for MIN_GEN to zero.
df['MIN_GEN'] = 0


# ## Variable bounds

# In[4]:


# Upper bound for emissions intensity baseline, phi
phi_max = 1.5

# Lower bound for emissions intensity baseline, phi
phi_min = 0

# Upper boud for permit price, tau
tau_max = 500

# Lower bound for permit price, tau
tau_min = 0


# ## Map variables

# In[5]:


# Map permit price --> x
x_up = tau_max
x_lo = tau_min

# Map baseline --> y
y_up = phi_max
y_lo = phi_min

# Map generator output --> z
z_up = df['REG_CAP'] /100
z_lo = df['MIN_GEN'] /100


# Check that relations hold for given mapping.

# In[6]:


# Relations to ensure appropriate mapping between problem variables and those in reformulation strategy
relation_1 = (x_up * y_lo * z_lo) + (x_lo * y_up * z_up) <= (x_lo * y_up * z_lo) + (x_up * y_lo * z_up)
relation_2 = (x_up * y_lo * z_lo) + (x_lo * y_up * z_up) <= (x_up * y_up * z_lo) + (x_lo * y_lo * z_up)

print('Both relations satisfied: {}'.format(relation_1.all() and relation_2.all()))


# ## References
# [1] - Meyer, C.A. and Floudas, C.A., 2004. Trilinear monomials with positive or negative domains: Facets of the convex and concave envelopes. In Frontiers in global optimization (pp. 327-352). Springer, Boston, MA.

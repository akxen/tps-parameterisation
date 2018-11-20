
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('generators.csv', index_col='DUID')
df['LO'] = 0


# In[30]:


phi_max = 1.5
phi_min = 0.8
tau_max = 300
tau_min = 0


x_up = tau_max
x_lo = tau_min
y_up = phi_max
y_lo = phi_min
z_up = df['REG_CAP']
z_lo = df['LO']

condition_1 = (x_up * y_lo * z_lo) + (x_lo * y_up * z_up) <= (x_lo * y_up * z_lo) + (x_up * y_lo * z_up)
condition_2 = (x_up * y_lo * z_lo) + (x_lo * y_up * z_up) <= (x_up * y_up * z_lo) + (x_lo * y_lo * z_up)

condition_1.all() and condition_2.all()


# In[ ]:





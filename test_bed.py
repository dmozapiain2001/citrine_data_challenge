
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




# ## Finding the files

# In[2]:


path_f=os.getcwd()

path_f_1=os.path.join(path_f, 'data')


names=[]
for files_txts in os.listdir(path_f_1):
    if files_txts.endswith(".csv"):
        #print(files_txts)
        names.append(files_txts)


# In[3]:


names


# In[4]:


path_train=os.path.join(path_f_1, names[0])
path_test=os.path.join(path_f_1, names[1])


# ## Import the csv using Pandas

# In[18]:


df_train=pd.read_csv(path_train)


# In[6]:


df_train


# ## Analyzing the Data

# In[7]:


df_train.iloc[0]


# In[8]:


list(df_train)


# In[9]:


df_train_fields=list(df_train)
print(type(df_train_fields))


# ## The features for each element are unique

# In[46]:


print(df_train['formulaA'])
uq_values=df_train['formulaA'].unique()
print(df_train['formulaA'].unique())
print(uq_values.size)
print(df_train['formulaA_elements_AtomicVolume'].unique())
print(df_train['formulaA_elements_AtomicVolume'].unique().size)


# In[11]:


tst=df_train.values
#print(type(tst[0,-1]))
#print((tst[0,-1]))
#print((tst[0,-2]))
#print(type(tst[0,-2]))
for x in tst[0,:]:
    print(type(x))
    print(x)




# In[27]:


stab_vector=df_train['stabilityVec'].values
print((stab_vector))
print((stab_vector.size))
print(type(stab_vector))
print(type(stab_vector[0]))
s=stab_vector[0]
a=np.fromstring(s[1:-1],sep=',').astype(int)
print(a)
print(type(a))


# ## Now do a for loop for each element in order to replace it for a numpy array and not a string and make it the "y"

# In[29]:


y=[]
for x in stab_vector:
    #print(x)
    a=np.fromstring(x[1:-1],sep=',').astype(int)
    y.append(a)
y=np.array(y)    


# In[32]:


print((y))
print(type(y))
print(type(y[0]))
print(y.shape)


# In[33]:


#how to transform the string into a numpy vec
#print(type(stab_vector[0]))
#stab_vector[0]=a
#print(type(stab_vector[0]))


# ## Notice how each of the compunds is always a stable output of the reaction
#     There is no need for a model these values then
#     We only need to predict the compounds that for from 10% to 90%

# In[78]:


print(y[:,0].mean())
print(y[:,-1].max())


# In[34]:


df_train['formulaA'].iloc[0]


# In[35]:


df_train['formulaB'].iloc[0]


# ## The order of the compounds does matter.
# 
# OsTi =/= TiOs

# In[15]:


print(df_train['formulaA'].iloc[2278],df_train['formulaB'].iloc[2278],stab_vector[2278])


# In[16]:


print(df_train['formulaA'].iloc[1509],df_train['formulaB'].iloc[1509],stab_vector[1509])


# ## Identifying the maximum values 

# In[61]:


df_values=df_train.values
print((df_values.shape))
print(type(df_values))


# In[66]:


max_val=df_values[:,2:-1].max(axis=0)
min_val=df_values[:,2:-1].min(axis=0)
print(max_val)
print(min_val)
print(max_val.shape)
print(max_val.max())
print(max_val.min())
print(max_val.argmin())


# In[62]:


df_values[:,2:-1].argmax(axis=0)


# ## Normalizing the input data

# In[70]:


norm_values=df_values[:,2:-1]/max_val
norm_values.max(axis=0)


# ## Analyzing the output of the Model
# 
#     - The output is a stability vector detailing the stable binary compounds that form on mixing. 
#     - It is a discretization of the 1D binary phase diagram at 10% intervals. 
#     - Meaning that the index of the vector specifies whether the compound is stable (1) or not (0)
#     - I believe then that I should establish 9 different models that determine whether each possible compund is likely to be stable or not based on the training data.
#     -Therefore this summarizes to a classification problem of 0 and 1 -use a neural network for classification and feed the data
#     - Check PCA, observe how it looks
#     
#     

# In[81]:


print(y.shape)
y[0]


# ## Plotting a histogram

# In[86]:


np.histogram(y[:,3])


# In[87]:


plt.hist(y[:,3])


# In[100]:


y[:,3].size


# In[110]:


import seaborn as sns

sns.set_style('darkgrid')
sns.distplot(y[:,3],kde=False)


# In[124]:


hist, bin_edges = np.histogram(y[:,3])
freq=hist/y[:,3].size
print(freq)
plt.bar(bin_edges[:-1], freq, width = .1)
plt.xlim(min(bin_edges), max(bin_edges))


# In[125]:



plt.hist(y[:,3], bins=100, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none');


# ## Pearson Correlation

# In[143]:


## creating a new DATA FRAME and assigning the values of the old one useful but not needed now
new_df=pd.DataFrame()
new_df[df_train_fields]=df_train[df_train_fields]
new_df.corr()


# In[144]:


df_train.corr()


# ### Next Steps
# 
#     1. Take PCA of Data, use element number instead of the string defining each element
#     2. Plot it and analyze it
#     3. Pearson Correlation in PCA

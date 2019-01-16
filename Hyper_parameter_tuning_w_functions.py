
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt

import csv
import numpy as np


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import os

import scores


# # Reading in the Data

# In[2]:


path_f=os.getcwd()

path_f_1=os.path.join(path_f, 'data')


names=[]
for files_txts in os.listdir(path_f_1):
    if files_txts.endswith(".csv"):
        #print(files_txts)
        names.append(files_txts)
        
path_train=os.path.join(path_f_1, names[0])
path_test=os.path.join(path_f_1, names[1])

df_train=pd.read_csv(path_train)
df_train.shape


# ## Data Manipulation
# 
#  - Transforming the outcome to a numpy vector

# In[3]:


stab_vector=df_train['stabilityVec'].values
y=[]
for x in stab_vector:
    #print(x)
    a=np.fromstring(x[1:-1],sep=',').astype(int)
    y.append(a)
y=np.array(y) 

df_tmp = pd.DataFrame(y, columns = ['A', 'A91B', 'A82B','A73B','A64B','A55B','A46B','A37B','A28B','A19B','B'])
stab_vec_list=[ 'A91B', 'A82B','A73B','A64B','A55B','A46B','A37B','A28B','A19B']

df_train=df_train.drop("stabilityVec",axis=1) #removing the results which originally are a string
feature_cols=list(df_train)

print(df_train.shape)

csvfile = csv.reader(open(path_train,'r'))
header = next(csvfile)

formulaA = []
formulaB = []

for row in csvfile:
    formulaA.append(row[0])
    formulaB.append(row[1])
formulas = formulaA + formulaB
formulas = list(set(formulas))

# -- /!\ need to save the dict as the ordering may difer at each run
formula2int = {}
int2formula = {}
for i, f in enumerate(formulas):
    formula2int[f] = i
    int2formula[i] = f

formulaAint = np.array([formula2int[x] for x in formulaA])
formulaBint = np.array([formula2int[x] for x in formulaB])

df_train['formulaA']=formulaAint
df_train['formulaB']=formulaBint

df_train=pd.concat([df_train, df_tmp],axis=1)
print(df_train.shape)


# ### Input Data Normalization and Feature Engineering
# 

# In[4]:


y_all=df_train[stab_vec_list]
df_tmp_stable = pd.DataFrame( columns = ['Stable_compunds'])
df_tmp_stable['Stable_compunds']=np.logical_not(y_all.sum(axis=1)==0).astype(int) ## A one means it has a stable value  a 0 

df_train=pd.concat([df_train, df_tmp_stable],axis=1)
print(df_train.shape)

df_train.head()


# Pearson Correlation to Identify the features that influence the most on the output 

# In[5]:


X_train_new=df_train[feature_cols]
y_new=df_train['Stable_compunds']

corr_df=pd.concat([X_train_new, y_new],axis=1)
a=corr_df.corr()
a['Stable_compunds'].hist(bins=7, figsize=(18, 12), xlabelsize=10)

print(a.shape)


# In[6]:


## Incorporating the Features that contribute the most based on a pearson correlation coefficient threshold
thr=.1
print(a[a['Stable_compunds'].abs()>thr].shape)
corr_variables=list(a[a['Stable_compunds'].abs()>thr].index)

del(corr_variables[-1])

print(corr_variables[-1])
print(a['avg_nearest_neighbor_distance_B'].iloc[-1])


# In[7]:


## Using Un-normalized data as input
X_train_new=df_train[corr_variables]

print(X_train_new.shape)





# Normalizing such that the magnitude is one
from sklearn.preprocessing import normalize

X_train_new_mag_1=normalize(X_train_new, axis=1) # vector magnitude is one
print(X_train_new_mag_1.shape)


## Normalizing by Zscore
from scipy.stats import zscore
X_train_new_Z_score=X_train_new.apply(zscore)
print(X_train_new_Z_score.shape)


## Normalizing by Zscore and then 0-1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_new_Z_0_1=min_max_scaler.fit_transform(X_train_new_Z_score)
print(X_train_new_Z_0_1.shape)


## Normalizing so that range is 0-1
X_train_new_0_1=min_max_scaler.fit_transform(X_train_new)
print(X_train_new_0_1.shape)


## Normalizing so that range is -1 to 1
from sklearn import preprocessing
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_new_m1_p1=max_abs_scaler.fit_transform(X_train_new)
print(X_train_new_m1_p1.shape)







# In[8]:


# Using PCA as input
X_train_4_PCA=df_train[feature_cols]
print(X_train_4_PCA.shape)
X_train_new_mag_1_PCA=normalize(X_train_4_PCA, axis=1)
print(X_train_new_mag_1_PCA.shape)

pca = PCA()
pca.fit(X_train_new_mag_1_PCA)
components = pca.components_[:20,:]
new_data = np.dot(X_train_new_mag_1_PCA, components.T)
X_train_new_PCA=new_data

print(X_train_new_PCA.shape)


# In[9]:


## Using Pearson Correlation in PCA
df1= pd.DataFrame(data=X_train_new_PCA)
print(df1.shape)


corr_df_PCA=pd.concat([df1, y_new],axis=1)

print(corr_df_PCA.shape)
a_PCA=corr_df_PCA.corr()
#a_PCA['Stable_compunds'].hist(bins=7, figsize=(18, 12), xlabelsize=10)
a_PCA.shape

thr=.01
print(a_PCA[a_PCA['Stable_compunds'].abs()>thr].shape)
corr_variables_PCA=list(a_PCA[a_PCA['Stable_compunds'].abs()>thr].index)

del(corr_variables_PCA[-1])
print(corr_variables_PCA)

X_train_PCA_PC=df1[corr_variables_PCA]
X_train_PCA_PC.shape


# ## Analyzing the Output Data Recieved
# 

# In[10]:


y = df_train[stab_vec_list]
print(y.sum(axis=1).value_counts())
## Observing how many element pairs produce a stable compound per % and overall
f,a = plt.subplots(3,3)
f.subplots_adjust(hspace=0.4, wspace=0.4)
a = a.ravel()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

y_all=df_train[stab_vec_list]

for count,ax in enumerate(a):
    
    y = df_train[stab_vec_list[count]]
    #print(y.value_counts())
    hist_1, bin_edges_1 = np.histogram(y)
    freq_1=hist_1/y.size
    
    ax.hist(y.values, bins=10, label='all elements')


    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    

#for count in range(9):

    #y = df_train[stab_vec_list[count]]
    stable_comp=df_train.loc[y==1,['formulaA','formulaB']]
    #print('Compound being analyzed is',stab_vec_list[count])
    stable_comp_num=stable_comp.values
    stable_A=np.unique(stable_comp_num[:,0])
    stable_B=np.unique(stable_comp_num[:,1])
    df_unique= pd.DataFrame()
    #print(df_unique.shape)

    y_unique= pd.DataFrame()
    
    for cnt in range(stable_A.shape[0]):
        #print(stable_A[cnt])
        df_tmp=y.loc[df_train['formulaA']==stable_A[cnt]]
        y_unique=pd.concat([y_unique, df_tmp],axis=0)
        #print(df_tmp.shape)
        #print(df_unique.shape)
    
    #print(y_unique.shape)

    for cnt in range(stable_B.shape[0]):
        #print(stable_A[cnt])
        df_tmp=y.loc[df_train['formulaB']==stable_B[cnt]]
        y_unique=pd.concat([y_unique, df_tmp],axis=0)

    
    y_unique=y.iloc[y_unique.index.unique()]
    ax.hist(y_unique.values, bins=10, label='stable elements')
    #print(y_unique.value_counts())

    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title()
    #print(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    
    y_stable=y_unique.loc[np.logical_not(y_all.sum(axis=1)==0)]
    ax.hist(y_stable.values, bins=10, label='stable elements')
    #print(y_stable.value_counts())

    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title()
    #print(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    ax.legend(loc='upper right')
    
    
    ax.legend(loc='upper right')

    


plt.tight_layout()


# ### First we will build a model to determine if the input elements will produce at least one stable compound

# In[11]:


y = df_train[stab_vec_list]
print(y.sum(axis=1).value_counts())

print('These are example of elements that produce no stable compounds')

(df_train.loc[y_all.sum(axis=1)==0].head())


# In[12]:


y_new=df_train['Stable_compunds']
y_new.mean()


# # Model Generation

# In[ ]:


## test-train split
X_train, X_test, y_train, y_test = train_test_split(X_train_new_Z_score, y_new,
                                                    test_size=.1,
                                                    shuffle=True,
                                                    random_state=42)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
#X_train.head()


# ## Random Forest

# In[ ]:


# Hyper-Parameter Search Grid Using 10-Fold CV and Test
print(' -- Random Forest --')

n_estimators = [50,100,200]
criterion=['gini', 'entropy']
bootstrap= [True, False]
max_depth=[10, 20,30]


df_results=scores.hp_tune_Random_Forest(X_train,y_train,X_test,y_test,10,n_estimators,criterion,bootstrap,max_depth)


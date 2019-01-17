
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt

import csv
import numpy as np


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import scores


# # Reading in the Data

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
print('Training Data is being read ....')
#  - Transforming the outcome to a numpy vector

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
print('Training Data has been read and feature engineering is being performed....')

y_all=df_train[stab_vec_list]
df_tmp_stable = pd.DataFrame( columns = ['Stable_compunds'])
df_tmp_stable['Stable_compunds']=np.logical_not(y_all.sum(axis=1)==0).astype(int) ## A one means it has a stable value  a 0 

df_train=pd.concat([df_train, df_tmp_stable],axis=1)
print(df_train.shape)

df_train.head()

# Pearson Correlation to Identify the features that influence the most on the output 
print('Pearson Correlation has been calculated to build the model in the most relevant features ....')

X_train_new=df_train[feature_cols]
y_new=df_train['Stable_compunds']

corr_df=pd.concat([X_train_new, y_new],axis=1)
a=corr_df.corr()
#a['Stable_compunds'].hist(bins=7, figsize=(18, 12), xlabelsize=10)

## Incorporating the Features that contribute the most based on a pearson correlation coefficient threshold

thr=.1

corr_variables=list(a[a['Stable_compunds'].abs()>thr].index)

del(corr_variables[-1])


print('Pearson Correlation has identified', len(corr_variables), 'with ', str(thr) )

## Normalization of Input Data

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



## Normalizing so that range is 0-1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_new_0_1=min_max_scaler.fit_transform(X_train_new)
print(X_train_new_0_1.shape)


## Normalizing so that range is -1 to 1
from sklearn import preprocessing
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_new_m1_p1=max_abs_scaler.fit_transform(X_train_new)
print(X_train_new_m1_p1.shape)


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


## Using Pearson Correlation in PCA
df1= pd.DataFrame(data=X_train_new_PCA)
print(df1.shape)


corr_df_PCA=pd.concat([df1, y_new],axis=1)

print(corr_df_PCA.shape)
a_PCA=corr_df_PCA.corr()
#a_PCA['Stable_compunds'].hist(bins=7, figsize=(18, 12), xlabelsize=10)


thr=.01
corr_variables_PCA=list(a_PCA[a_PCA['Stable_compunds'].abs()>thr].index)

del(corr_variables_PCA[-1])


X_train_PCA_PC=df1[corr_variables_PCA]



# ### First we will build a model to determine if the input elements will produce at least one stable compound

y_new=df_train['Stable_compunds']


# # Model Generation

print('Training Model Using Z-normalized Data')
## test-train split
X_train, X_test, y_train, y_test = train_test_split(X_train_new_Z_score, y_new,
                                                    test_size=.1,
                                                    shuffle=True,
                                                    random_state=42)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
# Hyper-Parameter Search Grid Using 10-Fold CV and Test
print(' -- Random Forest --')

n_estimators = [50,100,200]
criterion=['gini', 'entropy']
bootstrap= [True, False]
max_depth=[1,2,3,5,10, 20,30]

min_samples_splits=[10, 20, 50, 60, 90 ,120]
min_samples_leafs=[2 ,5, 10, 25 ,50, 90, 120]
min_impurity_splits=[5e-7 ,1e-6, 1e-5]

df_results_RF=scores.hp_tune_Random_Forest(X_train,y_train,X_test,y_test,10,n_estimators,criterion,bootstrap,max_depth,min_samples_splits,min_samples_leafs,min_impurity_splits)





print('This are the best Parameters for Random Forest:')
print(df_results_RF[df_results_RF['test_accuracy']==df_results_RF['test_accuracy'].max()])


# # Decision Trees


# Hyper-Parameter Search Grid Using 10-Fold CV and Test
print(' -- Decision Trees --')

criterion=['gini', 'entropy']
max_depth=[10,20,30,50]
split=['random','best']

min_samples_splits=[10, 20 ,50 ,60 ,90, 120]
min_samples_leafs=[2, 5, 10, 25, 50 ,90 ,120]
min_impurity_splits=[5e-7, 1e-6, 1e-5]

df_results_DT=scores.hp_tune_Decision_tree(X_train,y_train,X_test,y_test,10,criterion,max_depth,split,min_samples_splits,min_samples_leafs,min_impurity_splits)

print('This are the best Parameters for Decision Tree:')
print(df_results_DT[df_results_DT['test_accuracy']==df_results_DT['test_accuracy'].max()])


# # KNN 


# Hyper-Parameter Search Grid Using 10-Fold CV and Test
print(' -- KNN --')

criterion=['distance', 'uniform']
neighbors=[2,3,5,7,10]
distances = [1, 2, 3, 4, 5]

df_results_KNN=scores.hp_tune_KNN(X_train,y_train,X_test,y_test,10,criterion,neighbors,distances)




print('This are the best Parameters for KNN :')
print(df_results_KNN[df_results_KNN['test_accuracy']==df_results_KNN['test_accuracy'].max()])


# # SVM


# Hyper-Parameter Search Grid Using 10-Fold CV and Test
print(' -- SVM --')

kernel=['rbf', 'linear', 'poly', 'sigmoid']
gammas = [0.1,.5, 1]
cs = [0.1,.5, 1, 3,10]

df_results_SVM=scores.hp_tune_SVM(X_train,y_train,X_test,y_test,10,kernel,gammas,cs)



print('This are the best Parameters for SVM :')
print(df_results_SVM[df_results_SVM['test_accuracy']==df_results_SVM['test_accuracy'].max()])


# # Logistic Regression

# Hyper-Parameter Search Grid Using 10-Fold CV and Test
print(' -- Logistic Regression --')

criterion=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

df_results_log_reg=scores.hp_tune_log_reg(X_train,y_train,X_test,y_test,10,criterion)

print('This are the best Parameters for SVM :')
print(df_results_log_reg[df_results_log_reg['test_accuracy']==df_results_log_reg['test_accuracy'].max()])



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


# # Data Manipulation

# In[3]:


#Transforming the outcome to a numpy vector
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
df_train.shape


# In[4]:


#Transforming the Formulas to integers
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
df_train.shape


# # Input and Output of the Model
# 
# Analyzing the output first

# In[5]:


# Observing How many pair produce at least one stable compound
y = df_train[stab_vec_list]
print(y.sum(axis=1).value_counts())
y.sum(axis=1).hist(bins=7, figsize=(18, 12), xlabelsize=10)


# In[6]:


## Observing how many element pairs produce a stable compound per % and overall
f,a = plt.subplots(3,3)
f.subplots_adjust(hspace=0.4, wspace=0.4)
a = a.ravel()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

y_all=df_train[stab_vec_list]

for count,ax in enumerate(a):
    
    y = df_train[stab_vec_list[count]]
    print(y.value_counts())
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
    print(y_unique.value_counts())

    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title()
    #print(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    
    y_stable=y_unique.loc[np.logical_not(y_all.sum(axis=1)==0)]
    ax.hist(y_stable.values, bins=10, label='stable elements')
    print(y_stable.value_counts())

    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title()
    #print(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    ax.legend(loc='upper right')
    
    
    ax.legend(loc='upper right')

    


plt.tight_layout()


# ## Building model to determine if the elements will produce at least one stable compound

# In[7]:



print(y_all.sum(axis=1).value_counts())

print('These are example of elements that produce no stable compounds')

(df_train.loc[y_all.sum(axis=1)==0].head())


# In[8]:


f,a = plt.subplots(3,3)
f.subplots_adjust(hspace=0.4, wspace=0.4)
a = a.ravel()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

y_all=df_train[stab_vec_list]

for count,ax in enumerate(a):
    
    y = df_train[stab_vec_list[count]]
    print(y.value_counts())
    hist_1, bin_edges_1 = np.histogram(y)
    freq_1=hist_1/y.size
    
    ax.hist(y.values, bins=10, label='all elements')


    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    

#for count in range(9):

    y = df_train[stab_vec_list[count]]
    stable_comp=df_train.loc[y==1,['formulaA','formulaB']]
    #print('Compound being analyzed is',stab_vec_list[count])
    stable_comp_num=stable_comp.values
    stable_A=np.unique(stable_comp_num[:,0])
    stable_B=np.unique(stable_comp_num[:,1])
    df_unique= pd.DataFrame()
    #print(df_unique.shape)

   
    y_stable=y.loc[np.logical_not(y_all.sum(axis=1)==0)]
    ax.hist(y_stable.values, bins=10, label='stable elements')
    print(y_stable.value_counts())

    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title()
    #print(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    ax.legend(loc='upper right')

    




plt.tight_layout()


# ### PCA Visualization

# In[9]:


X = df_train[feature_cols]

from sklearn.preprocessing import normalize

X_norm = normalize(X, axis=1)



from scipy.stats import zscore

X_norm_zscore=X.apply(zscore)

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_norm_0_1=min_max_scaler.fit_transform(X)


# In[10]:


#Visualiziting the input using PCA
pca = PCA()

pca.fit(X_norm)

explained_var = pca.explained_variance_
print('top 10 explained variance: ', explained_var[:10])

#pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

components = pca.components_[:20,:]



# In[11]:


new_data = np.dot(X_norm, components.T)
print(new_data.shape)

indexes_0 = y_all.sum(axis=1)==0
indexes_1=np.logical_not(indexes_0)
plt.plot(new_data[indexes_0,0], new_data[indexes_0,1], 'b.', linestyle='', label='No Stable Compounds')
plt.plot(new_data[indexes_1,0], new_data[indexes_1,1], 'r.', linestyle='', label='At least One Stable ')
plt.title('visualization of the first class of the stability Vec on the two main components.')
plt.legend()


# In[12]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax = fig.add_subplot(111, projection='3d')



ax.scatter(new_data[indexes_0,0], new_data[indexes_0,1], new_data[indexes_0,2], label='No Stable Compounds')
ax.scatter(new_data[indexes_1,0], new_data[indexes_1,1], new_data[indexes_1,2], label='At least One Stable ')

ax.set_xlabel('PC1 Label')
ax.set_ylabel('PC2 Label')
ax.set_zlabel('PC3 Label')
ax.legend()
plt.show()


# In[13]:


## Extending the data series to encompass the information to determine if the elements procure a stable compund or not
df_train.head()


# In[14]:


df_tmp_stable = pd.DataFrame( columns = ['Stable_compunds'])
df_tmp_stable['Stable_compunds']=np.logical_not(y_all.sum(axis=1)==0).astype(int) ## A one means it has a stable value  a 0 
print(df_tmp_stable['Stable_compunds'].value_counts())
df_tmp_stable.head()


# In[15]:


df_train=pd.concat([df_train, df_tmp_stable],axis=1)
df_train.head()


# In[16]:


## Using Un-normalized data as input
X_train_new=df_train[feature_cols]



y_new=df_tmp_stable['Stable_compunds']


print(X_train_new.shape)
print(y_new.shape)


# In[17]:


# Using PCA as input
X_train_new=new_data

y_new=df_tmp_stable['Stable_compunds']

print(X_train_new.shape)
print(y_new.shape)


# In[18]:


# Normalizing such that the magnitude is one
X_train_new=X_norm # vector magnitude is one
y_new=df_tmp_stable['Stable_compunds']

print(X_train_new.shape)
print(y_new.shape)


# In[19]:


## Normalizing by Zscore
from scipy.stats import zscore
X_train_new=df_train[feature_cols]
X_train_new=X_train_new.apply(zscore)
y_new=df_tmp_stable['Stable_compunds']

print(X_train_new.shape)
print(y_new.shape)

#X_train_new.mean()
#X_train_new.std()

#X_train_new.max()


# In[36]:


## Normalizing by Zscore and then 0-1
from scipy.stats import zscore
X_train_new=df_train[feature_cols]
X_train_new=X_train_new.apply(zscore)

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()



#print(type(X_train_new))

X_train_new=min_max_scaler.fit_transform(X_train_new)


y_new=df_tmp_stable['Stable_compunds']

print(X_train_new.shape)
print(y_new.shape)

#X_train_new.mean()
#X_train_new.std()

#np.max(X_train_new,axis=1)


# In[21]:


## Normalizing so that range is 0-1

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


X_train_new=df_train[feature_cols]
#print(type(X_train_new))

X_train_new=min_max_scaler.fit_transform(X_train_new)

#print(type(X_train_new))

y_new=df_tmp_stable['Stable_compunds']

print(X_train_new.shape)
print(y_new.shape)

#X_train_new.mean()
#X_train_new.std()


# In[22]:


## Normalizing so that range is -1 to 1

from sklearn import preprocessing


max_abs_scaler = preprocessing.MaxAbsScaler()


X_train_new=df_train[feature_cols]
#print(type(X_train_new))

X_train_new=max_abs_scaler.fit_transform(X_train_new)

#print(type(X_train_new))

y_new=df_tmp_stable['Stable_compunds']

print(X_train_new.shape)
print(y_new.shape)

#X_train_new.mean()
#X_train_new.std()

#np.mean(X_train_new,axis=1)


# In[23]:


## test-train split
X_train, X_test, y_train, y_test = train_test_split(X_train_new, y_new,
                                                    test_size=0.33,
                                                    shuffle=True,
                                                    random_state=42)



# In[24]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics


# In[25]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate model
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)
#Make prediction using fitted model
y_pred = logreg.predict(X_test)


accuracy = np.mean((y_test == y_pred))

precision = metrics.precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
F1 = f1_score(y_test, y_pred, average='binary')

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)


#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)
print(metrics.precision_score(y_test, y_pred))


# In[26]:


precision,recall,F1,accuracy,confusion=scores.scores(y_test,y_pred)

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)

print('Confusion matrix')
print(confusion)


# In[27]:


# -- test with KNN
print(' -- KNN --')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')

accuracy = np.mean((y_test == y_pred))

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)

#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
#print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)


# In[28]:


# test with random forest
print(' -- Random Forest --')
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')

accuracy = np.mean((y_test == y_pred))

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)

#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
#print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)



# Next Steps: Incorporate SVM, Decision Trees 

# In[29]:


# test with Support Vector Machines
print(' -- Support Vector Machines --')

import sklearn.svm

svc_c = sklearn.svm.SVC()


svc_c.fit(X_train, y_train)
y_pred = svc_c.predict(X_test)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')

accuracy = np.mean((y_test == y_pred))

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)

#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
#print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)



# In[30]:


# test with Decision Trees
print(' -- Decision Trees --')

import sklearn.tree

#criterion={'gini','entropy'}
#splitter={'best','random'}

decission_c = sklearn.tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                                   max_features=None, max_leaf_nodes=None,
                                                   min_impurity_split=1e-07, min_samples_leaf=1,
                                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                   presort=False, random_state=None, splitter='random')


decission_c.fit(X_train, y_train)
y_pred = decission_c.predict(X_test)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')

accuracy = np.mean((y_test == y_pred))

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)

#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
#print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print('Area under curve:', roc_auc)


# In[31]:


# test with Naive Bayes
print(' -- Naive Bayes --')

import sklearn.naive_bayes

naive_B_c = sklearn.naive_bayes.GaussianNB()


naive_B_c.fit(X_train, y_train)
y_pred = naive_B_c.predict(X_test)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')

accuracy = np.mean((y_test == y_pred))

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)

#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
#print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)



# ### Assesing the Model Performance Code

# In[32]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

precision = precision_score(y_test.values, y_pred, average='micro')
recall = recall_score(y_test.values, y_pred, average='micro')
F1 = f1_score(y_test.values, y_pred, average='micro')

accuracy = np.mean((y_test.values == y_pred))

print('precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy)


# In[33]:



#Confusion Metric
confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix')
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print('accuracy score',(TP + TN) / float(TP + TN + FP + FN))
#print(metrics.accuracy_score(y_test, y_pred))

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('classification error', classification_error)
#print(1 - metrics.accuracy_score(y_test, y_pred_class))

sensitivity = TP / float(FN + TP)

print('sensitivity',sensitivity)


specificity = TN / (TN + FP)

print('specificity',specificity)

false_positive_rate = FP / float(TN + FP)

print('false positive rate',false_positive_rate)
#print(1 - specificity)

precision = TP / float(TP + FP)

print('precision',precision)


# In[34]:


# examine the class distribution of the testing set
print('Percentage of 1s is ', y_test.mean())
print('Percentage of 0s is ', 1-y_test.mean())


#Comparing against a null predictor
print('using a null predictor', max(y_test.mean(), 1 - y_test.mean()))


# ## Histogram Making Code

# In[35]:


f,a = plt.subplots(3,3)
f.subplots_adjust(hspace=0.4, wspace=0.4)
a = a.ravel()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for count,ax in enumerate(a):
    
    y = df_train[stab_vec_list[count]]
    print(y.value_counts())
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

    
    y.iloc[y_unique.index.unique()]
    ax.hist(y.iloc[y_unique.index.unique()].values, bins=10, label='stable elements')
    print(y.iloc[y_unique.index.unique()].value_counts())

    #ax.xlim(min(bin_edges), max(bin_edges))
    #ax.title()
    #print(stab_vec_list[count])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    
    
    
    
    
    ax.legend(loc='upper right')

    


plt.tight_layout()


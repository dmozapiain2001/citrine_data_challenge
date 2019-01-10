import csv
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

csvfile = csv.reader(open('training_data.csv','r'))
header = next(csvfile)

data = []
formulaA = []
formulaB = []
stabilityVec = []
for row in csvfile:
    formulaA.append(row[0])
    formulaB.append(row[1])
    data.append(np.array([np.float(x) for x in row[2:-1]]))
    stabilityVec.append(np.array([np.float(x) for x in row[-1][1:-1].split(',')]))

stabilityVec = np.array(stabilityVec)

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
data = np.array(data)
data = np.concatenate((formulaAint[:,None], formulaBint[:,None], data), axis=1)

data = normalize(data, axis=1)

pca = PCA()
pca.fit(data)

explained_var = pca.explained_variance_
print 'top 10 explained variance: ', explained_var[:10]

#pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

components = pca.components_[:2,:]

# -- some viz
new_data = np.dot(data, components.T)
cl = 2
indexes_0 = stabilityVec[:,2]==0.0
indexes_1 = stabilityVec[:,2]==1.0
plt.plot(new_data[indexes_0,0], new_data[indexes_0,1], 'b.', linestyle='', label='0')
plt.plot(new_data[indexes_1,0], new_data[indexes_1,1], 'r.', linestyle='', label='1')
plt.title('visualization of the first class of the stability Vec on the two main components.')
plt.legend()
#plt.show()

# -- some stats on the targets
# first and last digits are always ones.
for i in range(9):
    print 'class ', i, '  ', np.histogram(stabilityVec[:,i+1], bins=2)


# multi-label classification problem
y_true = stabilityVec[:,1:-1]
X_train, X_test, y_train, y_test = train_test_split(data, y_true,
                                                    test_size=0.33,
                                                    shuffle=True,
                                                    random_state=42)

# -- test with KNN
print ' -- KNN --'
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

accuracy = np.mean((y_test == y_pred).all(axis=1))

print 'precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy


# test with random forest
print ' -- Random Forest --'
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')

accuracy = np.mean((y_test == y_pred).all(axis=1))

print 'precision: ', precision, '  recall: ', recall, '  F1: ', F1, '  accuracy: ', accuracy





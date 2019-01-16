from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import pandas as pd

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
import sklearn.tree
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score 
from sklearn.metrics import roc_curve, auc

def scores(y_test,y_pred):

	precision = metrics.precision_score(y_test, y_pred, average='binary')
	recall = metrics.recall_score(y_test, y_pred, average='binary')
	F1 = metrics.f1_score(y_test, y_pred, average='binary')
	accuracy=metrics.accuracy_score(y_test, y_pred)
	confusion = metrics.confusion_matrix(y_test, y_pred)

	false_positive_rate, true_positive_rate, thresholds =metrics.roc_curve(y_test, y_pred)
	roc_auc = metrics.auc(false_positive_rate, true_positive_rate)


	return precision,recall,F1,accuracy,confusion,roc_auc


def grid_search_wrapper(X_train,y_train,X_test,y_test,clf,param_grid,scorers,cv_num,refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """

    
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=cv_num, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


def hp_tune_Random_Forest(X_train,y_train,X_test,y_test,Cv_folds,n_estimators,criterion,bootstrap,max_depth):
	print(' -- Random Forest --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	features=[]
	for estimator in n_estimators:
		for cr in criterion:
			for boots in bootstrap:
				for max_d in max_depth:
					rfc = RandomForestClassifier(n_estimators=estimator,criterion=cr,bootstrap=boots,max_depth=max_d, class_weight={0:1-y_train.mean(), 1:y_train.mean()},random_state=0,n_jobs=-1)
					all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
					train_results_mean.append(all_accuracies.mean())
					train_results_std.append(all_accuracies.std())
    
					rfc.fit(X_train, y_train)
    
					train_pred = rfc.predict(X_train)
    
					precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
					train_results_auc.append(roc_auc)
    
					y_pred = rfc.predict(X_test)
    
					precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)
					test_results_auc.append(roc_auc)
					test_accu.append(accuracy)
					test_precision.append(precision)
					features.append([estimator,cr,boots,max_d])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df7],axis=1)
	return df_results

def hp_tune_Decision_tree(X_train,y_train,X_test,y_test,Cv_folds,criterion,max_depth,split):
	print(' -- Decision Tree --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	features=[]
	for cr in criterion:
		for max_d in max_depth:
			for sp in split:
				rfc = sklearn.tree.DecisionTreeClassifier(class_weight={0:1-y_train.mean(), 1:y_train.mean()}, criterion=cr,max_depth=max_d,random_state=0, splitter=sp)
				all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
				train_results_mean.append(all_accuracies.mean())
				train_results_std.append(all_accuracies.std())
    
				rfc.fit(X_train, y_train)
    
				train_pred = rfc.predict(X_train)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
				train_results_auc.append(roc_auc)
    
				y_pred = rfc.predict(X_test)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)
				test_results_auc.append(roc_auc)
				test_accu.append(accuracy)
				test_precision.append(precision)
				features.append([cr,max_d,sp])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df7],axis=1)
	return df_results

def hp_tune_KNN(X_train,y_train,X_test,y_test,Cv_folds,criterion,neighbors,distances):

	print(' -- KNN Classifier --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	features=[]
	for cr in criterion:
		for n_n in neighbors:
			for dp in distances:
				rfc =KNeighborsClassifier(algorithm='auto',metric='minkowski',n_jobs=-1, n_neighbors=n_n, p=dp,weights=cr)
				all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
				train_results_mean.append(all_accuracies.mean())
				train_results_std.append(all_accuracies.std())
    
				rfc.fit(X_train, y_train)
    
				train_pred = rfc.predict(X_train)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
				train_results_auc.append(roc_auc)
    
				y_pred = rfc.predict(X_test)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)
				test_results_auc.append(roc_auc)
				test_accu.append(accuracy)
				test_precision.append(precision)
				features.append([cr,n_n,dp])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df7],axis=1)
	return df_results

def hp_tune_SVM(X_train,y_train,X_test,y_test,Cv_folds,criterion,gammas,cs):

	print(' -- SVM Classifier --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	features=[]
	for cr in criterion:
		for g in gammas:
			for c in cs:
				rfc =sklearn.svm.SVC(kernel=cr, gamma=g,C=c,random_state=0,class_weight={0:1-y_train.mean(), 1:y_train.mean()})
				all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
				train_results_mean.append(all_accuracies.mean())
				train_results_std.append(all_accuracies.std())
    
				rfc.fit(X_train, y_train)
    
				train_pred = rfc.predict(X_train)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
				train_results_auc.append(roc_auc)
    
				y_pred = rfc.predict(X_test)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)
				test_results_auc.append(roc_auc)
				test_accu.append(accuracy)
				test_precision.append(precision)
				features.append([cr,g,c])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df7],axis=1)
	return df_results


def hp_tune_log_reg(X_train,y_train,X_test,y_test,Cv_folds,criterion,):

	print(' -- Logistic Regression --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	features=[]
	for cr in criterion:
		rfc =LogisticRegression(C=1,class_weight={0:1-y_train.mean(),1:y_train.mean()},random_state=0,solver=cr,max_iter=1000)
		all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
		train_results_mean.append(all_accuracies.mean())
		train_results_std.append(all_accuracies.std())
    
		rfc.fit(X_train, y_train)
    
		train_pred = rfc.predict(X_train)
    
		precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
		train_results_auc.append(roc_auc)
    
		y_pred = rfc.predict(X_test)
    
		precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)
		test_results_auc.append(roc_auc)
		test_accu.append(accuracy)
		test_precision.append(precision)
		features.append([cr,g,c])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df7],axis=1)
	return df_results
        
    


    

    
    




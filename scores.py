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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

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


def hp_tune_Random_Forest(X_train,y_train,X_test,y_test,Cv_folds,n_estimators,criterion,bootstrap,max_depth,min_samples_splits,min_samples_leafs,min_impurity_splits):
	print(' -- Random Forest --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for estimator in n_estimators:
		for cr in criterion:
			for boots in bootstrap:
				for max_d in max_depth:
					for min_sample_sp in min_samples_splits:
						for min_samples_le in min_samples_leafs:
							for min_impurity_sp in min_impurity_splits:

								rfc = RandomForestClassifier(n_estimators=estimator,criterion=cr,bootstrap=boots,max_depth=max_d, class_weight={0:y_train.mean(), 1:1-y_train.mean()},min_samples_split=min_sample_sp,min_samples_leaf=min_samples_le,min_impurity_decrease=min_impurity_sp,random_state=0,n_jobs=-1)
								all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds,scoring='roc_auc')
								train_results_mean.append(all_accuracies.mean())
								train_results_std.append(all_accuracies.std())
    
								rfc.fit(X_train, y_train)
    
								train_pred = rfc.predict(X_train)
    
								precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
								train_recall.append(recall)
								train_results_auc.append(roc_auc)
    
								y_pred = rfc.predict(X_test)
    
								precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

								test_recall.append(recall)
								test_results_auc.append(roc_auc)
								test_accu.append(accuracy)
								test_precision.append(precision)
								features.append([estimator,cr,boots,max_d,min_sample_sp,min_samples_le,min_impurity_sp])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results

def hp_tune_Decision_tree(X_train,y_train,X_test,y_test,Cv_folds,criterion,max_depth,split,min_samples_splits,min_samples_leafs,min_impurity_splits):
	print(' -- Decision Tree --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for cr in criterion:
		for max_d in max_depth:
			for sp in split:
				for min_sample_sp in min_samples_splits:
						for min_samples_le in min_samples_leafs:
							for min_impurity_sp in min_impurity_splits:
								rfc = sklearn.tree.DecisionTreeClassifier(class_weight={0:y_train.mean(), 1:1-y_train.mean()}, criterion=cr,max_depth=max_d,random_state=0, splitter=sp,min_samples_split=min_sample_sp,min_samples_leaf=min_samples_le,min_impurity_decrease=min_impurity_sp)
								all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
								train_results_mean.append(all_accuracies.mean())
								train_results_std.append(all_accuracies.std())

    
								rfc.fit(X_train, y_train)
    
								train_pred = rfc.predict(X_train)
    
								precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
								train_results_auc.append(roc_auc)
								train_recall.append(recall)
    
								y_pred = rfc.predict(X_test)
    
								precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

								test_recall.append(recall)
								test_results_auc.append(roc_auc)
								test_accu.append(accuracy)
								test_precision.append(precision)
								features.append([cr,max_d,sp,min_sample_sp,min_samples_le,min_impurity_sp])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results

def hp_tune_KNN(X_train,y_train,X_test,y_test,Cv_folds,criterion,neighbors,distances):

	print(' -- KNN Classifier --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
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
				train_recall.append(recall)
    
				y_pred = rfc.predict(X_test)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

				test_recall.append(recall)
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
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results

def hp_tune_SVM(X_train,y_train,X_test,y_test,Cv_folds,criterion,gammas,cs):

	print(' -- SVM Classifier --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for cr in criterion:
		for g in gammas:
			for c in cs:
				rfc =sklearn.svm.SVC(kernel=cr, gamma=g,C=c,random_state=0,class_weight={0:y_train.mean(), 1:1-y_train.mean()})
				all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds)
				train_results_mean.append(all_accuracies.mean())
				train_results_std.append(all_accuracies.std())
    
				rfc.fit(X_train, y_train)
    
				train_pred = rfc.predict(X_train)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
				train_results_auc.append(roc_auc)
				train_recall.append(recall)
    
				y_pred = rfc.predict(X_test)
    
				precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

				test_recall.append(recall)
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
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results


def hp_tune_log_reg(X_train,y_train,X_test,y_test,Cv_folds,criterion,):

	print(' -- Logistic Regression --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
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
		train_recall.append(recall)
    
		y_pred = rfc.predict(X_test)
    
		precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

		test_recall.append(recall)
		test_results_auc.append(roc_auc)
		test_accu.append(accuracy)
		test_precision.append(precision)
		features.append([cr])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results


def hp_tune_ADAboosting_RF(X_train,y_train,X_test,y_test,Cv_folds,n_estimators,criterion,bootstrap,max_depth,min_samples_splits,min_samples_leafs,min_impurity_splits,num_estimators,learning_reates):
	print(' -- ADA Boosting Random Forest --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for estimator in n_estimators:
		for cr in criterion:
			for boots in bootstrap:
				for max_d in max_depth:
					for min_sample_sp in min_samples_splits:
						for min_samples_le in min_samples_leafs:
							for min_impurity_sp in min_impurity_splits:
								for num_e in num_estimators:
									for lr in learning_reates:
										rfc = RandomForestClassifier(n_estimators=estimator,criterion=cr,bootstrap=boots,max_depth=max_d, class_weight={0:y_train.mean(), 1:1-y_train.mean()},min_samples_split=min_sample_sp,min_samples_leaf=min_samples_le,min_impurity_decrease=min_impurity_sp,random_state=0,n_jobs=-1)
										clf = AdaBoostClassifier(base_estimator=rfc, n_estimators=num_e,learning_rate=lr)

										all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=Cv_folds,scoring='roc_auc')
										train_results_mean.append(all_accuracies.mean())
										train_results_std.append(all_accuracies.std())
    
										clf.fit(X_train, y_train)
    
										train_pred = clf.predict(X_train)
    
										precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
										train_recall.append(recall)
										train_results_auc.append(roc_auc)
    
										y_pred = clf.predict(X_test)
    
										precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

										test_recall.append(recall)
										test_results_auc.append(roc_auc)
										test_accu.append(accuracy)
										test_precision.append(precision)
										features.append([estimator,cr,boots,max_d,min_sample_sp,min_samples_le,min_impurity_sp,num_e,lr])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results


def hp_tune_ADABoost_Decision_tree(X_train,y_train,X_test,y_test,Cv_folds,criterion,max_depth,split,min_samples_splits,min_samples_leafs,min_impurity_splits,num_estimators,learning_reates):
	print(' -- ADABoosting Decision Tree --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for cr in criterion:
		for max_d in max_depth:
			for sp in split:
				for min_sample_sp in min_samples_splits:
						for min_samples_le in min_samples_leafs:
							for min_impurity_sp in min_impurity_splits:
								for num_e in num_estimators:
									for lr in learning_reates:
										rfc = sklearn.tree.DecisionTreeClassifier(class_weight={0:y_train.mean(), 1:1-y_train.mean()}, criterion=cr,max_depth=max_d,random_state=0, splitter=sp,min_samples_split=min_sample_sp,min_samples_leaf=min_samples_le,min_impurity_decrease=min_impurity_sp)
										clf = AdaBoostClassifier(base_estimator=rfc, n_estimators=num_e,learning_rate=lr)

										all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=Cv_folds)
										train_results_mean.append(all_accuracies.mean())
										train_results_std.append(all_accuracies.std())

    
										clf.fit(X_train, y_train)
    
										train_pred = clf.predict(X_train)
    
										precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
										train_results_auc.append(roc_auc)
										train_recall.append(recall)
    
										y_pred = clf.predict(X_test)
    	
										precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

										test_recall.append(recall)
										test_results_auc.append(roc_auc)
										test_accu.append(accuracy)
										test_precision.append(precision)
										features.append([cr,max_d,sp,min_sample_sp,min_samples_le,min_impurity_sp,num_e,lr])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results

def hp_tune_GRADBoost_Decision_tree(X_train,y_train,X_test,y_test,Cv_folds,max_depth,min_samples_splits,min_samples_leafs,min_impurity_splits,num_estimators,learning_reates):
	print(' -- ADABoosting Decision Tree --')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for max_d in max_depth:
		for min_sample_sp in min_samples_splits:
			for min_samples_le in min_samples_leafs:
				for min_impurity_sp in min_impurity_splits:
					for num_e in num_estimators:
						for lr in learning_reates:
							clf  = GradientBoostingClassifier(n_estimators=num_e, learning_rate=lr,min_samples_split=min_sample_sp,min_samples_leaf=min_samples_le,max_depth=max_d,random_state=0)

							all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=Cv_folds)
							train_results_mean.append(all_accuracies.mean())
							train_results_std.append(all_accuracies.std())

    
							clf.fit(X_train, y_train)
    
							train_pred = clf.predict(X_train)
    
							precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
							train_results_auc.append(roc_auc)
							train_recall.append(recall)
    
							y_pred = clf.predict(X_test)
    	
							precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

							test_recall.append(recall)
							test_results_auc.append(roc_auc)
							test_accu.append(accuracy)
							test_precision.append(precision)
							features.append([max_d,min_sample_sp,min_samples_le,min_impurity_sp,num_e,lr])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results

def hp_tune_Extra_trees(X_train,y_train,X_test,y_test,Cv_folds,n_estimators,criterion,bootstrap,max_depth,min_samples_splits,min_samples_leafs,min_impurity_splits):
	print('------- Extra Trees Classifier-------')
	train_results_mean = []
	train_results_std = []
	train_results_auc=[]
	test_results_auc=[]
	test_accu=[]
	test_precision=[]
	test_recall=[]
	train_recall=[]
	features=[]
	for estimator in n_estimators:
		for cr in criterion:
			for boots in bootstrap:
				for max_d in max_depth:
					for min_sample_sp in min_samples_splits:
						for min_samples_le in min_samples_leafs:
							for min_impurity_sp in min_impurity_splits:

								rfc = ExtraTreesClassifier(n_estimators=estimator,criterion=cr,bootstrap=boots,max_depth=max_d, class_weight={0:y_train.mean(), 1:1-y_train.mean()},min_samples_split=min_sample_sp,min_samples_leaf=min_samples_le,min_impurity_decrease=min_impurity_sp,random_state=0,n_jobs=-1)
								all_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=Cv_folds,scoring='roc_auc')
								train_results_mean.append(all_accuracies.mean())
								train_results_std.append(all_accuracies.std())
    
								rfc.fit(X_train, y_train)
    
								train_pred = rfc.predict(X_train)
    
								precision,recall,F1,accuracy,confusion,roc_auc=scores(y_train,train_pred)
								train_recall.append(recall)
								train_results_auc.append(roc_auc)
    
								y_pred = rfc.predict(X_test)
    
								precision,recall,F1,accuracy,confusion,roc_auc=scores(y_test,y_pred)

								test_recall.append(recall)
								test_results_auc.append(roc_auc)
								test_accu.append(accuracy)
								test_precision.append(precision)
								features.append([estimator,cr,boots,max_d,min_sample_sp,min_samples_le,min_impurity_sp])
	df1= pd.DataFrame({'train_results_mean':train_results_mean})
	df2= pd.DataFrame({'train_results_std':train_results_std})
	df3= pd.DataFrame({'train_results_auc':train_results_auc})
	df4= pd.DataFrame({'test_results_auc':test_results_auc})
	df5= pd.DataFrame({'test_accuracy':test_accu})
	df6= pd.DataFrame( {'test_precision':test_precision})
	df8= pd.DataFrame( {'train_recall':train_recall})
	df9= pd.DataFrame( {'test_recall':test_recall})
	df7= pd.DataFrame( {'features':features})

	df_results=pd.concat([df1, df2,df3,df4,df5,df6,df8,df9,df7],axis=1)
	return df_results



def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(y_scores,y_test,p, r, thresholds):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='r', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')

        
    


    

    
    




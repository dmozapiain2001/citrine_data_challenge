from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import pandas as pd

def scores(y_test,y_pred):

	precision = metrics.precision_score(y_test, y_pred, average='binary')
	recall = metrics.recall_score(y_test, y_pred, average='binary')
	F1 = metrics.f1_score(y_test, y_pred, average='binary')
	accuracy=metrics.accuracy_score(y_test, y_pred)
	confusion = metrics.confusion_matrix(y_test, y_pred)


	return precision,recall,F1,accuracy,confusion


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



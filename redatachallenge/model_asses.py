from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics


def scores(y_test,y_pred):
	precision = metrics.precision_score(y_test, y_pred, average='binary')
	recall = metrics.recall_score(y_test, y_pred, average='binary')
	F1 = metrics.f1_score(y_test, y_pred, average='binary')
	accuracy=metrics.accuracy_score(y_test, y_pred)
	confusion = metrics.confusion_matrix(y_test, y_pred)

	return precision, recall, F1, accuracy, confusion


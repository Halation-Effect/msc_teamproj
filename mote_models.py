import threading, Queue, logging, timeit
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import statsmodels.api as sm
from patsy import dmatrices
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(level = logging.DEBUG, format = "(%threadName)-1s %(message)s")
#pd.set_option('display.max_rows', None, 'display.max_columns', None)

def logit_model(ids):
	columns = ['date', 'time', 'temp', 'humid', 'lux']
	mote_id = "mote_" + str(ids) + ".csv"
	
	dataset = pd.read_csv(mote_id)
	dataset.columns = columns

	"""
	plots
	---------------
	"""

	dataset.drop(['date'], axis = 1, inplace = True)
	dataset.drop(['time'], axis = 1, inplace = True)
	dataset['hot'] = ((dataset.temp > 20) & (dataset.humid < 30 ) & (dataset.lux > 400)).astype(int)

	#print "average group\n----------------------\n"
	#print dataset.groupby('hot').mean()
	
	y, x = dmatrices('hot ~ temp + humid + lux', dataset, return_type = 'dataframe')
	y = np.ravel(y)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, test_size = 0.4, random_state = 0)

	model = LogisticRegression()
	model.fit(x_train, y_train)

	predicted = model.predict(x_test)
	prob = model.predict_proba(x_test)

	#confmatrix = metrics.confusion_matrix(y_test, predicted)
	#print "\nconfusion matrix\n-----------------------\n", confmatrix
	#print "\naccuracy score:", metrics.accuracy_score(y_test, predicted)

	modelparams = model.get_params(deep = True)
	modelcoefs = model.coef_

	print modelcoefs

if __name__ == "__main__":	
	datasets = [2, 9, 38, 47, 50]
	for ids in datasets:
		t = threading.Thread(name = ("mote_", str(ids)), target = logit_model, args = (ids, ))
		t.start()

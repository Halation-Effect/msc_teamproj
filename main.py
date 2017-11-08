import threading, Queue, logging, timeit
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from patsy import dmatrices
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(level = logging.DEBUG, format = "(%(threadName)-1s) %(message)s")
#pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.options.mode.chained_assignment = None

class logitmodel(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
	
	def run(self):
		datasets = [2, 9, 16, 27, 38, 47, 50]
		for i in datasets:
			t = threading.Thread(name = ("mote_" + str(i)), target = self.model, args = (i, ))
			t.start()
			#t.join()

	def model(self, ids):
		s1 = 0
		s2 = 10000

		local_mote_coefs = {}
		mote_ids = []
		mote_ids.append(ids)


		"""
		plots
		---------------
		plt.subplot(111)
		plt.subplots_adjust(bottom = 0.2, top = 0.9)
		plt.plot_date(x = dataset.date, y = dataset.temp, fmt = 'r-')
		plt.xticks(rotation = 90)

		plt.scatter(dataset.temp, dataset.humid, s = 5, c = ['red', 'green'])
		plt.xlabel("temp")
		plt.ylabel("humid")

		plt.subplot(2, 1, 1)
		dataset.temp.hist()
		plt.xlabel("temperature")
		plt.ylabel("frequency")

		plt.subplot(2, 1, 2)
		dataset.light.hist()
		plt.xlabel("lux")
		plt.ylabel("frequency")
		"""

		#plt.show()
	
		for _ in xrange(2):
			columns = ['date', 'time', 'temp', 'humid', 'lux']
			mote_id = "mote_" + str(ids) + ".csv"
		
			df = pd.read_csv(mote_id, header = 1)
			dataset = df.iloc[s1:s2, :]
			dataset.columns = columns

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

			local_mote_coefs.update({mote_id : modelcoefs})	

			logging.debug("%d - %d\t\t%s", s1, s2, modelcoefs)
			s1 = s2
			s2 += 1200

if __name__ == "__main__":
	print "mote coefficient vectors [b0, b1, b2, b3]\n"	
	m = logitmodel()
	m.run()


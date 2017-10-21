import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from patsy import dmatrices
from pylab import rcParams

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.max_rows', None, 'display.max_columns', None)

#dataset = pd.read_csv("a.csv", header = None)
dataset = pd.read_csv("mote_9.csv")
dataset.columns = ['temp', 'humid', 'light']

plt.scatter(dataset.temp, dataset.humid, s = 5, c = ['red', 'green'])
plt.xlabel("temp")
plt.ylabel("humid")

"""
plt.subplot(2, 1, 1)
dataset.temp.hist()
plt.xlabel("temperature")
plt.ylabel("frequency")

plt.subplot(2, 1, 2)
dataset.light.hist()
plt.xlabel("lux")
plt.ylabel("frequency")
"""

dataset['hot'] = ((dataset.temp > 20) & (dataset.humid < 38) & (dataset.light > 400)).astype(int)
#print dataset

print "average group\n------------------\n"
print dataset.groupby('hot').mean()

y, x = dmatrices('hot ~ temp + humid + light', dataset, return_type = 'dataframe')
y = np.ravel(y) # 1D array

#x = dataset.ix[:, (0, 1, 2)].values
#y = dataset.ix[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.3, test_size = 0.5, random_state = 0)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

predicted = log_reg.predict(x_test)
#print predicted
prob = log_reg.predict_proba(x_test)
#print prob

confmatrix = metrics.confusion_matrix(y_test, predicted)
print "\n", confmatrix

print "accuracy score: ", metrics.accuracy_score(y_test, predicted)

#plt.show()


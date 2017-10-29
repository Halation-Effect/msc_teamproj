import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from patsy import dmatrices
from pylab import rcParams

from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier

from mlxtend.classifier import StackingClassifier

pd.set_option('display.max_rows', None, 'display.max_columns', None)

estimator = []
columns = ['date', 'time', 'temp', 'humid', 'light']

#dataset = pd.read_csv("a.csv", header = None)
dataset_1 = pd.read_csv("mote_38.csv")
dataset_1.columns = columns

dataset_2 = pd.read_csv("mote_9.csv")
dataset_2.columns = columns

"""
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

dataset_1.drop(['date'], axis = 1, inplace = True)
dataset_1.drop(['time'], axis = 1, inplace = True)
dataset_1['hot'] = ((dataset_1.temp > 20) & (dataset_1.humid < 30) & (dataset_1.light > 400)).astype(int)

dataset_2.drop(['date'], axis = 1, inplace = True)
dataset_2.drop(['time'], axis = 1, inplace = True)
dataset_2['hot'] = ((dataset_2.temp > 20) & (dataset_2.humid < 30) & (dataset_2.light > 400)).astype(int)

#print dataset

print "average group\n------------------\n"
print dataset_1.groupby('hot').mean()
print "\n--------------------\n"
print dataset_2.groupby('hot').mean()

y_1, x_1 = dmatrices('hot ~ temp + humid + light', dataset_1, return_type = 'dataframe')
y_1 = np.ravel(y_1) # 1D array

y_2, x_2 = dmatrices('hot ~ temp + humid + light', dataset_2, return_type = 'dataframe')
y_2 = np.ravel(y_2)

#x = dataset.ix[:, (0, 1, 2)].values
#y = dataset.ix[:, 3].values

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, train_size = 0.6, test_size = 0.4, random_state = 0)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, train_size = 0.6, test_size = 0.4, random_state = 0)

logreg1 = LogisticRegression()
logreg1.fit(x_train_1, y_train_1)
logreg2 = LogisticRegression()
logreg2.fit(x_train_2, y_train_2)

estimator.append(logreg1)
estimator.append(logreg2)

predicted_1 = logreg1.predict(x_test_1)
prob_1 = logreg1.predict_proba(x_test_1)
predicted_2 = logreg2.predict(x_test_2)
prob_2 = logreg2.predict_proba(x_test_2)

confmatrix_1 = metrics.confusion_matrix(y_test_1, predicted_1)
print "\nconfusion matrix\n-----------------\n", confmatrix_1

confmatrix_2 = metrics.confusion_matrix(y_test_2, predicted_2)
print "\nconfusion matrix\n-----------------\n", confmatrix_2

print "\naccuracy score:", metrics.accuracy_score(y_test_1, predicted_1)
print logreg1.get_params(deep = True)

print "\naccuracy score:", metrics.accuracy_score(y_test_2, predicted_2)
print logreg2.get_params(deep = True)

#ensemble = VotingClassifier(estimator)
#res = model_selection.cross_val_score(ensemble)
#print res.mean()

lr = LogisticRegression()
sclf = StackingClassifier(classifiers = [logreg1, logreg2], meta_classifier = lr)

#res = model_selection.cross_val_score(sclf, scoring = 'accuracy')
#print res


import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#fetching thd data file
f = open("c:/users/kshitij/desktop/winequality-white1.csv")
f.readline() # skip the header
data = np.loadtxt(f,delimiter=',')

#setting the test and training parameters
train, test = train_test_split(data, train_size=0.6, test_size=0.4)
X  = train[:,0:10]
#normalising the data set
scaler = StandardScaler().fit(X)
K = scaler.transform(X)
L = train[:, 11]
m = test[:, 0:10]
m = scaler.transform(m)
n = test[:, 11]

# Build the model
clf = linear_model.SGDRegressor(loss='squared_loss', penalty='l1', alpha=.0001,
                     fit_intercept=True, n_iter=100, shuffle=True)
# Evaluating the model on training data
clf.fit(K,L)
print clf.score(K,L)
print clf.score(m,n)
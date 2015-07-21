import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#open the csv file 
f = open("c:/users/kshitij/desktop/winequality-white1.csv")
f.readline() # skip the header
data = np.loadtxt(f,delimiter=',')

# splitting the file into test and trainig data and normalising the data
train, test = train_test_split(data, train_size=0.6, test_size=0.4)
X  = train[:,0:10]
scaler = StandardScaler().fit(X)
K = scaler.transform(X)
L = train[:, 11]
m = test[:, 0:10]
m = scaler.transform(m)
n = test[:, 11]
# Build the model parameters
clf = linear_model.Ridge(alpha=.00001, fit_intercept=True, normalize=False,
                         copy_X=True, max_iter=100)
# Evaluating the model on training data
clf.fit(K,L)

print clf.score(K,L)
print clf.score(m,n)
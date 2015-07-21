import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

f = open("c:/users/kshitij/desktop/winequality-white1.csv")
f.readline() # skip the header
data = np.loadtxt(f,delimiter=',')

#splitting the data into training and test
train, test = train_test_split(data, train_size=0.6, test_size=0.4)
K = train[:, 0:10]
L = train[:, 11]
m = test[:, 0:10]
n = test[:, 11]

# Build the model
clf = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)      
# Evaluating the model on test data  
clf.fit(K,L)
#printing the results
print clf.score(K,L)
print clf.score(m,n)
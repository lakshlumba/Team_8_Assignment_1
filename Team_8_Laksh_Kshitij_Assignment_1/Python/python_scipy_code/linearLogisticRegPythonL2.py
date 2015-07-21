import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
#from sklearn import metrics
# reading the file
f = open("c:/users/kshitij/desktop/winequality-white1.csv")
f.readline() # skip the header
data = np.loadtxt(f,delimiter=',')

# splitting up the data set into training and test
train, test = train_test_split(data, train_size=0.6, test_size=0.4)
K = train[:, 0:10]
L = train[:, 11]
m = test[:, 0:10]
n = test[:, 11]

# setting up the model arguments
clf = linear_model.LogisticRegression(penalty='l2', fit_intercept=True,
                                       random_state=None, max_iter=100) 
#predctn = clf.predict(m)
clf.fit(K,L)
#printing the values
print clf.score(K,L)
print clf.score(m,n)

#print "confusion metrics below"
#print metrics.confusion_matrix(n, predctn)

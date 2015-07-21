import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

#Open the csv file
f = open("c:/users/kshitij/desktop/winequality-white1.csv")
f.readline() # skip the header
data = np.loadtxt(f,delimiter=',')

# Build the model
train, test = train_test_split(data, train_size=0.6, test_size=0.4)
K = train[:, 0:10]
L = train[:, 11]
m = test[:, 0:10]
n = test[:, 11]

# Evaluating the model on training data  - using regularization level as L2 , loss function used log
clf = linear_model.SGDClassifier(alpha=.10, average=False, epsilon=0.1,
         loss='log', n_iter=5,penalty='l2', power_t=0.5, shuffle=True,
         verbose=0)
clf.fit(K,L)

#print the scores
print clf.score(K,L)
print clf.score(m,n)
import math

import numpy as np
from numpy import asmatrix as mat
from numpy import asarray as arr

import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils import *

# Load preprocessed data
data = get_IMBD_preprocessing(load_cached=True)
X_train, X_val, X_test, y_train, y_val, y_test = \
			data["X_train"], data["X_val"], data["X_test"], data["y_train"], data["y_val"], data["y_test"]

# Basic online kernel learning implementation (no tuning currently)
M, N = X_train.shape
gamma = .0005
n_components = 3750
rbf_feature = np.random.randn(N,n_components)*np.sqrt(gamma)

X_features = X_train.dot(rbf_feature)
X_features = (1/np.sqrt(n_components))*np.concatenate((np.cos(X_features),np.sin(X_features)),axis=1)

clf = LogisticRegression(C=50)
clf.fit(X_features, y_train)

X_test = X_test.dot(rbf_feature)
X_test = (1/np.sqrt(n_components))*np.concatenate((np.cos(X_test),np.sin(X_test)),axis=1)

print ("Final Accuracy: %s" % accuracy_score(y_test, clf.predict(X_test)))

from pyriemnan.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from mne.decoding import CSP
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression

from data import load_data

import numpy as np
import joblib

filename = 'E:/Projects (Uni)/BCI/IV-2a/svm_bci_22ch.pkl' # Path for pre-trained model.

# Loading the data 
X, labels = load_data()
X = X.transpose()

# cross validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)

###############################################################################
# Classification with Rbf  SVM

# building pipeline
covest = Covariances()
ts = TangentSpace()
svc = SVC(C = 20,kernel='rbf', degree=10, gamma='auto', coef0=0.0, tol=0.001, cache_size=10000, max_iter=-1, decision_function_shape='ovr')
# svc = LinearSVC(C = 0.05, intercept_scaling=1, loss='hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=1, tol=0.00001)

clf = make_pipeline(covest, ts, svc)
# cross validation
accuracy = cross_val_score(clf, X, labels)

print("SVM Mean acuuracy: ", accuracy.mean())

###############################################################################
# Classification with MDM algorithm
cov = covest.fit_transform(X)
mdm = MDM()
accuracy = cross_val_score(mdm, cov, labels)

print("MDM Mean acuuracy: ", accuracy.mean())

##############################################################################
# Classification with CSP + SVC
# Assembling pre-trained SVM

svc2 = joblib.load(filename)
# clf2 = svc2.fit(X, labels)
clf2 = make_pipeline(covest,ts,svc2)
# cross validation
accuracy2 = cross_val_score(clf2, X, labels)

print("Pretrained SVM Mean acuuracy: ", accuracy2.mean())
###############################################################################
# # Classification with CSP + SVC

# # Assemble a classifier
# csp = CSP(n_components=4, reg='ledoit_wolf', log=True)

# clf = Pipeline([('CSP', csp), ('SVC', svc)])
# scores = cross_val_score(clf, X, labels, cv=cv, n_jobs=1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("CSP + SVC Classification accuracy: %f / Chance level: %f" %(np.mean(scores), class_balance))
# This performs really bad. 
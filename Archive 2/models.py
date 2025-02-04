# The functions for the models used. Was also used for the implementations of 
# many other models that were not used in the end.
# 1 Function:
#   final_classifier(X_train, X_test, y_train) - Implements the Quadratic 
#   Discriminant Analysis model using sklearn on the provided training 
#   dataset. Returns a list of the predicted labels
#   for the evaluation dataset.
#
# @author Simon Hocker
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB

# Final model used was initially Quadratic Discriminant Analysis. 
# But after some additional testing with different k-cross validation I decided 
# to go with the stacking classifier instead.

def final_classifier(X_train, X_test, y_train):    
    
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'))]    
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=10)
    model.fit(X_train, y_train)
    label_list = model.predict(X_test)
    return label_list



# The main file for the classifiers. Also held the function used for testing models.
# 4 Functions:
#   foldData(X, y) - Splits data into pieces, test sets and training sets.

#   train(X, y, data_folds, model, model_name) - Runs a series of tests using 
#   a provided model, training set and test set. Was used heavily during the 
#   model experimentation.

#   main() - Converts the csv files into dataframes using pandas. Then calls 
#   on the data cleaning functions to clean the data. Runs the final classifier on 
#   them and outputs the resulting label predictions to a txt file.

#   final_classifier(X_train, X_test, y_train) - Implements the Quadratic 
#   Discriminant Analysis model using sklearn on the provided training 
#   dataset. Returns a list of the predicted labels
#   for the evaluation dataset.
#
# @author Simon Hocker
import numpy as np
import pandas as pd
import data_cleaner
from sklearn.model_selection import KFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier

# Quadratic Discriminant Analysis
def QDA(X_train, X_test, y_train):    
    
    my_model = QuadraticDiscriminantAnalysis()
    my_model.fit(X_train, y_train)
    label_list = my_model.predict(X_test)
    return label_list


# K-fold cross validation - n_splits representing the k folds.
def foldData(X, y):

    foldedData = KFold(n_splits=5, shuffle=True)    
    return foldedData.split(X, y)


# Using the data_folds, tests are run on each of the folds using the provided model to 
# determine its average accuracy. Then prints the accuracy. 
# Ignoring converge warnings because multi-layer perceptron produced many of them but still seems to work
@ignore_warnings(category=ConvergenceWarning)
def train(X, y, data_folds, model_name):
    
    # acc_averages = 0
    
    accuracy_list = []

    for train_subset, test_subset in data_folds:

        # Save the info
        X_train, X_test = X[train_subset], X[test_subset]
        y_train, y_test = y[train_subset], y[test_subset]

        # Run the model and save its resulting labels
        label_list = QDA(X_train, X_test, y_train)

        # Check accuracy of the model results
        correctly_labeled = np.array(np.where(y_test == label_list))
        accuracy = correctly_labeled[0].shape[0] / y_test.shape[0]
        accuracy_list.append(accuracy)

    # Print the accuracy for the model
    accuracy_list = np.array(accuracy_list)
    print(f'{model_name}: {np.mean(accuracy_list)}')
    

def main():

    # Get Training Data File 
    trainFile = pd.read_csv('TrainOnMe.csv', usecols=range(1,15))
    print("Found File")

    # Clean Training Data
    Clean_train_data, y_labels = data_cleaner.clean_train_Data(trainFile)
    print("training data cleaned!")

    # Get Evaluation Data File
    evalFile = pd.read_csv('EvaluateOnMe.csv', usecols=range(1,14))
    print("Eval file found!")

    # Clean evaluation data
    Clean_eval_data = data_cleaner.clean_Eval_Data(evalFile)  
    print("Eval data cleaned!") 

    # Classifier
    final_labels = QDA(Clean_train_data, Clean_eval_data, y_labels)
    print("Classifier created!")

    # Create txt file and write the predicted labels there
    output_file = open("QDAPredictions.txt", "w")        
    for label in final_labels:
        output_file.write(f'{label}\n')
    print("Finished Writing predictions!")
    

if __name__ == "__main__":
    main()
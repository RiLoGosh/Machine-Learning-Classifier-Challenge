# The file responsible for processing the data.
# 2 Functions:
#   cleanData(data) - Cleans the overall training data by removing 
#   data that is deemed to be irrelevant or yields a low information 
#   gain based on previous testing.

#   clean_Eval_Data(data) - Cleans the evaluation data according to 
#   the same paradigm as the training data. 
#
# @author Simon Hocker
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# y - 3 unique labels: Boborg (422), Jorgsuto (258), atsutobob (317)
# x1 - x6, x8 - x11, x13 check ranges, normalize(?) and check for 'nan':s and outliers
# x7 - unique labels: slängpolskorgris (359), hambogris (165), schottisgris(134), polkagris(318)
# x12 is all TRUE (except for a single misspelling of "ture"). 
def clean_train_Data(data):

    # Remove misspelled y_label rows. Could perhaps just fix the spelling of 
    # them to produce more training data if they are still valid
    data = data[data.y.notna()]    
    unique_labels_y = pd.unique(data.y)
    data = data[data.y != unique_labels_y[3]]
    data = data[data.y != unique_labels_y[4]]
    data = data[data.y != unique_labels_y[5]]
    data = data[data.y != unique_labels_y[6]]


    # Drop all rows with hashtag noise entries
    # EDIT: Apparently this was pointless. Only the canvas page had hashtag entries. Actual file was fine
    data.x1 = data.x1[data.x1 != '########']
    data.x2 = data.x2[data.x2 != '########']
    data.x3 = data.x3[data.x3 != '########']
    data.x4 = data.x4[data.x4 != '########']
    data.x5 = data.x5[data.x5 != '########']
    data.x6 = data.x6[data.x6 != '########']
    data.x7 = data.x7[data.x7 != '########']
    data.x8 = data.x8[data.x8 != '########']
    data.x9 = data.x9[data.x9 != '########']
    data.x10 = data.x10[data.x10 != '########']
    data.x11 = data.x11[data.x11 != '########']
    data.x12 = data.x12[data.x12 != '########']
    data.x13 = data.x13[data.x13 != '########']

    data = data[data.x4.notna()]    # Remove empty cell's row in x4

    # x1 - drop rows with outliers
    data.x1 = data.x1.astype(float)
    data = data[data.x1 < 103.0]
    data = data[data.x1 > 97.0]     

    # x2 - drop rows with outliers
    data.x2 = data.x2.astype(float)
    data = data[data.x2 < 2.5]    # Could be 3.0
    data = data[data.x2 > -2.5]     # Could be -3.0 too    

    # x3 - drop rows with outliers (No need, no outliers)

    # x4 - drop rows with outliers (No outliers)
    # But found interesting detail: every x4 > -0.61539 results in Atsutobob (about 80 results)
    
    # x5 - drop rows with outliers
    # Seems to correspond to x6 result. 
    # Certain number ranges result in different types of pigs.
    data.x5 = data.x5.astype(float)
    data = data[data.x5 < 203.0]    
    data = data[data.x5 > 197.0] 

    # x6 - drop rows with outliers
    data.x6 = data.x6.astype(float)
    data = data[data.x6 < -85.0]    
    data = data[data.x6 > -96.0]            

    # x8 - drop rows with outliers
    data.x8 = data.x8.astype(float)
    data = data[data.x8 < 4.0]    
    data = data[data.x8 > -4.0]         

    # x9 - drop rows with outliers
    data.x9 = data.x9.astype(float)
    data = data[data.x9 < 4.0]    
    data = data[data.x9 > -4.0]   

    # x10 - drop rows with outliers
    data.x10 = data.x10.astype(float)
    data = data[data.x10 < 15.0]   # Maybe 14.5  
    data = data[data.x10 > 7.0]    # Maybe 6.5  

    # x11 - drop rows with outliers
    data.x11 = data.x11.astype(float)
    data = data[data.x11 < 6.0]     # Maybe 5.5 
    data = data[data.x11 > -6.0]    

    # x13 - drop rows with outliers
    data.x13 = data.x13.astype(float)
    data = data[data.x13 < 464.0]    
    data = data[data.x13 > 437.0]      

    # drop columns x2, x3, x4, x5, x7 and x12
    data.drop('x2', inplace=True, axis=1)   # Desn't seem to add or remove much, low info gain
    data.drop('x3', inplace=True, axis=1)   # Collinearity warning, should be omitted to avoid collinearity
    #data.drop('x4', inplace=True, axis=1)  # Input contains empty entry, fixed. IMPORTANT INCLUDE 10+%
    #data.drop('x5', inplace=True, axis=1)  # IMPORTANT INCLUDE 10+%
    data.drop('x7', inplace=True, axis=1)   # List of types of pigs, might be relevant but assuming (hoping) it isn't
    #data.drop('x8', inplace=True, axis=1)   # Include. 1% reduction
    #data.drop('x9', inplace=True, axis=1)   # Include. 1% reduction
    #data.drop('x10', inplace=True, axis=1)  # Include. 1% reduction
    #data.drop('x11', inplace=True, axis=1)  # IMPORTANT INCLUDE 4%
    data.drop('x12', inplace=True, axis=1)  # List of the word "TRUE", pointless to include
    data.drop('x13', inplace=True, axis=1)  # Doesn't add or remove much, low info gain
    
    rows, columns = data.shape

    # normalise the feature vectors
    normaliser_tool = StandardScaler()
    Cleaned_data = normaliser_tool.fit_transform(data.iloc[:, 1:columns])

    labels = data['y'].to_numpy()    
    
    return Cleaned_data, labels


def clean_Eval_Data(data):

    # Drops the least relevant features as determined during earlier testing
    data.drop('x2', inplace=True, axis=1)
    data.drop('x3', inplace=True, axis=1)
    data.drop('x7', inplace=True, axis=1)
    data.drop('x12', inplace=True, axis=1)
    data.drop('x13', inplace=True, axis=1)

    # Normalise the features' vectors
    normaliser_tool = StandardScaler()    
    X = normaliser_tool.fit_transform(data)
    return X


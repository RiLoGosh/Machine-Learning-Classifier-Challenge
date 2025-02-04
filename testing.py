import numpy as np
import pandas as pd

def average(col):
    colcopy = col
    colcopy = colcopy[colcopy != '########']
    sum = 0
    for item in colcopy:
        sum += item
        print(item)
   
    return sum/len(colcopy)

# Used at one point to compare classifier similarities
def checkDifferences(file1, file2):

    count = 0
    for i in (range(len(file1))):
        if file1[i] == file2[i]:
            count = count+1

    return count
    


# Used for multiple testing of code. Deprecated now.
def findOutliers(data):
    data = data[data.y.notna()]    
    unique_labels_y = pd.unique(data.y)
    data = data[data.y != unique_labels_y[3]]
    data = data[data.y != unique_labels_y[4]]
    data = data[data.y != unique_labels_y[5]]
    data = data[data.y != unique_labels_y[6]]

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


    # # Replacing all '########' instances with 999999999.0 
    # # (arbitrary number as long as it isn't present in dataset')
    # # Must do this before casting things to float 
    # data.replace('########', 999999999.0)
    
    # # x1 - make all numbers into floats, drop the outliers
    # # and replace all instances of 999999999.0 with the average of that column
    
    data.x1 = data.x1.astype(float)
    # data = data[data.x1 < 103.0]
    # data = data[data.x1 > 97.0] 
    # average = testing.average(data.x1)
    # data.x1.replace(999999999.0, average)
    # # Replace all instances of '########' with the average of that column

    # x2 - drop rows with outliers
    data = data[data.x1 < 103.0]
    data = data[data.x1 > 97.0] 
    

    #data.x1 = float(data.x1)
    data.x1 = data.x1.astype(float)
    data = data[data.x1 < 103.0]
    data = data[data.x1 > 97.0] 

    sum = 0
    for item in data.x1:
        sum += float(item)
        print(item)
   
    #average = data["x1"].mean()
    return sum/len(data.x1)


__author__ = 'michaelbinger'

# This contains the homemade data filtering algorithms df and dfrange, as well as showstats
import numpy as np
from numpy import *

# First, we might want to ask for only the data which has a certain value for a feature (like class for example)
# def datafilter(value,index): return data[data[0::,index] == value]
# this outputs all of the data which has the value 'value' for index 'index'

# Now we implement this for any list of features...
# The function below supercedes datafilter, which does not need to be used, but is included above to
# make it easier to understand the logic of df
def df(features,dataset):
    """
    Filters the data by certain features.
    Features is a list of features we want to filter by.
    For ex. [[1,5],[2,6]] # gives all data with sibsp = 1 and parch = 2.
    """
    if size(dataset) == 0:
        return []
    features=list(features) # new copy of the features list
    dataset=dataset.copy() #start with a copy of the regularized data, either data, test8, or testdata
    n = np.size(features,0) #number of features we are filtering by
    for x in xrange(n):
        feat = features[x] #pair [value,index] for each feature
        value = feat[0]
        index = feat[1]
        dataset = dataset[dataset[0::,index] == value]
        if size(dataset) == 0:
            return []
    return dataset
# for data and test8 note indices [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# for testdata note indices [0=class, 1=sex, 2=age, 3=sibsp, 4=parch, 5=fare, 6=embarked]
# it is probably better to only use test8, and not testdata, so as to avoid confusion on the indices:
# using only data and test8 you will have uniform index categories.

# dfrange will enable us to filter by age range or fare range, as well as ranges in the discrete features
def dfrange(fmin,fmax,index,dataset):
    """
    Takes in a dataset, along with min and max values for a feature at an index, and returns the subset of dataset
    that has those feature value. Can be applied recursively (i.e. dfrange(1,3,1,dfrange(0,18,3,data)))
    """
    if size(dataset) == 0:
        return []
    dataset=dataset.copy() #start with a copy of the regularized data, either data, test8, or testdata
    truthtable = (dataset[0::,index] <= fmax) & (dataset[0::,index]>=fmin)
    dataset = dataset[truthtable]
    return dataset

# Usage examples
#print dfrange(0,1,3,data) # 0 and 1 year olds
#print df([[1,2],[3,1]],dfrange(0,10,3,data)) #3rd class females with age<=10
#print df([[4,4],[1,2]],data) # female passengers with sibsp=4


def showstats(datasubset): # use this on any subset of data. don't use this on test data b/c we don't know sur values
    """
    This takes in any subset of data and returns a list [Number survived, total number, percentage]
    """
    if size(datasubset) == 0:
        return "none"
    datasubset=datasubset.copy() #start with a copy of the subset
    nsur = int(np.sum(datasubset[0::,0]))
    ntot = np.size(datasubset[0::,0])
    if ntot == 0:
        per=0
    else:
        per=round(float(nsur)/ntot,3)
    return [nsur, ntot, per] # return the number survived, the total in the datasubset, and the percent survived


def databin(dataset, agebin = "on"):
    """
    bins our data:
    ages: 0-10 become 5, 11-19 become 15, 20+ become 50
    sibsp and parch: 0 = 0, 1 and 2 = 1, and 3+ = 3 so have value 0,1, and 3 only
    thus we have the following number of discrete values:
    [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
    [2,3,2,3,3,3,fare,3]
    fare will not be binned
    """
    dataset = dataset.copy()
    for row in dataset:
        if agebin == "on":
            if (row[3] >=0) and (row[3]<=10):
                row[3] = 5
            elif (row[3] >=11) and (row[3]<=19):
                row[3] = 15
            else:
                row[3] = 50
        if (row[4] == 0):
            row[4] = 0
        elif (row[4] == 1) or (row[4] == 2):
            row[4] = 1
        else:
            row[4] = 3
        if (row[5] == 0):
            row[5] = 0
        elif (row[5] == 1) or (row[5] == 2):
            row[5] = 1
        else:
            row[5] = 3
    return dataset

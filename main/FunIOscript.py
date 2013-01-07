
__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

from time import time, clock
starttime = time()
#import csv as csv
import numpy as np
#import scipy
from numpy import *
#from sklearn.ensemble import RandomForestClassifier
#from predictions import genderpred, f3sm12pred, newpred, predicttrain, comparepreds, randomforests

set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)

# File PrepareTitanicData.py goes through the details of preparing the data.
# It converts sex to 1,0 for F,M, and 0,1,2,3 for city embarked "S","C","Q",""
# Also for no-age we put in placeholder 1000
# for no fare we put placeholder 1000
# Cabin, Ticket #, and name are deleted

from PrepareTitanicData import titandata

data=titandata("train") #(891,8) array
testdata=titandata("test") #(418,7) array
test8 = titandata("test8") #(418,8) array

# Call function titandata which takes an argument string, which must be either "train", "test", or "test8"
#print data[0:10]
#print testdata[0:10]
#print test8[0:10]
# Note that data and test8 are regularized and floated into an array of 8 columns:
# [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# The survival values are 0=die, 1=live, 2=don't know (for test8)
# Note that testdata is regularized and floated into an array of 7 columns
# [class, sex, age, sibsp, parch, fare, embarked]

totdata = vstack((data,test8)) # stacks data on top of test8 to create a (1309,8) array

def df(features,dataset):
# features is a list of features we want to filter by. For ex. [[1,5],[2,6]]
# gives all data with sibsp = 1 and parch = 2.
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
    if size(dataset) == 0:
        return []
    datatemp=dataset #start with the regularized data, either data, test8, or testdata
    truthtable = (datatemp[0::,index] <= fmax) & (datatemp[0::,index]>=fmin)
    datatemp = datatemp[truthtable]
    return datatemp

# Usage examples
#print dfrange(0,1,3,data) # 0 and 1 year olds
#print df([[1,2],[3,1]],dfrange(0,10,3,data)) #3rd class females with age<=10
#print df([[4,4],[1,2]],data) # female passengers with sibsp=4

#We'll also need to replace placeholder values for age = 1000
def convertages(dataset,ageindex): #ageindex=3 for test8 and data, and 2 for testdata
    for row in dataset:
        if row[ageindex] == 1000:
            if row[ageindex+1]<=2: # replace placeholder age 1000 with age 30 if sibsp<=2
                row[ageindex] = 30
            else:
                row[ageindex] = 10 # replace placeholder age 1000 with age 10 if sibsp>=3
    return dataset

# Now let's actually convert the unknown ages to our best guesses.
# These derive from analysis below which were performed BEFORE doing this conversion
# (i.e. only on the data where the age is given). To repeat and confirm this analysis you can simply
# comment out the data conversion below

data = convertages(data,3)
test8 = convertages(test8,3)
testdata = convertages(testdata,2)
totdata = convertages(totdata,3)


def showstats(datasubset): # use this on any subset of data. don't use this on test data b/c we don't know sur values
    if size(datasubset) == 0:
        return "none"
    nsur = int(np.sum(datasubset[0::,0]))
    ntot = np.size(datasubset[0::,0])
    if ntot == 0:
        per=0
    else:
        per=round(float(nsur)/ntot,3)
    return [nsur, ntot, per] # return the number survived, the total in the datasubset, and the percent survived



indict8 = { 0 : 'sur', 1 : 'class', 2 : 'sex', 3 : 'age', 4 : 'sibsp', 5 : 'parch', 6 : 'fare' , 7 : 'city' }

constraints = []
print "Recall indices [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]"
while True:
    query = raw_input("Input a feature constraint (in form 'min max index')(type 'x' to quit inputting constraints):")
    if query == "x":
        break
    query = [float(x) for x in query.split()] # split breaks the 3 string inputs up, and they are then floated
    fmin = query[0]
    fmax = query[1]
    index = query[2]
    print "Great! We'll apply the constraint: %i <= %s <= %i" %(fmin, indict8[index], fmax)
    constraints.append(query)
print "To summarize, you said constrain data by:", constraints

ncon = np.size(constraints)/3
tempdata = data #using the training data set.
#tempdata = test8 # test8 can be used to look at passenger attributes. But note unknown survival value = 2.
for x in xrange(ncon):
    fmin = constraints[x][0]
    fmax = constraints[x][1]
    index = constraints[x][2]
    tempdata = dfrange(fmin,fmax,index,tempdata)

print tempdata
print showstats(tempdata)


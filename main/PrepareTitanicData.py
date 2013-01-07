__author__ = 'michaelbinger'

# This file and the function titandata at the bottom will streamline
# importing, regularizing, and floating the data sets. Only works for test.csv and train.csv.

import csv as csv
import numpy as np
import scipy
from numpy import *

#create the data array from the original file train.csv, as explained in kaggle tutorial
traincsv = csv.reader(open("../train.csv", 'rb'))
traincsv.next()
fulldata=[]
for row in traincsv:
    fulldata.append(row)
data = np.array(fulldata)
#NOTE: data[j] for j=0..890 is of the form of 11 strings:
# ['survived?' 'class' 'name' 'sex' 'age' 'sibsp' 'parch' 'ticket #' 'fare' 'cabin' 'embarked']

#convert sex F,M to 1,0
data[data[0::,3] == "female", 3] = 1
data[data[0::,3] == "male", 3] = 0

#convert embarked strings into numbers for easier coding
data[data[0::,10] == "S", 10] = 0
data[data[0::,10] == "C", 10] = 1
data[data[0::,10] == "Q", 10] = 2
data[data[0::,10] == "", 10] = 3

# for passengers with no age, put age 1000 placeholder
data[data[0::,4] == "", 4] = 1000

# passengers with no fare: there is only one man in the testdata from 3rd class S.
# data[data[0::,8] == "", 8] = 13


# Now we will delete the unwanted columns of data. I print it out
# for a passenger so you can see it's doing the right thing
data = scipy.delete(data,9,1) # delete cabin
data = scipy.delete(data,7,1) # delete ticket#
data = scipy.delete(data,2,1) # delete name

# convert the strings to floats
nrow = np.size(data[0::,0]) #number of rows and columns in data array
ncol = np.size(data[0,0::])
#print nrow, ncol
data = data.ravel()
data = np.array([float(x) for x in data])
data = data.reshape(nrow,ncol)

# data is nicely in the form floated and regularized form:
# [sur, class, sex, age, sibsp, parch, fare, embarked]

# Now let's round ages to the nearest year and round fares to one decimal. This makes reading results easier
ntrain = np.size(data,0)
for x in xrange(ntrain):
    data[x,6] = round(data[x,6],1)
for x in xrange(ntrain):
    data[x,3] = round(data[x,3],0)

#Perform all of the same conversions on the testdata as we did on data (train)
#NOTE the test data doesn't have a survival column, so indices will generally be one less than for data

#create the TEST data array from the original file test.csv, as explained in kaggle tutorial
testcsv = csv.reader(open("../test.csv", 'rb'))
header = testcsv.next()
testdata=[]
for row in testcsv:
    testdata.append(row)
testdata = np.array(testdata)

#convert sex F,M to 1,0
testdata[testdata[0::,2] == "female", 2] = 1
testdata[testdata[0::,2] == "male", 2] = 0

#convert embarked strings into numbers for easier coding
testdata[testdata[0::,9] == "S", 9] = 0
testdata[testdata[0::,9] == "C", 9] = 1
testdata[testdata[0::,9] == "Q", 9] = 2
testdata[testdata[0::,9] == "", 9] = 3

# for passengers with no age, put age 1000
testdata[testdata[0::,3] == "", 3] = 1000
# passengers with no fare given. there is only one man and he is from 3rd class S. That average fare was around 12...
testdata[testdata[0::,7] == "", 7] = 12

testdata = scipy.delete(testdata,8,1) #delete cabin
testdata = scipy.delete(testdata,6,1) # delete ticket number
testdata = scipy.delete(testdata,1,1) # delete name


nrow = np.size(testdata[0::,0]) #number of rows and columns in data array
ncol = np.size(testdata[0,0::])
#print nrow, ncol
testdata = testdata.ravel()
testdata = np.array([float(x) for x in testdata])
testdata = testdata.reshape(nrow,ncol)

# testdata is nicely in the form floated and regularized form:
# [class, sex, age, sibsp, parch, fare, embarked]

# Now let's round ages to the nearest year and round fares to one decimal. This makes reading results easier
ntest = np.size(testdata,0)
for x in xrange(ntest):
    testdata[x,5] = round(testdata[x,5],1)
for x in xrange(ntest):
    testdata[x,2] = round(testdata[x,2],0)


# Now let's create the test data in the same form as train data, i.e. with 8 columns.
# Since we don't know the survival value of of any of the test data, we'll put '2' in the first column...
# Therefore the new survival values are 0=die, 1=live, 2=don't know
testdata8=np.zeros([418,8])
for x in xrange(nrow):
    testdata8[x] = insert(testdata[x],0,2)

#print testdata8[0:10]
#print testdata[0:10]

np.set_printoptions(linewidth=132)

# We can now call the following function, and ask for:
# "train", which returns data (891,8) array
# "test", which returns testdata (418,7) array
# "test8", which returns testdata8 (418,8) array
# "originaldata", which returns full data (418,8) array


#We'll also need to replace placeholder values for age = 1000
def convertages(dataset,ageindex): #ageindex=3 for test8 and data, and 2 for testdata
    dataset=dataset.copy() #start with a copy of the regularized data, either data, test8, or testdata
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


def titandata(dataset):
    if dataset == "train":
        return data
    elif dataset == "test":
        return testdata
    elif dataset == "test8":
        return testdata8
    elif dataset == "originaldata":
        return fulldata
    else:
        print "ERROR: input must be string 'test8', 'test', 'train', or 'originaldata'"
        return


__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

import csv as csv
import numpy as np
import scipy
from numpy import *
from sklearn.ensemble import RandomForestClassifier
from time import time
from DataFilters import df, dfrange, showstats
from PrepareTitanicData import titandata, convertages

start = time()

set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)


data=titandata("train") #(891,8) array
testdata=titandata("test") #(418,7) array
test8 = titandata("test8") #(418,8) array
totdata = vstack((data,test8)) # stacks data on top of test8 to create a (1309,8) array
# convert unknown placeholder ages to reasonable averages
data = convertages(data,3)
test8 = convertages(test8,3)
testdata = convertages(testdata,2)
totdata = convertages(totdata,3)


def genderpred(dataset):
    new = []
    for row in dataset: # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
        if (row[2] == 1):
            new.append(1)
        else:
            new.append(0)
    return np.array(new) # convert to array

def f3sm12pred(dataset):
    new = []
    for row in dataset: # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
        if (row[2] == 1) and not( (row[1] == 3) & (row[7] == 0) ):
            new.append(1)
        elif (row[2] == 0) and (row[3] <= 10) and not(row[1] == 3):
            new.append(1)
        else:
            new.append(0)
    return np.array(new) # convert to array

def newpred(dataset): #3rd class males age<=12 with sibsp = 0,1 and 3rd class females S age<=8 with sibsp=0,1 make it
    new = []
    for row in dataset: # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
        if (row[2] == 1) and not( (row[1] == 3) & (row[7] == 0) ):
            new.append(1)
        elif (row[2] == 0) and (row[3] <= 12) and not(row[1] == 3):
            new.append(1)
        elif (row[2] == 0) and (row[3] <= 12) and (row[1] == 3) and (0 <= row[4] <= 1 ):
            new.append(1)
        elif (row[2] == 1) and (row[3] <= 8) and (row[1] == 3) and (row[7] == 0) and  (0 <= row[4] <= 1 ):
            new.append(1)
        else:
            new.append(0)
    return np.array(new) # convert to array

def predicttrain(pred):
    comptrain = pred-data[0::,0] #compare pred with reality
    #print np.nonzero(comptrain) #This shows which elements were predicted wrong
    numwrong = sum(abs(comptrain[0::])==1)
    score = round(1 - float(numwrong) / float(np.size(comptrain)),5)
    return score

def comparepreds(pred1,pred2, dataset = testdata): #takes in predictions in form of 418,1 array
    dif = pred1-pred2 #+1,0,-1 for each passenger
    dispass = np.nonzero(dif)[0] # passengers numbers for which the predictions disagree.
    # the [0] due to formatting of nonzero function
    dataset = dataset.copy()
    n = np.size(dataset[0]) #size of a row... usually 7 or 8
    datadispass = dataset[dispass] # the data for those passengers
    numdifpass = np.size(dispass) # how many of them there are
    datadp = np.zeros([numdifpass,n+1]) # placeholder
    for x in xrange(numdifpass):
        dp = dispass[x] # passenger index for each disagreement
        datadp[x] = insert(datadispass[x],0,dif[dp]) #insert the value of dif into the 0th column
    print "The number of disagreements: %i" %numdifpass
    print "The disagreements: +1 means (pred1=1, pred2=0) and -1 means (pred1=0 and pred2=1)"
    print datadp
    return

def randomforests(numforests, n_est, traindataset, testdataset):
    Forest = RandomForestClassifier(n_estimators = n_est)
    # This function will run many random forests and yield the average prediction for each passenger.
    # NOTE: It takes about 40 sec to run this code on my 4 yr old macbook pro for numforests=100, n_est=100
    # so don't make them too big
    ntest = size(testdataset[0::,0])
    RFCpredcount = np.zeros(ntest)
    for x in xrange(numforests):
        Forest = Forest.fit(traindataset[0::,1::],traindataset[0::,0])
        # fit the training data to the training output and create decision trees
        RFCpredcount += Forest.predict(testdataset)
        # Take the decision trees and run on the test data.
        # Note if using on test8 (or similar with sur value in 0th column), you must enter testdataset variable
        # in form test8[0::,1::]
        # Then add to RFCpredcount, which will count the number of times an RFC
        # (remember each loop creates a new random RFC) predicts the passenger will live.
    RFCpredprob = RFCpredcount/float(numforests)
    RFCpred = [round(x,0) for x in RFCpredprob]
    RFCpred = [int(x) for x in RFCpred]
    return np.array(RFCpred)




#totaltime = time() - start
#print "This code took %f s to run" %totaltime
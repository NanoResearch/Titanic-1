__author__ = 'michaelbinger'

# http://epicentre.massey.ac.nz/Portals/0/EpiCentre/Downloads/Personnel/MarkStevenson/Stevenson_logistic_regression_091208.pdf
###########
# IMPORTS #
###########
import numpy as np
#from numpy import *
from random import randint
from pylab import *
from itertools import combinations as combs
import sympy
import scipy
import csv
from MultivariateRegression import Jcost, gradientdescent, featurescale, glog
from PrepareTitanicData import titandata, convertages
from DataFilters import databin, df, dfrange
from predictions import predicttrain, f3sm12pred, comparepreds
from time import time
import pickle

start = time()
np.set_printoptions(suppress=True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)

####################################################################################################
# SUMMARY OF GENERAL REGRESSION ALGORITHM
####################################################################################################
# 1. Import Data
# 2. Convert strings to floats
# 3. seperate the data into "independent" or "predictor" variables X (m,n') and dependent varaible Y (m,1)
#   Notation: label 'm' indices by 'i' or 'j' and 'n' indices by 'a' or 'b'
# 4. Plot or analyze the data to determine the best functional form of the hypothesis h
# 5. If higher powers of any of the X[a] variables are needed then include extra columns into X to reflect this.
#   X will now have shape (m,n) where n = n' + number of higher power terms included.
# 6. Insert into the first column of X all 1's. (to accommodate the theta_0 term in the same notation)
#   X is now (m,n+1)
# 7. Feature scale the X[a], a=1..n as necessary, resulting in Xfs[a]
# 8. Be sure the correct hypothesis function h is implemented into the gradient descent algorithm, for both functions
#   Jcost() and gradientdescent(). h_theta (X) = f(theta.T*X) where f(z) = z for most continuous regression problems.
#   For discrete classification use logistic function f(z) = g(z) = 1/(1+e^(-z))
#   For some special cases non polynomial transformations may be required, for example f(z) = (1+z)/(1-z) etc.
# 9. Run gradient descent :
#   > graddes = gradientdescent(Xfs,Y,alpha,niter,lam)
#   > thetapred = graddes[0]
#   > Jsteps = graddes[1]
# 10. Plot the Jcost function vs. the iteration number to confirm convergence
# 11. Define algebraic symbols (via sympy) Xa to write the final prediction (hypothesis function)
# 12. Note the features scaled variables Xfs that went in to gradient descent are related to our original features X by
#   Xfs = (X-mean)/std, therefore our final hypothesis is :
#   h(X) = f( thetapred.T * Xfs )
####################################################################################################
####################################################################################################

####################################################################################################
# Algorithm for correctly implementing Log Reg With train, CV, and test sets:
####################################################################################################
# 1. Decide on model space to test. Which features, and what values for lam for each model?
# 2. Create those data sets with the appropriate interactions using logreg_prepdata()
# 3. Partition each set into training (), cross validation (CV), and test (T) sets. Be sure to use the same passengers
#    from each model for each of the 3 sets. Note that different models have different length output parameters theta,
#    therefore we need to apply them to CV and test sets of appropriate dimension. But when comparing between models
#    we should use the same CV data, just in different forms.
# 4. Perform Log Reg (with lam appropriate for model) on the training data for each model to get the parameters theta.
# 5. Calculate the CV cost (and the CV score) for each model. Choose the best model.
# 6. Calculate the test cost (and test score) using that model. This will give a good idea of generalization error.
# 7. Perhaps perform the above multiple times on different randomly chosen subsets
####################################################################################################

####################################################################################################
# DATA
####################################################################################################
data = titandata("train") #(891,8) array
test8 = titandata("test8") #(418,8) array
# [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# The survival values are 0=die, 1=live, 2=don't know (for test8)
totdata = vstack((data, test8)) # stacks data on top of test8 to create a (1309,8) array

# convert unknown placeholder ages to reasonable averages
data = convertages(data, 3)
test8 = convertages(test8, 3)
totdata = convertages(totdata, 3)

db = databin(data)
dba = databin(data, agebin="off")
tba = databin(test8, agebin="off")
####################################################################################################


def logreg_prepdata(dataset, interactionlist="none"):
    """
    This function takes in data in the shape (m,8) and outputs a longer array of shape
    (m, 13 + N_ints) where N_ints = number of interaction terms.
    This form is suitable for logistic regression, as the continuous vars age and fare have been
    feature scaled, and the discrete features are converted into binary vars.
    Note the ordering of index labels is changed.
    With no interaction terms we have each row of data transformed as:
    [sur, cl, sex, age, sibsp, parch, fare, city] ==>
    [sur, 1, <age>, <fare>, sex, cl1, cl2, sibsp0, sibsp1, parch0, parch1, city0, city1 ]
    where <age> = (age-mean(age))/std(age)
    and <fare> = (fare-mean(fare))/std(fare)
    Note: age, fare are continuous while other vars are binary (0 or 1)
    """
    ds = dataset.copy()
    m = np.size(ds[0::, 0]) # number of data samples
    xx = []
    y = ds[0::, 0].reshape([m, 1])

    for row in ds: # for each row of length 8, we create a new row of length 12, as indicated above.
        if row[1] == 1:
            cl1 = 1 # class = 1?
        else:
            cl1 = 0
        if row[1] == 2:
            cl2 = 1 # class = 2?
        else:
            cl2 = 0

        if row[4] == 0:
            sib0 = 1 # sibsp = 0?
        else:
            sib0 = 0
        if row[4] == 1:
            sib1 = 1 # sibsp = 1? (actually 1 or 2 since databin has been applied to get dba)
        else:
            sib1 = 0

        if row[5] == 0:
            par0 = 1 # parch = 0?
        else:
            par0 = 0
        if row[5] == 1:
            par1 = 1 # parch = 1? (actually 1 or 2 since databin has been applied to get dba)
        else:
            par1 = 0

        if row[7] == 0:
            city0 = 1 # city = 0(S)?
        else:
            city0 = 0
        if row[7] == 1:
            city1 = 1 # city = 1(C)?
        else:
            city1 = 0

        xx.append([1, row[3], row[6], row[2], cl1, cl2, sib0, sib1, par0, par1, city0, city1])

    xx = (np.array(xx)).reshape(m, 12)
    fs = featurescale(xx[0::, 0:3]) # FS the 2 continuous vars age and fare.
    xcontfs = fs[0]
    means = fs[1]
    stds = fs[2]
    datalin = np.hstack([y, xcontfs, xx[0::, 3::]]) # put the y, the cont vars, and binary vars back together.
    # 'lin'=linear terms only. interaction terms are an optional input variable:
    dataints = datalin.copy() # make a copy of datalin
    if interactionlist != "none": # if interaction terms are specified, include those and stack them
        for (i, j) in interactionlist:
            xintterm = (datalin[0::, i] * datalin[0::, j]).reshape(m, 1)
            dataints = np.hstack([dataints, xintterm])
    return [dataints, means, stds] # xints is a (m, 12+N_ints) array. means and stds are (1,2) arrays.

def surpred(xdata, thetapreds):
    """
    Given a data set, and some predicted values of theta,
    this function returns an (m,) array of 0's and 1's of survival predictions for each passenger in xdata.
    NOTE: xdata must be of shape (m,n+1) and thetapreds of shape (1,n+1)
    (this is how the gradient descent function outputs them, for easier reading).
    When taking each row inside the for loop, the row has shape (n+1,).
    The output from lrdat = logreg_prepdata()[0] is of shape (m,n+2) since sur is the 0th column.
    Thus you should input lrdat[0::,1::] for xdata.
    """
    shapetheta = np.shape(thetapreds)
    shapex = np.shape(xdata)
    if (shapex[1] != shapetheta[1]):
        print "Shape error"
        return
    pred = []
    for row in xdata:
        z = np.dot(thetapreds, row)
        if z >= 0:
            pred.append(1)
        else:
            pred.append(0)
    return np.array(pred)

def scorepreds(dataset, preds):
    """
    Returns the actual score for dataset of any shape (m, N), so long as the y (survival) values are in 0th column.
    preds must be of shape (m,1) or (m,) with values 0 or 1.

    """
    m = np.size(preds)
    preds = (preds.copy()).reshape(m, 1)

    if np.shape(dataset)[0] != m:
        return "ERROR"
    ds = np.hstack([preds, dataset])
    count = 0
    for row in ds:
        if row[0] == row[1]:
            count += 1
    return round(float(count) / float(m), 5)

class Datasets:
    """
    Creates class of data sets
    """
    trainper = 0.60
    cvper = 0.20
    testper = 0.20

    def __init__(self, dat, train=0.60, cv=0.20, test=0.20):
        self.data = dat
        trainper = train
        cvper = cv
        testper = test

    def sh(self):
        return np.shape(self.data)

    def m(self):
        return self.sh()[0]

    def n(self):
        return self.sh()[1] - 1

    def mtr(self):
        return int(round(Datasets.trainper * self.m()))

    def mcv(self):
        return int(round(Datasets.cvper * self.m()))

    def mtest(self):
        return int(round(Datasets.testper * self.m()))

    def train(self):
        return np.array(self.data[0: self.mtr()])

    def cv(self):
        return np.array(self.data[self.mtr(): self.mtr() + self.mcv()])

    def test(self):
        return np.array(self.data[self.mtr() + self.mcv(): self.m()])

def titanlogregCV(alpha, niter, lam, interactions, dataset=dba):
    """
    Returns score on CV data given
    """
    tempds = logreg_prepdata(dataset, interactionlist=interactions)[
        0] # prepare the data, yielding an (m, 13 + Nint) array
    tempds = Datasets(tempds) # make the data a class instance, so as to segregate it by train, cv, and test
    trainds = tempds.train() # pull out the training data
    y = trainds[0::, 0]
    x = trainds[0::, 1::]
    graddes = gradientdescent(x, y, alpha, niter, lam, logreg=True)
    thetapreds = graddes[0] # predicted values of thetas from grad des
    cvds = tempds.cv() # pull up the CV data to test our prediction on
    cvpreds = surpred(cvds[0::, 1::], thetapreds) # generate the predicted survivals
    cvscore = scorepreds(cvds, cvpreds) # compare the predictions to true results for CV
    return cvscore


dbds = Datasets(db)

print dbds.mtr(), dbds.mcv(), dbds.mtest()

print np.shape(dbds.train())

print np.shape(dbds.cv())

print np.shape(dbds.test())


# create all 55 possible quadratic interaction terms.
# Note there are really only 51, since cl1*cl2, sib0*sib1, par0*par1, and city0*city1 are always zero.
intlist = list(combs(range(2, 13), 2))

alpha = 0.2
niter = 20000
lam = 0

tLRCV = titanlogregCV(alpha, niter, lam, "none")
print "score on CV set for no ints:", tLRCV

print "Time elapsed:", time() - start

####################################################################################################
# This randomly selects 2 out of the 55 possible interaction terms and runs log reg using them.
# The function titanlogregCV is used, which returns the score on the CV data for each such randomly
#   created 2-interaction model.
# The results are written to a file.
run_rand_ints = False # This takes a long time for 1000 iterations
if run_rand_ints:
    newfile = open('../quadintsCVscores.txt', 'wb')
    count = 0
    results = []
    while count < 1000:
        count += 1
        r1 = randint(0, 54)
        r2 = randint(0, 54)
        rints = [intlist[r1], intlist[r2]]
        tLRCV = titanlogregCV(alpha, niter, lam, rints)
        res = [tLRCV, rints]
        newfile.write(str(res) + '\n')
        results.append(res)
        print "score on CV set for ints", rints, ":", tLRCV
    newfile.close()
print "Time elapsed:", time() - start
####################################################################################################

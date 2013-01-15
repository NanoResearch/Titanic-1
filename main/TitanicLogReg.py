__author__ = 'michaelbinger'

# http://epicentre.massey.ac.nz/Portals/0/EpiCentre/Downloads/Personnel/MarkStevenson/Stevenson_logistic_regression_091208.pdf

import csv
import numpy as np
from numpy import *
from random import randint
from pylab import *
import sympy
import scipy
from MultivariateRegression import Jcost, gradientdescent, featurescale, glog
from PrepareTitanicData import titandata, convertages
from DataFilters import databin, df, dfrange
from predictions import predicttrain, f3sm12pred, comparepreds
from time import time

start = time()
set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)

def normeqn(X,y):
    return np.dot( np.linalg.inv( np.dot(X.T,X) ), np.dot(X.T, y) ).T

# Summary of general regression algorithm
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


############# DATA ################################
data=titandata("train") #(891,8) array
test8 = titandata("test8") #(418,8) array
# [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# The survival values are 0=die, 1=live, 2=don't know (for test8)
totdata = vstack((data,test8)) # stacks data on top of test8 to create a (1309,8) array

# convert unknown placeholder ages to reasonable averages
data = convertages(data,3)
test8 = convertages(test8,3)
totdata = convertages(totdata,3)

db = databin(data)
dba = databin(data, agebin ="off")
tba = databin(test8, agebin ="off")
###################################################

def logreg_sexonly():
    y = data[::,0]
    m = np.size(y)
    y = y.reshape([m,1])
    x2 = data[::,2].reshape([m,1])

    xones = np.ones([m,1])
    print np.shape(xones), np.shape(x2)
    x = np.hstack([xones,x2])

    alpha = 0.1
    niter = 10000
    lam = 0
    graddes = gradientdescent(x,y,alpha,niter,lam, logreg = True)

    thetapred = graddes[0]
    Jsteps = graddes[1]
    print "prediction for theta:", thetapred
    #print Jsteps

    scatter(np.arange(niter)+1,Jsteps)
    xlabel("Number of iterations")
    ylabel("Jcost")
    title("The convergence of the cost function")
    show()

    xf = np.array([1,1])
    xm = np.array([1,0])

    Pf = glog(np.dot(thetapred,xf))
    Pm = glog(np.dot(thetapred,xm))

    print Pf, Pm

    print "Time elapsed:", time() - start

def logreg_sex_and_class():
    y = data[::,0]
    m = np.size(y)
    y = y.reshape([m,1])

    xs = data[::, 2].reshape([m,1])
    xcl = data[::, 1].reshape([m,1])
    xcl1 = (xcl == 1).astype(int)
    xcl2 = (xcl == 2).astype(int)
    xcl3 = (xcl == 3).astype(int)


    xones = np.ones([m,1])
    print np.shape(xcl2), np.shape(xs)
    x1int = ((xs.reshape(m))*(xcl1.reshape(m))).reshape([m,1])
    x2int = ((xs.reshape(m))*(xcl2.reshape(m))).reshape([m,1])

    x = np.hstack([xones, xcl1, xcl2, xs, x1int, x2int])

    alpha = 0.1
    niter = 40000
    lam = 0
    graddes = gradientdescent(x,y,alpha,niter,lam, logreg = True)

    thetapred = graddes[0]
    Jsteps = graddes[1]
    print "prediction for theta:", thetapred
    #print Jsteps

    scatter(np.arange(niter)+1,Jsteps)
    xlabel("Number of iterations")
    ylabel("Jcost")
    title("The convergence of the cost function")
    show()

    for cl in [1,2,3]:
        cl1 = int(cl == 1)
        cl2 = int(cl == 2)
        cl3 = int(cl == 3)
        for sex in [0,1]:
            print "class=", cl, "and sex=", sex
            xx = np.array([1,cl1,cl2,sex, cl1*sex, cl2*sex])
            print glog(np.dot(thetapred, xx))
    print "Time elapsed:", time() - start

def logreg_sex_class_city():
    y = data[::,0]
    m = np.size(y)
    y = y.reshape([m,1])

    xs = data[::, 2].reshape([m,1])
    xcl = data[::, 1].reshape([m,1])
    xem = data[::, 7].reshape([m,1])

    xcl1 = (xcl == 1).astype(int)
    xcl2 = (xcl == 2).astype(int)
    #xcl3 = (xcl == 3).astype(int)
    xem1 = (xem == 0).astype(int)
    xem2 = (xem == 1).astype(int)
    #xem3 = (xem == 2).astype(int)

    xones = np.ones([m,1])

    xscl1int = ((xs.reshape(m))*(xcl1.reshape(m))).reshape([m,1])
    xscl2int = ((xs.reshape(m))*(xcl2.reshape(m))).reshape([m,1])
    xsem1int = ((xs.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xsem2int = ((xs.reshape(m))*(xem2.reshape(m))).reshape([m,1])

    xcl1em1 = ((xcl1.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xcl2em1 = ((xcl2.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xcl1em2 = ((xcl1.reshape(m))*(xem2.reshape(m))).reshape([m,1])
    xcl2em2 = ((xcl2.reshape(m))*(xem2.reshape(m))).reshape([m,1])

    xscl1em1 = ((xs.reshape(m))*(xcl1.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xscl2em1 = ((xs.reshape(m))*(xcl2.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xscl1em2 = ((xs.reshape(m))*(xcl1.reshape(m))*(xem2.reshape(m))).reshape([m,1])
    xscl2em2 = ((xs.reshape(m))*(xcl2.reshape(m))*(xem2.reshape(m))).reshape([m,1])

    doubles = np.hstack([xscl1int, xscl2int, xsem1int, xsem2int, xcl1em1, xcl2em1, xcl1em2, xcl2em2])
    triples = np.hstack([xscl1em1, xscl2em1, xscl1em2, xscl2em2])


    x = np.hstack([xones, xcl1, xcl2, xs, xem1, xem2, doubles, triples])
    # note that after running with the triples on and off there was virtually no difference in results...
    #perhaps only linear and quadratic terms are necessary?

    alpha = 0.3
    niter = 50000
    lam = 0
    graddes = gradientdescent(x,y,alpha,niter,lam, logreg = True)

    thetapred = graddes[0]
    Jsteps = graddes[1]
    print "prediction for theta:", thetapred
    #print Jsteps

    scatter(np.arange(niter)+1,Jsteps)
    xlabel("Number of iterations")
    ylabel("Jcost")
    title("The convergence of the cost function")
    show()

    for cl in [1,2,3]:
        cl1 = int(cl == 1)
        cl2 = int(cl == 2)
        cl3 = int(cl == 3)
        for em in [0,1,2]:
            em1 = int(em == 0)
            em2 = int(em == 1)
            em3 = int(em == 2)
            for sex in [0,1]:
                print "class=", cl, "and sex=", sex, "and emb=", em
                xx = np.array([1,cl1,cl2,sex, em1, em2, cl1*sex, cl2*sex, em1*sex, em2*sex,
                               cl1*em1, cl2*em1, cl1*em2, cl2*em2, sex*cl1*em1, sex*cl2*em1, sex*cl1*em2, sex*cl2*em2])
                print glog(np.dot(thetapred, xx))
    print "Time elapsed:", time() - start

def farelogreg():
    """
    This runs logistic regression using only the fare variable as the predictor for y = survival.
    """
    #datass = dfrange(20, 100, 3, df([[3,1],[1,2],[0,7]],db))
    datass = db
    y = datass[::,0]
    m = np.size(y)
    y = y.reshape([m,1])

    x6 = datass[::, 6].reshape([m,1])
    xones = np.ones([m,1])
    x = np.hstack([xones,x6])

    fs = featurescale(x)
    xfs = fs[0]
    means = fs[1]
    stds = fs[2]

    alpha = 0.2
    niter = 1000
    lam = 0

    #scatter(x6,y)
    #xlabel("Number of iterations")
    #ylabel("Jcost")
    #title("The convergence of the cost function")
    #show()

    graddes = gradientdescent(xfs,y,alpha,niter,lam, logreg = True)

    thetapred = graddes[0]
    Jsteps = graddes[1]
    print "prediction for theta:", thetapred
    print means, stds

    scatter(np.arange(niter)+1,Jsteps)
    xlabel("Number of iterations")
    ylabel("Jcost")
    title("The convergence of the cost function")
    show()

    print "Time elapsed:", time() - start

#farelogreg()

def fareplots(nbins, features, dataset = db, faremin = 0, faremax = 1000):
    features = list(features) # new copy of the features list
    dataset = dataset.copy() #start with a copy of the regularized data, either data, test8, or testdata
    datatemp = df(features, dataset)
    fares = datatemp[0::,6]
    fmax = np.max(fares)
    fmin = 0
    if faremax != 1000:
        fmax = faremax
    if faremin != 0:
        fmin = faremin
    counts = []
    faren = []
    bin = float(fmax - fmin) / float(nbins)
    for n in xrange(nbins):
        truthtable = (fares[0::] >= n*bin + fmin) & (fares[0::] <= (n+1)*bin + fmin)
        faren.append((n + 0.5)*bin + fmin)
        counts.append(np.size(fares[truthtable]))
    scatter(faren, counts)
    show()

#fareplots(20, [[0,2],[3,1],[0,7]], faremin = 5, faremax = 10)
#fareplots(20, [[0,2],[1,1],[0,7]], faremin = 5, faremax = 200)

y = dba[::,0]
m = np.size(y)
y = y.reshape([m,1])

# Now we will map the input data into a longer array that we can dot into the thetas.
# [sur?, cl, sex, age, sibsp, parch, fare, city] ==>
# [1, age, fare, sex, cl1, cl2, sibsp0, sibsp1, parch0, parch1, city0, city1 ]
# age, fare are continuous while other vars are 0 or 1

xx = []
for row in dba:
    if row[1] == 1: cl1 = 1
    else: cl1 = 0
    if row[1] == 2: cl2 = 1
    else: cl2 = 0

    if row[4] == 0: sib0 = 1
    else: sib0 = 0
    if row[4] == 1: sib1 = 1
    else: sib1 = 0

    if row[5] == 0: par0 = 1
    else: par0 = 0
    if row[5] == 1: par1 = 1
    else: par1 = 0

    if row[7] == 0: city0 = 1
    else: city0 = 0
    if row[7] == 1: city1 = 1
    else: city1 = 0

    xx.append([1, row[3], row[6], row[2], cl1, cl2, sib0, sib1, par0, par1, city0, city1])

xx = (np.array(xx)).reshape(m,12)

fs = featurescale(xx[0::,0:3])
xcontfs = fs[0]
means = fs[1]
stds = fs[2]

xlin = np.hstack([xcontfs, xx[0::,3::]])

mt = 418
testxx = []
for row in tba:
    if row[1] == 1: cl1 = 1
    else: cl1 = 0
    if row[1] == 2: cl2 = 1
    else: cl2 = 0

    if row[4] == 0: sib0 = 1
    else: sib0 = 0
    if row[4] == 1: sib1 = 1
    else: sib1 = 0

    if row[5] == 0: par0 = 1
    else: par0 = 0
    if row[5] == 1: par1 = 1
    else: par1 = 0

    if row[7] == 0: city0 = 1
    else: city0 = 0
    if row[7] == 1: city1 = 1
    else: city1 = 0

    testxx.append([1, row[3], row[6], row[2], cl1, cl2, sib0, sib1, par0, par1, city0, city1])

testxx = (np.array(testxx)).reshape(mt,12)

fs = featurescale(testxx[0::,0:3])
testxcontfs = fs[0]
testmeans = fs[1]
teststds = fs[2]

testxlin = np.hstack([testxcontfs, testxx[0::,3::]])

alpha = 0.3
niter = 10000
lam = 0

graddes = gradientdescent(xlin, y, alpha, niter, lam, logreg = True)
thetapred = graddes[0]
Jsteps = graddes[1]
print "prediction for theta:", thetapred
print means, stds

#scatter(np.arange(niter)+1,Jsteps)
#xlabel("Number of iterations")
#ylabel("Jcost")
#title("The convergence of the cost function")
#show()

pred = []
for row in xlin:
    z = np.dot(thetapred, row)
    if z >= 0: pred.append(1)
    else: pred.append(0)

pred = np.array(pred)
scorelin = predicttrain(pred)
print scorelin


posints = [(1, 4), (1, 8), (1, 9), (2, 6), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 8), (4, 9),
           (4, 10), (4, 11), (5, 10), (5, 11), (7, 9), (8, 10), (9, 10)]

posints2 = [(1,8),(3,4)]

intlist = [(i,j) for i in xrange(1,12) for j in xrange(1,12)]
intlist = [x for x in intlist if x[0] <  x[1]]

xints = xlin.copy()
for (i,j) in intlist:
    xint = (xlin[0::, i] * xlin[0::, j]).reshape(m,1)
    xints = np.hstack([ xints, xint])

testxints = testxlin.copy()
for (i,j) in intlist:
    testxint = (testxlin[0::, i] * testxlin[0::, j]).reshape(mt,1)
    testxints = np.hstack([ testxints, testxint])

graddes = gradientdescent(xints, y, alpha, niter, lam, logreg = True)
thetapred = graddes[0]
Jsteps = graddes[1]

scatter(np.arange(niter)+1,Jsteps)
xlabel("Number of iterations")
ylabel("Jcost")
title("The convergence of the cost function")
show()

pred = []
for row in xints:
    z = np.dot(thetapred, row)
    if z >= 0: pred.append(1)
    else: pred.append(0)

testpred = []
for row in testxints:
    z = np.dot(thetapred, row)
    if z >= 0: testpred.append(1)
    else: testpred.append(0)
testpred = np.array(testpred)

print np.shape(testpred)
comparepreds(testpred, f3sm12pred(test8))

newcsv = csv.writer(open('../log_reg_all_ints.csv','wb'))
newpredict = testpred
for x in xrange(418):
    if newpredict[x]==0:
        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if newpredict[x]==1:
        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.


pred = np.array(pred)
scoreint = predicttrain(pred)
dif = scoreint-scorelin
print "With interaction terms the score and number correct improvement on train data:"
print scoreint, "  ", round(dif*m)



import sys
sys.exit()

print  "1=age, 2=fare, 3=sex, 4=cl1, 5=cl2, 6=sib0, 7=sib1, 8=par0, 9=par1, 10=city0, 11=city1]"
print "For (i,j) the score on train data and difference from linear case"

posints = []
for (i,j) in intlist:
    xint = (xlin[0::, i] * xlin[0::, j]).reshape(m,1)
    xlin1int = np.hstack([ xlin, xint])
    graddes = gradientdescent(xlin1int, y, alpha, niter, lam, logreg = True)
    thetapred = graddes[0]
    #Jsteps = graddes[1]

    pred = []
    for row in xlin1int:
        z = np.dot(thetapred, row)
        if z >= 0: pred.append(1)
        else: pred.append(0)

    pred = np.array(pred)
    scoreint = predicttrain(pred)
    dif = scoreint-scorelin
    print (i,j), "  ", scoreint, "  ", round(dif*m)
    if dif > 0:
        posints.append((i,j))

print posints

print "Time elapsed:", time() - start

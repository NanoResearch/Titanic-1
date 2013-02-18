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
from TitanicLogReg import *
import pickle

start = time()
np.set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)



def logreg_sexonly():
    """
    Here we implement logistic regression using only sex as the independent predictor var
    """
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

    xf = np.array([1,1]) # X for females
    xm = np.array([1,0]) # X for males

    Pf = glog(np.dot(thetapred,xf)) # predicted survival probabilities for female and male
    Pm = glog(np.dot(thetapred,xm))

    print Pf, Pm

    print "Time elapsed:", time() - start

def logreg_sex_and_class():
    """
    Logistic Regression using both sex and class as independent feature vars
    """
    y = data[::,0]
    m = np.size(y)
    y = y.reshape([m,1])

    xs = data[::, 2].reshape([m,1])
    xcl = data[::, 1].reshape([m,1])
    xcl1 = (xcl[::] == 1).astype(int) # convert class into two separate binary vars. xcl1=1 only if 1st class. 0 else
    xcl2 = int(xcl[::] == 2) # xcl2=1 only if 2nd class. 0 else
    xcl3 = int(xcl == 3) # not used

    xones = np.ones([m,1])
    print np.shape(xcl2), np.shape(xs)
    x1int = ((xs.reshape(m))*(xcl1.reshape(m))).reshape([m,1]) # include interaction terms sex*cl1 and sex*cl2
    x2int = ((xs.reshape(m))*(xcl2.reshape(m))).reshape([m,1])

    x = np.hstack([xones, xcl1, xcl2, xs, x1int, x2int]) # full x array

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

    for cl in [1,2,3]: # generates the predicted survival table
        cl1 = int(cl == 1)
        cl2 = int(cl == 2)
        cl3 = int(cl == 3)
        for sex in [0,1]:
            print "class=", cl, "and sex=", sex
            xx = np.array([1,cl1,cl2,sex, cl1*sex, cl2*sex])
            print glog(np.dot(thetapred, xx))
    print "Time elapsed:", time() - start

def logreg_sex_class_city():
    """
    Incorporating sex, class, and city into the logistic regression.
    """
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

    xscl1 = ((xs.reshape(m))*(xcl1.reshape(m))).reshape([m,1])
    xscl2 = ((xs.reshape(m))*(xcl2.reshape(m))).reshape([m,1])
    xsem1 = ((xs.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xsem2 = ((xs.reshape(m))*(xem2.reshape(m))).reshape([m,1])

    xcl1em1 = ((xcl1.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xcl2em1 = ((xcl2.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xcl1em2 = ((xcl1.reshape(m))*(xem2.reshape(m))).reshape([m,1])
    xcl2em2 = ((xcl2.reshape(m))*(xem2.reshape(m))).reshape([m,1])

    xscl1em1 = ((xs.reshape(m))*(xcl1.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xscl2em1 = ((xs.reshape(m))*(xcl2.reshape(m))*(xem1.reshape(m))).reshape([m,1])
    xscl1em2 = ((xs.reshape(m))*(xcl1.reshape(m))*(xem2.reshape(m))).reshape([m,1])
    xscl2em2 = ((xs.reshape(m))*(xcl2.reshape(m))*(xem2.reshape(m))).reshape([m,1])

    doubles = np.hstack([xscl1, xscl2, xsem1, xsem2, xcl1em1, xcl2em1, xcl1em2, xcl2em2]) #quadratic interactions
    triples = np.hstack([xscl1em1, xscl2em1, xscl1em2, xscl2em2]) #cubic interactions


    x = np.hstack([xones, xcl1, xcl2, xs, xem1, xem2, doubles, triples])
    # note that after running with the triples on and off there was virtually no difference in results...
    # perhaps only linear and quadratic terms are necessary?

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

    for cl in [1,2,3]: # create predicted survival table
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

    scatter(x6,y)
    show()

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
    binlen = float(fmax - fmin) / float(nbins)
    for n in xrange(nbins):
        truthtable = (fares[0::] >= n*binlen + fmin) & (fares[0::] <= (n+1)*binlen + fmin)
        faren.append((n + 0.5)*binlen + fmin)
        counts.append(np.size(fares[truthtable]))
    scatter(faren, counts)
    show()

    #fareplots(20, [[0,2],[3,1],[0,7]], faremin = 5, faremax = 10)
    #fareplots(20, [[0,2],[1,1],[0,7]], faremin = 5, faremax = 200)



if __name__ == "__main__":
    # Only runs the code below if running this file directly, i.e. not if importing the functions above.
    y = dba[::,0]
    m = np.size(y)
    y = y.reshape([m,1])

    dbaprep = logreg_prepdata(dba) # the train data
    xlindba = dbaprep[0]
    meansdba = dbaprep[1]
    stdsdba = dbaprep[2]

    tbaprep = logreg_prepdata(tba) # the test data
    xlintba = tbaprep[0]
    meanstba = tbaprep[1]
    stdstba = tbaprep[2]

    print "means for age and fare, train data: ", meansdba
    print "means for age and fare, test data: ", meanstba
    print "stds for age and fare, train data: ", stdsdba
    print "stds for age and fare, test data: ", stdstba

    alpha = 0.3
    niter = 10000
    lam = 0

    graddes = gradientdescent(xlindba, y, alpha, niter, lam, logreg = True)
    thetapredlin = graddes[0]
    Jsteps = graddes[1]
    print "prediction for theta with only linear terms:"
    print thetapredlin

    #scatter(np.arange(niter)+1,Jsteps)
    #xlabel("Number of iterations")
    #ylabel("Jcost")
    #title("The convergence of the cost function")
    #show()

    predlin = surpred(xlindba, thetapredlin)
    scorelin = predicttrain(predlin)
    print "scorelin", scorelin


    #posints = [(1, 4), (1, 8), (1, 9), (2, 6), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 8), (4, 9),
    #           (4, 10), (4, 11), (5, 10), (5, 11), (7, 9), (8, 10), (9, 10)]
    #posints2 = [(1,8),(3,4)]

    # create all 55 possible quadratic interaction terms.
    # Note there are really only 51, since cl1*cl2, sib0*sib1, par0*par1, and city0*city1 are always zero.
    intlist = [(i,j) for i in xrange(1,12) for j in xrange(1,12)]
    intlist = [x for x in intlist if x[0] <  x[1]]

    dbaprepints = logreg_prepdata(dba, interactionlist = intlist) # the train data
    xintsdba = dbaprepints[0]
    tbaprepints = logreg_prepdata(tba, interactionlist = intlist) # the train data
    xintstba = tbaprepints[0]

    graddes = gradientdescent(xintsdba, y, alpha, niter, lam, logreg = True)
    thetapredints = graddes[0]
    Jsteps = graddes[1]

    scatter(np.arange(niter)+1,Jsteps)
    xlabel("Number of iterations")
    ylabel("Jcost")
    title("The convergence of the cost function")
    show()

    print np.shape(xintsdba), np.shape(thetapredints)
    predints = surpred(xintsdba, thetapredints)
    scoreints = predicttrain(predints)
    print "scoreints", scoreints


    testpredlin = surpred(xlintba, thetapredlin)
    testpredints = surpred(xintstba, thetapredints)


    comparepreds(testpredlin, f3sm12pred(test8))
    comparepreds(testpredints, f3sm12pred(test8))
    comparepreds(testpredlin, testpredints)

    #newcsv = csv.writer(open('../log_reg_all_ints.csv','wb'))
    #newpredict = testpred
    #for x in xrange(418):
    #    if newpredict[x]==0:
    #        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    #    if newpredict[x]==1:
    #        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.


    dif = scoreints-scorelin
    print "With interaction terms the score and number correct improvement on train data:"
    print scoreints, "  ", round(dif*m)


    print "Time elapsed:", time() - start

    #import sys
    #sys.exit()

    print  "1=age, 2=fare, 3=sex, 4=cl1, 5=cl2, 6=sib0, 7=sib1, 8=par0, 9=par1, 10=city0, 11=city1]"
    print "For (i,j) the score on train data and difference from linear case"

    def interactionterms():
        """
        This runs grad des with 1 quadratic interaction term at a time (and all linear terms), and compares the
        result to the linear case.
        """
        posints = []
        for (i,j) in intlist:
            xint = (xlindba[0::, i] * xlindba[0::, j]).reshape(m,1)
            xlin1int = np.hstack([ xlindba, xint])
            graddes = gradientdescent(xlin1int, y, 0.3, 10000, 0, logreg = True)
            thetapred = graddes[0]
            pred = surpred(xlin1int, thetapred)
            scoreint = predicttrain(pred)
            dif = scoreint-scorelin
            print (i,j), "  ", scoreint, "  ", round(dif*m)
            if dif > 0:
                posints.append((i,j))

        print posints

    # interactionterms()
    # Note this takes about 3 min

    print "Time elapsed:", time() - start

intlist = list(combs(range(2, 13), 2))

alpha = 0.2
niter = 20000
lam = 0

# This was a first implementation of what later became titanlogregCV()
for (i, j) in intlist:
    tempds = logreg_prepdata(dba, [(i, j)])[0]
    tempds = Datasets(tempds)
    trainds = tempds.train()

    y = trainds[0::, 0]
    x = trainds[0::, 1::]

    graddes = gradientdescent(x, y, alpha, niter, lam, logreg=True)
    thetapredints = graddes[0]

    cvds = tempds.cv()
    cvpreds = surpred(cvds[0::, 1::], thetapredints)
    cvscore = scorepreds(cvds, cvpreds)

    print "The score of CV data with linear terms and the interaction term", (i, j), "is: ", cvscore


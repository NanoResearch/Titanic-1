__author__ = 'michaelbinger'

###########
# IMPORTS #
###########
import numpy as np
from numpy import *
from random import randint
from pylab import *
import sympy
import scipy
import csv
from MultivariateRegression import Jcost, gradientdescent, featurescale, glog
from PrepareTitanicData import titandata, convertages
from DataFilters import databin, df, dfrange
from predictions import predicttrain, f3sm12pred, comparepreds
from TitanicLogReg import logreg_prepdata
from time import time

start = time()
set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)


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

def featprob(features, dataset = data):
    """
    This yields the statistical probability of a data sample having the input list of feature values.
    Note features is a list of the form [ [v1,i1], [v2,i2], [v3,i3],... ] where v's are values and i's are indices.
    """
    dataset = dataset.copy()
    dff = df(features, dataset)
    if dff == []:
        num = 0
    else:
        num = np.size( dff[0::,0] )
    tot = np.size(dataset[0::,0])
    return float(num) / float(tot)

def entropy(index, dataset = data):
    """
    Returns the information entropy, or entropy, of a feature (call it X) given by 'index' in dataset.
    The entropy is the smallest number of bits required (on average) to transmit a stream of x's drawn from the
    distribution of X.
    high entropy <---> a uniform, boring, featureless dist. of X
    low entropy <---> highly variable dist. of X (clumpy)
    """
    dataset = dataset.copy()
    possvals = set( dataset[0::, index] )
    ent = 0
    for val in possvals:
        prob = featprob([ [val, index] ], dataset = dataset)
        if prob != 0:
            ent += -prob * np.log2( prob )
    return ent

def jointentropy(index1, index2, dataset = data):
    """
    This returns the joint entropy of two variables, which are indicated by index1 and index2 in dataset.
    The joint entropy of two random variables is always greater than the maximum of individual entropies,
    and less than the sum of two independent entropies.
    """
    dataset = dataset.copy()
    possvals1 = set( dataset[0::, index1] ) # possible values of index1
    possvals2 = set( dataset[0::, index2] )
    print possvals1, possvals2
    je = 0
    for v1 in possvals1:
        for v2 in possvals2:
            p12 = featprob([ [v1, index1], [v2, index2] ], dataset = dataset)
            if p12 != 0:
                je += -p12 * np.log2( p12 )
    return je

def condentropy(yindex, xindex, dataset = data):
    """
    Returns the conditional entropy H(Y|X) where.
    """
    dataset = dataset.copy()
    possvalsY = set( dataset[0::, yindex] ) # possible values of index1
    possvalsX = set( dataset[0::, xindex] )
    #print possvalsY, possvalsX
    ce = 0
    for valY in possvalsY:
        for valX in possvalsX:
            pXY = featprob([ [valY, yindex], [valX, xindex] ], dataset = dataset)
            pX = featprob([ [valX, xindex] ], dataset = dataset)
            if pXY != 0:
                ce += -pXY * np.log2( pXY / ( pX ) )
    return ce

def mutualinfo(index1, index2, dataset = data):
    """
    This returns the mutual information (MI) of two variables, which are indicated by index1 and index2 in dataset.
    The MI measures the dependence of two random variables.
    """
    dataset = dataset.copy()
    possvals1 = set( dataset[0::, index1] ) # possible values of index1
    possvals2 = set( dataset[0::, index2] )
    #print possvals1, possvals2
    mi = 0
    for v1 in possvals1:
        for v2 in possvals2:
            p12 = featprob([ [v1, index1], [v2, index2] ], dataset = dataset)
            p1 = featprob([ [v1, index1] ], dataset = dataset)
            p2 = featprob([ [v2, index2] ], dataset = dataset)
            if p12 != 0:
                mi += p12 * np.log2( p12 / ( p1*p2 ) )
    return mi

def infogain(index, dataset = data):
    """
    Returns the information gain for feature given by 'index'.
    This measures the reduction in entropy (or increase in info) caused by partitioning the examples by that feature.
    """
    dataset = dataset.copy()
    return entropy(0, dataset) - condentropy(0, index, dataset = dataset)

def condmutualinfo(iX, iY, iZ, dataset = data):
    """
    Returns the conditional mutual info I(X,Y|Z).
    """
    ds = dataset.copy()
    possvalsX = set( ds[0::, iX] ) # possible values of index iX
    possvalsY = set( ds[0::, iY] )
    possvalsZ = set( ds[0::, iZ] )
    cmi = 0
    for valX in possvalsX:
        for valY in possvalsY:
            for valZ in possvalsZ:
                pXYZ = featprob([ [valX, iX], [valY, iY], [valZ, iZ] ], dataset = ds)
                pXZ = featprob([ [valX, iX], [valZ, iZ] ], dataset = ds)
                pYZ = featprob([ [valY, iY], [valZ, iZ] ], dataset = ds)
                pZ = featprob([ [valZ, iZ] ], dataset = ds)
                if pXYZ != 0:
                    cmi += pXYZ * np.log2( ( pZ*pXYZ ) / ( pXZ*pYZ ) )
    return cmi

def multimutualinfo(iX, iY, iZ, dataset = data):
    """
    Returns the mutual info for three vars.
    Symmetric in X, Y, Z
    """
    ds = dataset.copy()
    return mutualinfo(iX, iY, dataset = ds) - condmutualinfo(iX, iY, iZ, dataset = ds)


if __name__ == "__main__":

    print "Entropies in data:"
    for i in xrange(8):
        print "index =",i , "    ", entropy(i)

    print "Entropies in db:"
    for i in xrange(8):
        print "index =",i , "    ", entropy(i, dataset = db)

    print "mutual info (with survival):"
    for i in xrange(8):
        print "index =", i, "    ", mutualinfo(0, i)
        #print infogain(i)

    print "mutual info for db:"
    for i in xrange(8):
        print "index =", i, "    ", mutualinfo(0, i, dataset = db)
        #print infogain(i, dataset = db)

    print "conditional mutual info of fare and sur given class: "
    print condmutualinfo(0,6,1)

    print "mutual info of sur-sex:"
    print mutualinfo(0,2)
    print "conditional mutual info of sur-sex given i:"
    for i in [1,2,3,4,5,6,7]:
        print "conditioning index i =",i, "   ", condmutualinfo(0,2,i)

    xLRlin = logreg_prepdata(dba)[0]
    y = dba[::,0]
    m = np.size(y)
    y = y.reshape([m,1])
    dataLRlin = np.hstack([y,xLRlin])
    # data in the form :
    datacol = ["sur", "1", "age", "fare", "sex", "cl1", "cl2", "sibsp0", "sibsp1", "parch0", "parch1", "city0", "city1"]
    for i in xrange(13):
        print "index =", datacol[i], "     ",  "mutual info =", mutualinfo(0,i, dataset = dataLRlin)

    print "multi MI for 0,4,5"
    print multimutualinfo(0,4,5)


    intlist = [(i,j) for i in xrange(1,12) for j in xrange(1,12)]
    intlist = [x for x in intlist if x[0] <  x[1]]
    print intlist

    for (i,j) in intlist:
        xLRints = logreg_prepdata(dba, interactionlist=[(i,j)])[0]
        y = dba[::,0]
        m = np.size(y)
        y = y.reshape([m,1])
        dataLRfullints = np.hstack([y,xLRints])
        print "indices =", (i,j), "     ",  "mutual info =", mutualinfo(0,13, dataset = dataLRfullints)

    print "Time elapsed:", time() - start


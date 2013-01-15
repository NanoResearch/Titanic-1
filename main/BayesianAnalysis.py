__author__ = 'michaelbinger'

#
from time import time, clock
start = time()
import numpy as np
import csv
from numpy import *
from PrepareTitanicData import titandata, convertages
from DataFilters import df, dfrange, showstats, databin
from predictions import f3sm12pred, predicttrain, comparepreds

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

###################################################

def prod( iterable ):# returns the product of each element
    p = 1
    for x in iterable:
        p *= x
    return p

def pxy(value, index, y, dataset, laplacesmooth = "on"):
    """
    computes prob(x|y), i.e the probability of having feature x=(value,index) when the survival value is y (0 or 1)
    This is accomplished using frequencies in dataset to approximate p(x and y)/p(y)
    i.e using maximum likelihood estimation (MLE)
    Laplace smoothing has the effect of giving small nonzero values for features which don't appear
    in the training data. It has a pretty small effect overall
    """
    dataset = dataset.copy()
    # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
    J = [0,3,2,3,3,3,1,3] # number of discrete feature values for each index, assuming age is binned into
    # (0-10), (11-19), 20+ and sibsp
    J1 = 1
    if laplacesmooth == "off":
        J = [0,0,0,0,0,0,0,0]
        J1 = 0*J1
    if df([[y,0],[value,index]],dataset) == []:
        nxy = 0
    else:
        nxy = showstats(df([[y,0],[value,index]],dataset))[1]
    ny = showstats(df([[y,0]],dataset))[1]
    return float(J1 + nxy)/(J[index] + ny)

db = databin(data)

def bayespreddepend(x, dataset = db):
    db = dataset.copy()
    x0 = list(x) #list of features [[value1, index1],[value2,index2],...]
    x1 = list(x) #list of features [[value1, index1],[value2,index2],...]
    n = np.size(x,0)
    x0.append([0,0]) #all of the features x, along with die
    x1.append([1,0]) #all of the features x, along with live
    if (df(x0,dataset) == []) or (df(x1,dataset) == []):
        return 100
    nxand0 = showstats(df(x0,dataset))[1]
    nxand1 = showstats(df(x1,dataset))[1]
    return (1/ (( float(nxand0)/float(nxand1) ) + 1))

def bayespred(x, dataset = db):
    db = dataset.copy()
    x = list(x) #list of features [[value1, index1],[value2,index2],...]
    n = np.size(x,0)
    px0s = []
    px1s = []
    for a in xrange(n):
        px0s.append( pxy(x[a][0], x[a][1], 0, db, laplacesmooth = "on") ) #(x[a][0],x[a][1]) gives the value and # index of feature x[a]
        px1s.append( pxy(x[a][0], x[a][1], 1, db, laplacesmooth = "on") )
    pX0 = prod(px0s)
    pX1 = prod(px1s)
    p1 = showstats(db)[2]
    p0 = 1-p1
    #print pX0,pX1,p0,p1
    pYgivenX = pX1*p1/(pX1*p1+pX0*p0)
    return pYgivenX

def bayespreddata(dataset, indexignore = [], impose = False, depend = False, imposedepend = False):
    """
    must be binned data of 8 length each row
    """
    dataset = dataset.copy()
    pred = []
    indices = [1,2,3,4,5,7]
    for index in indexignore:
        indices.remove(index)
    for row in dataset:
        newrow = []
        for index in indices:
            newrow.append([row[index],index])
        if depend == True:
            bp = round(bayespreddepend(newrow),0)
        else:
            bp = round(bayespred(newrow),0)
        if impose and (row[2] == 1) and not(row[1] == 3):
            bp = 1
        if imposedepend == True:
            if (bp == 100) and (row[2] == 1) and not( (row[1] == 3) & (row[7] == 0) ):
                bp = 1
            elif (bp == 100) and (row[2] == 0) and (row[3] <= 10) and not(row[1] == 3):
                bp = 1
            elif bp ==100:
                bp = 0
        pred.append(bp)
    return np.array(pred)

if __name__ == "__main__":
    test8db = databin(test8)

    #bptrain = bayespreddata(db)
    #bptraindep = bayespreddata(db, depend = True, indexignore = [], imposedepend = True)
    #bptrainimp = bayespreddata(db, impose = True)
    #bptrain5 = bayespreddata(db,indexignore=[5])
    #bptrain45 = bayespreddata(db,indexignore=[4,5])

    bptestdep = bayespreddata(test8db, depend = True, indexignore = [], imposedepend = True)
    bptestind = bayespreddata(test8db)
    bptestindimp = bayespreddata(test8db, impose = True)

    #f3sm12train = f3sm12pred(db)
    f3sm12test = f3sm12pred(test8db)

    #print predicttrain(bptrain)
    #print predicttrain(bptraindep)
    #print predicttrain(bptrainimp)

    #print predicttrain(bptrain5)
    #print predicttrain(bptrain45)

    #comparepreds(bptrain, f3sm12train, dataset = db)
    #comparepreds(bptrainimp, f3sm12train, dataset = db)
    #comparepreds(bptraindep, f3sm12train, dataset = db)

    comparepreds(bptestdep, f3sm12test)
    comparepreds(bptestind, f3sm12test)
    comparepreds(bptestind, bptestdep)


    #comparepreds(bptrain, bptrain5, dataset = db)
    #comparepreds(bptrain5, bptrain45, dataset = db)




    totaltime = time() - start
    print "This code took %f s to run" %totaltime

    import sys
    sys.exit()


    for sex in xrange(2):
        for cl in xrange(1,4):
            feat = [[cl,1],[sex,2]]
            print "class=",cl, "and sex=",sex
            print bayespred(feat)
            print showstats(df(feat,data))

    xvals = [ [1,2,3], 1], [ [0,1], 2 ], [ [5,15,50], 3], [ [0,1,3], 4 ], [ [0,1,3], 5], [ [0,1,2], 7 ]

    for cl in [1,2,3]:
        for sex in [0,1]:
            for age in [5,15,50]:
                for sib in [0,1,3]:
                    for par in [0,1,3]:
                        for emb in [0,1,2]:
                            xval = [[cl,1], [sex,2], [age,3], [sib,4], [par,5], [emb,7]]
                            bp = bayespred(xval)
                            # there are 2*(3^5) = 486 different categories given the way we binned the data.
                            # We want to display only the results that disagree with the F3SM12 model
                            if (sex == 1) and (bp <= 0.5) and not( (cl == 3) and (emb == 0) ):
                                print xval, "yields:", bp
                            elif (sex == 1) and (bp >= 0.5) and (cl == 3) and (emb == 0) :
                                print xval, "yields:", bp
                            elif (sex == 0) and (bp <= 0.5) and (age == 5) and not(cl == 3) :
                                print xval, "yields:", bp
                            elif (sex == 0) and (bp >= 0.5) and not((age == 5) and not(cl == 3)) :
                                print xval, "yields:", bp

    print bayespred([[1,1],[0,2],[0,7],[3,4]])














    for sur in xrange(2):
        for sex in xrange(2):
            for cl in xrange(1,4):
                Ns = showstats(df([[sur,0]],data))[1]
                Nsc = showstats(df([[sur,0],[cl,1]],data))[1]
                Nss = showstats(df([[sur,0],[sex,2]],data))[1]
                Nscs = showstats(df([[sur,0],[cl,1],[sex,2]],data))[1]
                print "For sur=",sur, "class=", cl, "sex=", sex
                print "Is class ind of sex? "
                print "[LT,LB,Lp] = ", [Nscs,Nss,round(float(Nscs)/Nss,3)]
                print "[RT,RB,Rp] = ", [Nsc,Ns,round(float(Nsc)/Ns,3)]
                print "Is sex ind of class? "
                print "[LT,LB,Lp] = ", [Nscs,Nsc,round(float(Nscs)/Nsc,3)]
                print "[RT,RB,Rp] = ", [Nss,Ns,round(float(Nss)/Ns,3)]
                print "#"*50
    print "#"*100
    print "#"*100
    for sur in xrange(2):
        for sex in xrange(2):
            for emb in xrange(3):
                Ns = showstats(df([[sur,0]],data))[1]
                Nse = showstats(df([[sur,0],[emb,7]],data))[1]
                Nss = showstats(df([[sur,0],[sex,2]],data))[1]
                Nses =showstats(df([[sur,0],[emb,7],[sex,2]],data))[1]
                print "For sur=",sur, "city=", emb, "sex=", sex
                print "Is city ind of sex? "
                print "[LT,LB,Lp] = ", [Nses,Nss,round(float(Nses)/Nss,3)]
                print "[RT,RB,Rp] = ", [Nse,Ns,round(float(Nse)/Ns,3)]
                print "Is sex ind of city? "
                print "[LT,LB,Lp] = ", [Nses,Nse,round(float(Nses)/Nse,3)]
                print "[RT,RB,Rp] = ", [Nss,Ns,round(float(Nss)/Ns,3)]
                print "#"*50
    print "#"*100
    print "#"*100
    for sur in xrange(2):
        for cl in xrange(1,4):
            for emb in xrange(3):
                Ns = showstats(df([[sur,0]],data))[1]
                Nse = showstats(df([[sur,0],[emb,7]],data))[1]
                Nsc = showstats(df([[sur,0],[cl,1]],data))[1]
                Nsec =showstats(df([[sur,0],[emb,7],[cl,1]],data))[1]
                print "For sur=",sur, "class=", cl, "city=", emb
                print "Is city ind of class? "
                print "[LT,LB,Lp] = ", [Nsec,Nsc,round(float(Nsec)/Nsc,3)]
                print "[RT,RB,Rp] = ", [Nse,Ns,round(float(Nse)/Ns,3)]
                print "Is class ind of city? "
                print "[LT,LB,Lp] = ", [Nsec,Nse,round(float(Nsec)/Nse,3)]
                print "[RT,RB,Rp] = ", [Nsc,Ns,round(float(Nsc)/Ns,3)]
                print "#"*50





    totaltime = time() - start
    print "This code took %f s to run" %totaltime
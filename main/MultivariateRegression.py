

__author__ = 'michaelbinger'
from time import time, clock
starttime = time()

import numpy as np
from pylab import *
import random
import sympy
import sys
#from random import random, randrange
set_printoptions(suppress = True) # makes for nice printing without scientific notation

########################################################################################################################
########################################################################################################################
########################################################################################################################

def featurescale(Xmat):
    """
    Scales the x data (features) to be of somewhat uniform size. This greatly helps with convergence.
    X=(m,n+1) matrix, with m rows (each representing a training data case).
    The n column vectors, where n=number of features, will be rescaled.
    Note the first column is all 1's and should not be scaled.
    """
    X = Xmat.copy() #VERY important to copy the input matrix, or else you will change the array outside of this function.
    n = np.shape(X)[1]-1
    means = np.zeros(n)
    stds = np.zeros(n)
    for col in xrange(1,n+1):
        means[col-1] = np.mean(X[:,col])
        stds[col-1] = np.std(X[:,col])
        X[:,col] = (X[:,col] - means[col-1])/stds[col-1]
    return [X, means, stds]

def Jcost(X,y,theta,lam):
    """
    Cost function J for linear regression of one variable.
    X is (m,n+1) the feature matrix, where m = number of data examples and n = number of features
    y is the dependent variable we are trying to train on. y is a (m,1) column vector
    theta is a (n+1,1) column vector
    """
    X = X.copy()
    y = y.copy()
    theta = theta.copy()
    n = np.shape(X)[1]-1 #number of features
    regcon = np.ones([(n+1),1])
    regcon[0] = 0 # (n+1,1) dim array [0,1,1,...,1].T
    m = np.size(y) # number of features
    h = np.dot(X,theta) # hypothesis function is a (m,1) column vector
    sqErrors = (h - y) ** 2 #squared element-wise, still (m,1) vector
    regterm = lam*sum( (theta*regcon)**2 ) #regularization term
    J = (1.0 / (2 * m)) * (sum(sqErrors) + regterm)# sum up the sqErrors for each term
    return J

def gradientdescent(X,y,alpha,niter,lam):
    """
    Performs gradient descent algorithm for linear regression.
    X is (m,n+1) the feature matrix, where m = number of data examples and n = number of features
    y is the dependent variable we are trying to train on. y is a (m,1) column vector
    theta is a (n+1,1) column vector
    alpha is the learning rate.
    niter is the number of iterations.
    lam = lambda the regularization parameter
    """
    X = X.copy()
    y = y.copy()
    m = np.size(y) # number of training data examples
    n = np.shape(X)[1]-1 #number of features
    theta = np.zeros((n+1,1))
    if (n+1)*m != np.size(X):
        print "ERROR"
    Jsteps = np.zeros((niter,1))
    regcon = np.ones([(n+1),1])
    regcon[0] = 0 # (n+1,1) dim array [0,1,1,...,1].T
    for i in xrange(niter):
        h = np.dot(X,theta) #(m,1) column vector
        err_x = np.dot((h - y).T, X) #(1,n+1) row vector
        theta = theta - (float(alpha) / m) * err_x.T - lam*(float(alpha) / m)*theta*regcon #(n+1,1) column vector
        Jsteps[i, 0] = Jcost(X, y, theta, lam)
    #print theta.T
    return [theta.T, Jsteps.T]


def graddesexample(m,n,niter,alpha,lam,randomness,Jplot=0,datplot=0,extrahigh=0):
    """
    Creates an example of use of gradient descent with multiple variables.
    m = number of data training case examples
    n = number of features of the data
    niter = number of iterations for gradient descent
    alpha = learning rate for grad des
    randomness = governs the degree of randomness in the fake data set that is generated
    lam = lambda the regularization parameter
    Jplot = 1 means display the Jplot. No otherwise
    datplot = 1 mean display the data plot. No otherwise.
    extrahigh = if set to 1 then replace the last feature column with x[1]^4 and leave y linear
    """

    x = np.ones((m,n+1))
    for a in xrange(1,n+1):
        x[::,a] = np.array([random.randint(1,100) for i in xrange(m)])

    if extrahigh == 1:
        x[::,n] = x[::,1]**4 #let's throw an unnecessary term in to the hypothesis

    #print "This is X:"
    #print x
    fs = featurescale(x)
    xx = fs[0]
    means = fs[1]
    stds = fs[2]
    #print "This is X after feature scaling:"
    #print xx
    #print "means and stds:", means, stds

    y = np.zeros((m,1))
    for i in xrange(m):
        r = random.random()
        if extrahigh == 1:
            y[i] = sum(x[i,0:n]) + randomness*n*(r - 0.5)
        else:
            y[i] = sum(x[i,0:n+1]) + randomness*n*(r - 0.5)
            # Note the true distribution of y is y=1+x_1+x_2+...+x_n for n features. Randomness is added.

    #print "This is y:", y.T

    if datplot == 1: #make a 3-D plot only if n=2 features
        from mpl_toolkits.mplot3d import Axes3D
        fig = figure()
        ax = Axes3D(fig)
        ax.scatter(x[::,1],x[::,2],y)
        xlabel("x1")
        ylabel("x2")
        title("Random data")
        show()

    graddes = gradientdescent(xx,y,alpha,niter,lam)
    thetapred = graddes[0]
    Jsteps = graddes[1]
    #print "prediction for theta:", thetapred
    #print Jsteps

    if Jplot == 1:
        scatter(np.arange(niter)+1,Jsteps)
        xlabel("Number of iterations")
        ylabel("Jcost")
        title("The convergence of the cost function")
        show()

    X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = sympy.symbols('X1,X2,X3,X4,X5,X6,X7,X8,X9,X10')
    XN = [1,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10] #extend this manually if you want more than 10 features

    Xa = np.array([XN[a] for a in xrange(n+1)])
    #print Xa
    for i in xrange(1,n+1):
        Xa[i] = (Xa[i]-means[i-1])/stds[i-1]

    hyp = np.dot(thetapred,Xa)


    #print "final hypothesis function h_theta(x)"
    return hyp


def graddesexpower(m,n,niter,alpha,lam,randomness,p=1):
    """
    Creates an example of use of gradient descent with multiple variables.
    m = number of data training case examples
    n = number of features of the data
    niter = number of iterations for gradient descent
    alpha = learning rate for grad des
    randomness = governs the degree of randomness in the fake data set that is generated
    lam = lambda the regularization parameter
    Jplot = 1 means display the Jplot. No otherwise
    datplot = 1 mean display the data plot. No otherwise.
    extrahigh = if set to 1 then replace the last feature column with x[1]^4 and leave y linear
    """
    x = np.ones((m,p*n+1))
    for a in xrange(1,n+1):
        x[::,a] = np.array([random.randint(1,100) for i in xrange(m)])
        for b in xrange(1,p):
            x[::,b*n+a] = x[::,a]**(b+1)


    fs = featurescale(x)
    xx = fs[0]
    means = fs[1]
    stds = fs[2]

    y = np.zeros((m,1))
    for i in xrange(m):
        r = random.random()
        y[i] = sum(x[i,0:n+1]) + randomness*n*(r - 0.5)
        # Note the true distribution of y is y=1+x_1+x_2+...+x_n for n features. Randomness is added.

    graddes = gradientdescent(xx,y,alpha,niter,lam)
    thetapred = graddes[0]
    Jsteps = graddes[1]

    X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20 \
    = sympy.symbols('X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20')
    XN = [1,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20]
    #extend this manually if you want more than 20 features

    Xa = np.array([XN[a] for a in xrange(p*n+1)])

    for i in xrange(1,p*n+1):
        Xa[i] = (Xa[i]-means[i-1])/stds[i-1]

    hyp = np.dot(thetapred,Xa)

    return hyp




########################################################################################################################
########################################################################################################################
########################################################################################################################

print graddesexample(1000,3,1000,0.1,0,10)

print graddesexpower(1000,3,1000,0.1,0,10,p=2)
print graddesexpower(1000,3,1000,0.1,1,10,p=2)
print graddesexpower(1000,3,1000,0.1,10,10,p=2)

print "Time elapsed:", time() - starttime
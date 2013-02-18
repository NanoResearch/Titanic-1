__author__ = 'michaelbinger'

from time import time, clock
starttime = time()

import numpy as np
from pylab import *
import random
import sympy
import scikits.statsmodels.api as sm
import sys
#from random import random, randrange
set_printoptions(suppress = True) # makes for nice printing without scientific notation

start = time()

########################################################################################################################
########################################################################################################################
########################################################################################################################

# Summary of general regression algorithm

# 1. Import Data
# 2. Convert strings to floats
# 3. Separate the data into "independent" or "predictor" variables X (m,n') and dependent variable Y (m,1)
#   Notation: label 'm' (number of data samples) indices by 'i' or 'j' and 'n'(number of features) indices by 'a' or 'b'
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

########################################################################################################################
########################################################################################################################
########################################################################################################################

def normeqn(X,y):
    """
    returns the predicted values of parameters theta.
    X has shape (m,n+1)
    y has shape (m,1)
    """
    return np.dot( np.linalg.inv( np.dot(X.T,X) ), np.dot(X.T, y) ).T

def featurescale(Xmat):
    """
    Scales the x data (features) to be of somewhat uniform size. This greatly helps with convergence.
    Xmat=(m,n+1) array, with m rows (each representing a training data case).
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

def glog(z):
    """
    The logistic, or sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def Jcost(X,y,theta,lam, logreg = False):
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
    m = np.size(y) # number of training examples
    y = y.reshape(m,1) # ensures proper shape of y (i.e. that it is not accidentally shape (m,))
    if logreg:
        h = glog(np.dot(X,theta)) # hypothesis function is a (m,1) column vector
        costi = y*np.log(h) + (1-y)*np.log(1-h)
        J = -(1.0 / float(m)) * sum(costi)
    else:
        h = np.dot(X,theta) # hypothesis function is a (m,1) column vector
        costi = 0.5 * (h - y) ** 2 #squared element-wise, still (m,1) vector.
            # These are the squared errors for each of the m training data (i=0..m-1)
        regterm = 0.5*lam*sum( (theta*regcon)**2 ) #regularization term
        J = (1.0 / float(m)) * (sum(costi) + regterm) # sum up the squared errors and regularization terms
    return J

def gradientdescent(X,y,alpha,niter,lam, logreg = False):
    """
    Performs gradient descent algorithm for multivariate regression.
    X is (m,n+1) the feature matrix, where m = number of data examples and n = number of features
    y is the dependent variable we are trying to train on. y is a (m,1) column vector
    theta is a (n+1,1) column vector
    alpha is the learning rate.
    niter is the number of iterations.
    lam = lambda the regularization parameter
    logreg = True will perform gradient descent with a logistic cost function
    """
    X = X.copy()
    y = y.copy()
    m = np.size(y) # number of training data examples
    y = y.reshape(m,1) # ensures proper shape of y (i.e. that it is not accidentally shape (m,))
    n = np.shape(X)[1]-1 #number of features
    theta = np.zeros((n+1,1))
    if (n+1)*m != np.size(X):
        print "ERROR"
    Jsteps = np.zeros(niter)
    regcon = np.ones([(n+1),1])
    regcon[0] = 0 # (n+1,1) dim array [0,1,1,...,1].T
    for i in xrange(niter):
        if logreg:
            h = glog(np.dot(X,theta)) # hypothesis function is a (m,1) column vector
        else:
            h = np.dot(X,theta) # hypothesis function is a (m,1) column vector
        delJdeltheta = (1.0 / float(m)) * np.dot( X.T , h - y ) #(n+1,1) column vector.
            # These are the partial derivatives of J wrt thetas. Note that this take the same form (with different h)
            # for multivariate linear regression and for logistic regression.
        theta = theta - float(alpha) * delJdeltheta  - lam*(float(alpha) / m)*theta*regcon #(n+1,1) column vector
        Jsteps[i] = Jcost(X, y, theta, lam, logreg = logreg)
    return [theta.T, Jsteps]

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
    datplot = 1 means display the data plot. No otherwise.
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

    print "OLS package reults:"
    olstest = sm.OLS(y,x).fit()
    #olstest.summary()
    #print olstest
    print olstest.params
    print olstest.bse
    print olstest.summary()

    print "My normal eqn results:"
    print normeqn(x,y)

    print "My multivariate regression with gradient descent results "

    if Jplot == 1:
        scatter(np.arange(niter)+1,Jsteps)
        xlabel("Number of iterations")
        ylabel("Jcost")
        title("The convergence of the cost function")
        show()

    X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = sympy.symbols('X1,X2,X3,X4,X5,X6,X7,X8,X9,X10')
    XN = [1,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]
    #Creates algebraic symbols to represent final prediction.
    # extend this manually if you want more than 10 features

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
    #Creates algebraic symbols to represent final prediction.
    # extend this manually if you want more than 10 features

    Xa = np.array([XN[a] for a in xrange(p*n+1)])

    for i in xrange(1,p*n+1):
        Xa[i] = (Xa[i]-means[i-1])/stds[i-1]

    hyp = np.dot(thetapred,Xa)

    return hyp

########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == "__main__":

    m = 100000
    n = 10
    randomness = 1

    x = np.ones((m,n+1))
    for a in xrange(1,n+1):
        x[::,a] = np.array([random.randint(1,100) for i in xrange(m)])

    #print "This is X:"
    #print x

    y = np.zeros((m,1))
    for i in xrange(m):
        r = random.random()
        y[i] = sum(x[i,0:n+1]) + randomness*n*(r - 0.5)
            # Note the true distribution of y is y=1+x_1+x_2+...+x_n for n features. Randomness is added.

    ne = normeqn(x,y)
    print ne[0:100]

    #print graddesexample(1000,3,1000,0.1,0,10)
    # compares my regression with grad des, the normal eqn solution, and the OLS package from scikits

    #print graddesexpower(1000,3,1000,0.1,0,10,p=2)
    #print graddesexpower(1000,3,1000,0.1,1,10,p=2)
    #print graddesexpower(1000,3,1000,0.1,10,10,p=2)

    print "Time elapsed:", time() - starttime




__author__ = 'michaelbinger'

import numpy as np
from random import random, randrange
from pylab import *
import matplotlib
#import random


m = 100 # number of training data examples

x = np.array([randint(1,100) for x in xrange(m)])#randint returns random integers from low (inclusive) to high (exclusive), xrange allows the for loop to iterate 100 times.  Ultimately this line fills up an array with random numbers between 1-100
y = np.zeros((m,1))#zeros returns a new array filled with zeros. the argument (m,1) structures the array to have M (100) rows and 1 column

for i in xrange(m):
    r = random() #random number btw 0 and 1
    y[i] = x[i] + 100*(r - 0.5) #The true distribution of y is simply y=x, but we add some randomness to simulate real data.  This for loop first grabs a number from the x array and then it adds an additional random number.  Double randomness I suppose.

scatter(x,y)#scatter is a scatter plot function.  It's from MatPlotLib.
xlabel("x")# simply labels the x axis of the plot
ylabel("y") # simply labels the x axis of the plot
title("Random data (from y=x straight line)")# simply labels the the plot purpose
show()#this prints out the plot. It's from matplotlib

xx = np.ones((m,2))#Return a new array of given shape and type, filled with ones
xx[:,1] = x # here, we have now substitute the array with all the random numbers--x--into the first column of the xx array.  I'm not sure why we didn't just use x
print "This is X"
print xx
print "This is y"
print y

theta = np.zeros((2,1))# this produces an array that is 2X1.  Why is it only 2 rows?  Is this just initializing?
print "This is theta"
print theta

niter = 100
alpha = 0.0001

def Jcost(X,y,theta):
    """
    cost function J for linear regression of one variable.  The idea is to minimize J as much as possible.  When J is minimized as much as possible, then the linear fit of the hypothesis is as good as it can be
    """
    m = np.size(y)# the size function returns the size of the list or array.  M is the size of your sample
    h = np.dot(X,theta) # hypothesis function.  the dot function multiplies X and theta to produce a product of the two arrays.  X is the independent variable and Theta is the slope.  Y-intercept is assumed to be zero
    sqErrors = (h - y) ** 2 #Here, we find the difference of between the predictive hypothesis (h) and the actual values of Y from training set.  We square them to find begin corresponding to the least-squares cost-minimizing function
    J = (1.0 / (2 * m)) * sqErrors.sum()#J is the sum of the squared errors divided by m--the sample size. So we are averaging the size of the squared errors.  1/2 is introduced as a constant for ease-of-math reasons
    return J

def gradientdescent(X,y,theta,alpha,niter):
    """
    performs gradient descent algorithm for linear regression
    """
    m = np.size(y) # # of features
    Jsteps = np.zeros((niter,1))
    for i in xrange(niter):
        h = np.dot(X,theta)
        err0 = np.dot((h-y).T, X[:, 0])
        err1 = np.dot((h-y).T, X[:, 1])
        theta[0] = theta[0] - (alpha / m) * err0.sum()
        theta[1] = theta[1] - (alpha / m) * err1.sum()

        Jsteps[i, 0] = Jcost(X, y, theta)
    return [theta,Jsteps.T]

graddes = gradientdescent(xx,y,theta,alpha,niter)
thetapred = graddes[0]
Jsteps = graddes[1]
print "theta=", thetapred
#Figure out how to write this as
# print "theta = %??" %theta
# where ?? is unknown. Or something simple like that.


#print Jsteps #uncomment to see the value of J at each step


scatter(x,y,c='r')
plot(x,x,c='b')
plot(x,thetapred[0]+thetapred[1]*x,c='g')
xlabel("x")
ylabel("y")
title("LR pred in green, 'true' distribution in blue")
show()

scatter(np.arange(niter)+1,Jsteps)
xlabel("Number of iterations")
ylabel("Jcost")
title("The convergence of the cost function")
show()

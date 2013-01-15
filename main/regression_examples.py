
import numpy as np
from numpy import *
from random import randint
from pylab import *
import sympy
import scipy
from MultivariateRegression import Jcost, gradientdescent, featurescale
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


#NIST/ITL StRD
#Dataset Name:  Wampler2 (Wampler2.dat)
#Import the data which is from a txt file on columns with no delimiter
openfile = open( "../datasamples.txt", 'rb' )
data = []
for line in openfile.readlines():
    y = [value for value in line.split()]
    data.append( y )
openfile.close()
data = np.array(data)

#print data[0::,0]

# convert the strings to floats
nrow = np.size(data[0::,0]) #number of rows and columns in data array
ncol = np.size(data[0,0::])
#print nrow, ncol
data = data.ravel()
data = np.array([float(x) for x in data])
data = data.reshape(nrow,ncol)
print data

m = np.size(data[::,0]) #number of data samples

y = data[::,0]
y = y.reshape(m,1) # create the results vector y as a (m,1) array

x = data[::,1]
x = x.reshape(m,1) # create the feature array

scatter(x,y,c='r') # plot the original data
xlabel("x")
ylabel("y")
show()


#Based on the plot, we know we need higher powers of x...
xc = x.copy()
for p in xrange(2,6): #use the hypothesis of up to 5 powers of x
    xp = xc**p
    x = np.hstack([x,xp])

x1s = np.ones([m,1])
x = np.hstack([x1s,x])

print x # this is our (6,m) feature array

n = np.shape(x)[1]-1

# Now feature scale
fs = featurescale(x)
xfs = fs[0]
means = fs[1]
stds = fs[2]

#Run grad des
alpha = 0.1
niter = 1000
lam = 0

graddes = gradientdescent(xfs,y,alpha,niter,lam)
thetapred = graddes[0]
Jsteps = graddes[1]
print "prediction for theta:", thetapred
#print Jsteps

scatter(np.arange(niter)+1,Jsteps)
xlabel("Number of iterations")
ylabel("Jcost")
title("The convergence of the cost function")
show()

X,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = sympy.symbols('X,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10')
XN = [1,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]
#Creates algebraic symbols to represent final prediction.
# extend this manually if you want more than 10 features

Xa = np.array([XN[a] for a in xrange(n+1)])
#print Xa
for i in xrange(1,n+1):
    Xa[i] = (Xa[i]-means[i-1])/stds[i-1]

X1 = X
X2 = X**2
X3 = X**3
X4 = X**4
X5 = X**5

# Normal eqn prediction
timenorm =time()
thetanorm = normeqn(xfs,y)
print "time to calculate normal eqn =", time()-timenorm


print np.shape(thetapred)
print np.shape(thetanorm)

hyp = np.dot(thetapred,Xa)
hypnorm = np.dot(thetanorm,Xa)

#print "final hypothesis function h_theta(x)"
print hyp
print hypnorm

print "Time elapsed:", time() - start


import sys
sys.exit()

import random

timemat10 = time()
mat = []
for x in xrange(100):
    mat.append(random.random())
mat = np.array(mat)
mat = mat.reshape([10,10])
mati = np.linalg.inv(mat)
#print mat
#print mati
time10 = time()-timemat10
print "time to invert 10x10 matrix = ", time10

timemat100 = time()
mat = []
for x in xrange(10000):
    mat.append(random.random())
mat = np.array(mat)
mat = mat.reshape([100,100])
mati = np.linalg.inv(mat)
#print mat
#print mati
time100 = time()-timemat100
print "time to invert 100x100 matrix = ", time100


timemat1000 = time()
mat = []
for x in xrange(1000000):
    mat.append(random.random())
mat = np.array(mat)
mat = mat.reshape([1000,1000])
mati = np.linalg.inv(mat)
#print mat
#print mati
time1000 = time()-timemat1000
print "time to invert 1000x1000 matrix = ", time1000

print time1000/time100
print time100/time10


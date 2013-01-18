
import numpy as np
from numpy import *
from random import randint
from pylab import *
import sympy
import scipy
from MultivariateRegression import Jcost, gradientdescent, featurescale, glog
from time import time

start = time()
set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)

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
#   Jcost() and gradientdescent(). h_theta(X) = f(theta.T*X) where f(z) = z for most continuous regression problems.
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


# http://www.umass.edu/statdata/statdata/stat-logistic.html
# Coronary Heart Disease (CHD) (0=none, 1=present) data by age
# Import the data which is from a txt file on columns with no delimiter

openfile = open( "../CHD_logreg_example_data.txt", 'rb' )
data = []
for line in openfile.readlines():
    y = [value for value in line.split()]
    data.append( y )
openfile.close()
data = np.array(data)

# convert the strings to floats
nrow = np.size(data[0::,0]) #number of rows and columns in data array
ncol = np.size(data[0,0::])
data = data.ravel()
data = np.array([float(x) for x in data])
data = data.reshape(nrow,ncol)
print data

# The following one line command does all of the above!!!
data2 = np.loadtxt("../CHD_logreg_example_data.txt")
print data2
print "Are the two data arrays the same?: ", data == data2

m = np.size(data[::,0]) #number of data samples

y = data[::,2]
y = y.reshape(m,1) # create the results vector y as a (m,1) array

x = data[::,1]
x = x.reshape(m,1) # create the feature array

#scatter(x,y,c='r') # plot the original data
#xlabel("x")
#ylabel("y")
#show()

# Add column of ones and feature scale.
xx = np.hstack([ np.ones([m,1]) , x ])
fs = featurescale(xx)
xfs = fs[0]
means = fs[1]
stds = fs[2]
print means, stds

#example calculation of cost for a guessed value of theta = [-1,0.05]:
# This is just to convince yourself that the cost function is defined properly for log reg
#theta = (np.array([-1,0.05])).reshape(2,1)
#print np.shape(theta)
#h = glog(np.dot(xx,theta))
#print "h", np.shape(h)
#print "log(h)", np.shape(np.log(h))
#costi = y*np.log(h) + (1-y)*np.log(1-h)
#J = -(1/float(m)) * sum(costi)
#print J

alpha = 0.1
niter = 10000
lam = 0

# Perform gradient descent
graddes = gradientdescent(xfs, y, alpha, niter, lam, logreg = True)
thetapred = graddes[0]
Jsteps = graddes[1]
print "prediction for theta:", thetapred

# Verify convergence
scatter(np.arange(niter)+1,Jsteps)
xlabel("Number of iterations")
ylabel("Jcost")
title("The convergence of the cost function")
show()

# Write hypothesis
X1 = sympy.symbols('X1')
X1 = (X1 - means[0])/stds[0]
XN = np.array([1,X1])
hypz = np.dot(thetapred,XN) # the hypothesis is hyp = glog(hypz)
print "hypothesis for z (take glog(z) for actual hypothesis prob): ", hypz

# Plot the hypothesis CHD probability
ages = (np.arange(20,81,1)).reshape(61,1)

agesfs = (ages-means[0])/stds[0]
na = np.size(agesfs)
X0 = np.ones([na,1])
Xa = np.hstack([X0, agesfs])
#print Xa

chd = glog( np.dot( Xa, thetapred.T) )

plot1 = plot(ages,chd, 'b')
xlim(20,80)
ylim(-0.1,1.3)
xlabel("Age")
ylabel("Probability of CHD")
title("Hypothesis probability of CHD by age")
plot2 = plot(x,y,'ro') # plot the original data
#legend([plot1], ("blue line"), "best", numpoints = 1)
show()


print "Time elapsed:", time() - start


__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

from time import time, clock
starttime = time()
import numpy as np
from numpy import *
from DataFilters import df, dfrange, showstats
from PrepareTitanicData import titandata, convertages

set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)

data=titandata("train") #(891,8) array
testdata=titandata("test") #(418,7) array
test8 = titandata("test8") #(418,8) array


# Call function titandata which takes an argument string, which must be either "train", "test", or "test8"
#print data[0:10]
#print testdata[0:10]
#print test8[0:10]
# Note that data and test8 are regularized and floated into an array of 8 columns:
# [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# The survival values are 0=die, 1=live, 2=don't know (for test8)
# Note that testdata is regularized and floated into an array of 7 columns
# [class, sex, age, sibsp, parch, fare, embarked]

totdata = vstack((data,test8)) # stacks data on top of test8 to create a (1309,8) array


# Now let's actually convert the unknown ages to our best guesses.
# These derive from analysis below which were performed BEFORE doing this conversion
# (i.e. only on the data where the age is given). To repeat and confirm this analysis you can simply
# comment out the data conversion below

data = convertages(data,3)
test8 = convertages(test8,3)
testdata = convertages(testdata,2)
totdata = convertages(totdata,3)


indict8 = { 0 : 'sur', 1 : 'class', 2 : 'sex', 3 : 'age', 4 : 'sibsp', 5 : 'parch', 6 : 'fare' , 7 : 'city' }

constraints = []
print "Recall indices [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]"
while True:
    query = raw_input("Input a feature constraint (in form 'min max index')(type 'x' to quit inputting constraints):")
    if query == "x":
        break
    query = [float(x) for x in query.split()] # split breaks the 3 string inputs up, and they are then floated
    fmin = query[0]
    fmax = query[1]
    index = query[2]
    print "Great! We'll apply the constraint: %i <= %s <= %i" %(fmin, indict8[index], fmax)
    constraints.append(query)
print "To summarize, you said constrain data by:", constraints

ncon = np.size(constraints)/3
tempdata = data #using the training data set.
#tempdata = test8 # test8 can be used to look at passenger attributes. But note unknown survival value = 2.
for x in xrange(ncon):
    fmin = constraints[x][0]
    fmax = constraints[x][1]
    index = constraints[x][2]
    tempdata = dfrange(fmin,fmax,index,tempdata)

print tempdata
print showstats(tempdata)




# This file stores these file writing methods which should be copied and pasted into other files and ran there.
# I had these in predictions.py, but it greatly slows down running any code that calls the functions
# in predictions.py. Thus I moved them here.

data7 = scipy.delete(data,6,1) #delete fare column
test7 = scipy.delete(test8,6,1)
RFC7 = randomforests(100,100,data7,test7[0::,1::])
rfccsv = csv.writer(open('../RFC7pred.csv','wb'))
for x in xrange(418):
    if RFC7[x]==0:
        rfccsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if RFC7[x]==1:
        rfccsv.writerow(["1"]) # We only need the predictions, not the other passenger data.



newcsv = csv.writer(open('../newpredpy.csv','wb'))
newpredict = newpred(test8)
for x in xrange(418):
    if newpredict[x]==0:
        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if newpredict[x]==1:
        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.



newcsv = csv.writer(open('../bayespredind.csv','wb'))
newpredict = bptestind # bptestind = bayespreddata(test8db)
for x in xrange(418):
    if newpredict[x]==0:
        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if newpredict[x]==1:
        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.

newcsv = csv.writer(open('../bayespreddep.csv','wb'))
newpredict = bptestdep # bptestdep = bayespreddata(test8db, depend = True, indexignore = [], imposedepend = True)
for x in xrange(418):
    if newpredict[x]==0:
        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if newpredict[x]==1:
        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.



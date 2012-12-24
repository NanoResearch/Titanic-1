__author__ = 'michaelbinger'

import csv as csv
import numpy as np

traincsv = csv.reader(open("../train.csv", 'rb'))

header = traincsv.next()

data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)

Npass = np.size(data[0::,0].astype(np.float))
Nsur = np.sum(data[0::,0].astype(np.float))
persur = Nsur / Npass

Wstats = data[0::,3] == "female"
Mstats = data[0::,3] != "female"

Won = data[Wstats,0].astype(np.float)
Mon = data[Mstats,0].astype(np.float)

perWsur = np.sum(Won) / np.size(Won)
perMsur = np.sum(Mon) / np.size(Mon)



print Wstats[0:10]
print Mstats[0:10]
print data[1]
print data[2]
print Won

print persur
print perWsur
print perMsur

print data[1]
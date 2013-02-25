__author__ = 'Andrew'

import csv as csv
import numpy as np
#import scipy
#from numpy import *


#create the data array from any csv file
def read_csv_into_array (csv_file):
    file_reader = csv.reader(open("../"+csv_file))#append the ../ to find the file in the current direcctory
    file_reader.next()#skips the header row
    file_list = []#defines a list that we'll read the file into
    for row in file_reader:
        file_list.append(row)
    file_array = np.array(file_list)
    return file_array

#train_array = read_csv_into_array("train.csv")
#print train_array[0]


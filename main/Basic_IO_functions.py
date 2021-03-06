__author__ = 'Andrew'

import csv as csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from numpy import *


#create the data array from any csv file
def read_csv_into_array (csv_file):
    file_reader = csv.reader(open("../"+csv_file))#append the ../ to find the file in the current directory
    file_reader.next()#skips the header row
    file_list = []#defines a list that we'll read the file into
    for row in file_reader:
        file_list.append(row)
    file_array = np.array(file_list)
    return file_array

#output a .csv file from either a list or an array
def convert_into_csv (array_to_file, output_name):
    array_writer = csv.writer(open("../main/"+output_name+".csv", "wb"))
    #list_with_rows = array_to_file.tolist()#will throw an error if "array_to_file" is already a list
    for row in array_to_file:
        array_writer.writerow(row)
    return

def plot_histogram_of_comparison(array_to_plot):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # the histogram of the data
    ax.hist(array_to_plot, facecolor='green')

    plt.show()
    return

#train_array = read_csv_into_array("train.csv")
#print train_array[0]
#convert_into_csv(train_array, "test_csv_file")

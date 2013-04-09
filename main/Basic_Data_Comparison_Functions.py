__author__ = 'Andrew'

import csv as csv
import numpy as np
import scipy
import Basic_IO_functions as io
from numpy import *

def compare_columns_in_array(array_to_be_compared, anchor_column_index, anchor_value, compared_to_column_index, compared_to_column_value):

    #setting default values if parameters are not specified coming in
    if anchor_column_index == "":# make the anchor the first column if not specified otherwise
        anchor_column_index = 0 #need better error handling later because if the anchor is not 0 then the compared columns will be also be messed up
    if compared_to_column_index == "":
        compared_to_column_index == 1
    if anchor_value == "":
        anchor_value = 1
        print "anchor value was received as null and reset to 1"
    if compared_to_column_value == "":
        compared_to_column_value = 0
        print "compared_to_value was received as null and reset to 0"


    #figure out the number of columns and rows
    array_shape = np.shape(array_to_be_compared)# return the dimensions of the array (# of columns, # of rows)
    number_of_columns = array_shape[1]
    number_of_rows = array_shape[0]
    #assert isinstance(number_of_rows, object)

    #repositories for various comparisons
    anchor_column_array = ([])
    compared_column_array = ([])
    complete_stats_array = ([])
    column_stats_array = ([])
    compared_column_index = 0


    #comparison stats...assumes a hypothesis
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    null_comparison = 0

    i = 0
    ii = 0
    while i < (number_of_columns-1):
        #take anchor column and next column
        anchor_column_array = array_to_be_compared[:, anchor_column_index] # single column array with the anchor values

        #checks to make sure anchor isn't comparing itself if we later make the function robust enough to iterate through columns
        if compared_column_index == anchor_column_index:
            compared_column_index = compared_column_index + 1

        compared_column_array = array_to_be_compared[:, compared_column_index]
        print "compared_column_index"
        print compared_column_index

        #Reset Storage Values
        ii = 0
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        null_comparison = 0
        #Reset Storage Values

        while ii < number_of_rows:
            #print "in second while"
            #print "anchor column value"
            #print anchor_column_array[ii]

            if (anchor_column_array[ii] == "1") and (compared_column_array[ii] == "female"):
                true_positive = true_positive + 1
            elif (anchor_column_array[ii] == "1") and (compared_column_array[ii] == "male"):
                false_positive = false_positive + 1
            elif (anchor_column_array[ii] == "0") and (compared_column_array[ii] == "female"):
                false_negative = false_negative + 1
            elif (anchor_column_array[ii] == "0") and (compared_column_array[ii] == "male"):
                true_negative = true_negative + 1
            else:
                null_comparison = null_comparison + 1

            ii = ii + 1 # iterate for the nested while loop

        column_stats_array = (("true_positive",true_positive), ("false_positive",false_positive), ("true_negative",true_negative), ("false_negative",false_negative), ("null_comparison",null_comparison))

        complete_stats_array.append(column_stats_array)
        i = i+1 # iterate for the outside while loop
        compared_column_index = compared_column_index + 1
        #print "big loop iterations"
        #print i
        #print column_stats_array
    return complete_stats_array

train_array = io.read_csv_into_array("train.csv")
print np.array(compare_columns_in_array(train_array, 0))

#print train_array[0]
#convert_into_csv(train_array, "test_csv_file")
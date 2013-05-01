__author__ = 'Andrew'

import csv as csv
import numpy as np
#import scipy
import Basic_IO_functions as io
#from numpy import *

def compare_columns_in_array(array_to_be_compared, anchor_column_index=0, anchor_column_value="", compared_to_column_index=1, compared_to_column_value="", true_positive_list=[0,1]):

    #figure out the number of columns and rows
    array_shape = np.shape(array_to_be_compared)# return the dimensions of the array (# of columns, # of rows)
    number_of_columns = array_shape[1]
    number_of_rows = array_shape[0]

    #repositories for various comparisons
    anchor_column_array = array_to_be_compared[:, anchor_column_index] # single column array with the anchor values
    if compared_to_column_index == anchor_column_index:
        compared_to_column_index = compared_to_column_index + 1
    compared_column_array = array_to_be_compared[:, compared_to_column_index]
    complete_stats_array = ([])
    column_stats_array = ([])
    pair_count_array = ([])
    all_counts_array = ([])

    #extract comparison values from anchor and comparison arrays
    anchor_column_values_list = extract_comparison_values_from_array(anchor_column_array,0,0,"clco",0)
    comparison_column_values_list = extract_comparison_values_from_array(compared_column_array,0,0,"clco",30)

    i = 0
    ii = 0
    iii = 0
    while i < anchor_column_values_list.__len__():
        anchor_column_value = anchor_column_values_list[i][0]#extract value from the anchor value list

        #iterate through the comparison value list and compare the anchor value to each value in the comparison value list
        #reset ii for next comparison value
        ii = 0
        while ii < (comparison_column_values_list.__len__()):
            compared_to_column_value = comparison_column_values_list[ii][0]

            iii = 0
            pair_count = 0
            while iii < (number_of_rows-1):
                if ( (anchor_column_array[iii]== anchor_column_value) and (compared_column_array[iii]==compared_to_column_value)):
                    #need to store the count only
                    pair_count = pair_count+1
                iii = iii + 1
                #need to store the pair of anchor and comparison and their count
            pair_count_array.append([anchor_column_value, compared_to_column_value, pair_count])
            ii = ii + 1

            #possible to limit this loop by filtering out values that have less than 30 occurrences (< than what a t test requires)
        #After each anchor value pairing, extract the pair_count_array lists, put their real values into an array, and erase pair_count_array
        all_counts_array.append(["column "+compared_to_column_index.__str__(),pair_count_array])
        pair_count_array = ([])

        #repeat for next value on the anchor list
        i = i + 1 # iterate for the nested while loop

    return all_counts_array

def extract_comparison_values_from_array(array_to_be_compared, column_index=0, row_index =0, output_type="clco", minimum_number_of_elements = 30):#need to document function

    #set default values if array_index is blank
    original_row_index = row_index

    array_shape = np.shape(array_to_be_compared)# return the dimensions of the array (# of columns, # of rows)
    number_of_rows = array_shape[0]
    if array_shape.__len__()> 1:
        number_of_columns = array_shape[1]
    else:
        number_of_columns = 1

    i = 0
    list_of_comparison_values = []
    extracted_value =""

    while i < (number_of_rows-1):

        if number_of_columns == 1:
            extracted_value = array_to_be_compared[row_index]
        else:
            extracted_value = array_to_be_compared[row_index, column_index]

        if list_of_comparison_values.__sizeof__() == 0:
            list_of_comparison_values.append(extracted_value)
        elif extracted_value not in list_of_comparison_values:
            list_of_comparison_values.append(extracted_value)

        row_index = row_index + 1
        i = i + 1
    #END i WHILE

    #Initialize or ReInitialize the values needed to count and sort the extracted values
    ii = 0
    iii = 0
    row_index = 0
    extracted_value_count = 0
    array_of_comparison_values_with_counts = ([])
    #END INITIALIZATION
    while ii <(len(list_of_comparison_values)):
        row_index = original_row_index
        iii = 0
        extracted_value_count = 0
        while iii<(number_of_rows - 1):#number of rows is not the right iterator in case the row index is not zero
            if array_to_be_compared[row_index] == list_of_comparison_values[ii]:
                extracted_value_count = extracted_value_count + 1

            row_index = row_index + 1
            iii= iii + 1
        #END iii WHILE
        if (extracted_value_count >= minimum_number_of_elements):
            array_of_comparison_values_with_counts.append([list_of_comparison_values[ii],extracted_value_count])

        ii = ii + 1
    #END ii WHILE

    #place IFs so that comparison, comparison with count, or ordered comparison can be returned
    #could move all this up into the function so only the necessary parts execute.
    if output_type == "cl":
        return list_of_comparison_values
    elif output_type == "clc":
        return array_of_comparison_values_with_counts
    else:
        iiii = 0
        size = 0
        comparison_list_a = []
        comparison_list_b = []
        array_of_comparison_values_with_counts_shape = np.shape(array_of_comparison_values_with_counts)# return the dimensions of the array (# of columns, # of rows)
        if array_shape.__len__()> 1:
            number_of_columns = array_of_comparison_values_with_counts_shape[1]
        number_of_rows = array_of_comparison_values_with_counts_shape[0]
        while iiii < (number_of_rows-1):
            comparison_list_a = array_of_comparison_values_with_counts[iiii]
            comparison_list_b = array_of_comparison_values_with_counts[iiii+1]
            if comparison_list_b[1] > comparison_list_a[1]:
                  array_of_comparison_values_with_counts[iiii]=comparison_list_b
                  array_of_comparison_values_with_counts[iiii+1]=comparison_list_a
            iiii = iiii + 1
        return array_of_comparison_values_with_counts # really, ordered

def compare_all_columns_in_array(array_to_be_compared, anchor_column_index=0, anchor_column_value="", compare_column_index_list="", true_positive_list=[0,1]):
    array_shape = np.shape(array_to_be_compared)# return the dimensions of the array (# of columns, # of rows)
    number_of_columns = array_shape[1]
    number_of_rows = array_shape[0]

    all_columns_compared_array = ([])
    i = 0

    #compare_column_index_list
    #limit to values who have counts over 30
    while i < (number_of_columns-1):
        if i != anchor_column_index:
            all_columns_compared_array.append(compare_columns_in_array(array_to_be_compared, anchor_column_index, anchor_column_value, i))
        i = i+1

    return all_columns_compared_array
#scripting lines for test purposes
train_array = io.read_csv_into_array("train.csv")
io.convert_into_csv(compare_all_columns_in_array(train_array, 0), "comparison values")


# print extract_comparison_values_from_array(train_array[:,1])#how do I do default values?
#sample_array= ([34,65,32,76,2])
#io.plot_histogram_of_comparison(sample_array)
#io.plot_histogram_of_comparison(compare_columns_in_array(train_array))

#print np.array(compare_columns_in_array(train_array, 0))

#print train_array[0]
#convert_into_csv(train_array, "test_csv_file")
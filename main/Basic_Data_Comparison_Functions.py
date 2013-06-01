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
    pair_count_dictionary = {}

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
        while ii < (comparison_column_values_list.__len__()): #find each unique comparison value in the column.  the iii loop then counts how many pairs there are with an anchor value
            compared_to_column_value = comparison_column_values_list[ii][0]

            iii = 0
            pair_count = 0
            while iii < (number_of_rows-1): #iterate through each row in the anchor and compared column to find pairs
                if ( (anchor_column_array[iii]== anchor_column_value) and (compared_column_array[iii]==compared_to_column_value)):
                    #need to store the count only
                    pair_count = pair_count+1
                iii = iii + 1
                #need to store the pair of anchor and comparison and their count
            #consider using a dictionary to pull out counts by unique pair key
            if pair_count >=30:
                pair_count_array.append(["Clm"+compared_to_column_index.__str__(), anchor_column_value, compared_to_column_value, pair_count])
                #pair_count_dictionary["Clm"+compared_to_column_index.__str__()+"Acr:"+anchor_column_value + "/Cpr:" + compared_to_column_value]= [anchor_column_value, compared_to_column_value, pair_count]

            ii = ii + 1
            #end while ii loop

        #repeat for next value on the anchor list
        i = i + 1 # iterate for the nested while loop

    #print pair_count_dictionary
    return pair_count_array
    #return pair_count_dictionary
    #return all_counts_array

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
    #temp_array = ([])
    i = 0

    #compare_column_index_list
    #limit to values who have counts over 30 possibly.  proxy for statistical significance
    #need to clean up outputs to not be a bunch of lists
    while i < (number_of_columns-1):
        if i != anchor_column_index:
            #temp_array = compare_columns_in_array(array_to_be_compared, anchor_column_index, anchor_column_value, i)
            all_columns_compared_array.append(compare_columns_in_array(array_to_be_compared, anchor_column_index, anchor_column_value, i))
            #all_columns_compared_array.append("")
        i = i+1
    #print "The shape of all_columns_compared_array is... "+np.shape(all_columns_compared_array).__str__()
    return all_columns_compared_array

def find_high_value_comparison_pairs (array_to_be_compared):#use the dictionary or array to find values that are highly correlated with the anchor values (1 or 0)
    #find entries where the column is the same, the anchor value differs, but the comparison value is the same
    #EAch list is composed of [ClmX, anchor_column_value, compared_to_column_value, pair_count].  One list per pair match,
    original_comparison_pairs_array = compare_all_columns_in_array(array_to_be_compared)
    row_list = []
    zero_list = []
    one_list = []
    rules_array = ([])
    #rules_array_ordered = ([])
    rule_prediction_factor = 0 #equals prediction_ratio (accuracy) * comparitor_count (reliability)
    prediction_ratio = 0 #numerator is the number of outcome-comparitor pairs for outcome that is predicted more over the number of outcome-comparitor pair that is predicted less.  The greater the difference, the more the comparitor predicts the numerator outcome
    comparitor_count = 0 #the count of rows that have the comparitor in it regardless of whether it lines up against 1 or 0
    comparitor_percentage = 0 #the comparitor count over the overall sample size
    comparitor_outcome_percentage = 0
    overall_sample_size = 892 #total number of records

    i=0
    ii=0
    iii=0
    iiii=0
    while i < original_comparison_pairs_array.__len__():
        row_list = original_comparison_pairs_array[i]

        if row_list.__len__() > 0:
            ii = 0
            while ii < row_list.__len__():

                    if (row_list[ii][1] == '0'):
                        zero_list = row_list[ii]

                        iii = 0
                        while iii < row_list.__len__():
                            #!!!!zero_list is probably somewhat messed up.  It might need an iii index!!!  SEEE PRINT OUTS ABOVE
                            #print " in  iii while " + iii.__str__()
                            if row_list[iii][1] == "1" and row_list[iii][2] == zero_list[2]: # find the comparison value
                                if row_list[iii][3] > zero_list[3]:
                                    rule_prediction_factor = ((float(row_list[iii][3])/float(zero_list[3]))*(row_list[iii][3]+zero_list[3]))
                                    rules_array.append([row_list[iii][1], row_list[iii][2], row_list[iii][2]+" in column "+row_list[iii][0] +" has a prediction factor of "+rule_prediction_factor.__str__() + " for outcome flag"+ row_list[iii][1], rule_prediction_factor, "prediction ratio is "+(float(row_list[iii][3])/float(zero_list[3])).__str__()])
                                else:
                                    rule_prediction_factor = (float(zero_list[3])/float(row_list[iii][3])) * (zero_list[3]+row_list[iii][3])
                                    rules_array.append([zero_list[1],zero_list[2], zero_list[2].__str__()+" in column "+zero_list[0] + " has a prediction factor of "+rule_prediction_factor.__str__()+" for outcome flag "+zero_list[1], rule_prediction_factor, "prediction ratio is "+(float(zero_list[3])/float(row_list[iii][3])).__str__()])
                                #write out that it's predictive of the correct anchor + the higher count over the lower count

                            iii = iii + 1
                    ii = ii + 1

        i = i+1
        #take the one that has the higher count and put its count as numerator and the lower count as denominator
        #the result gives a ratio or score which shows how differently each comparison value predicts the anchor
        #using a function flag, choose only pairs where the ratio is above some number (e.g. above 1.5)
        #append these into an array and return it
        #this should show which values have high leverage.
        #later this could be done across an anchor and multiple comparison values.
    rules_array_list_a = []
    rules_array_list_b = []
    swapped = True
    #print rules_array
    while (swapped):
        swapped = False
        while iiii < (rules_array.__len__()-1):
            rules_array_list_a = rules_array[iiii]
            rules_array_list_b = rules_array[iiii+1]
            if rules_array_list_b[3] > rules_array_list_a[3]:
                print "Switch took place "+iiii.__str__()
                rules_array[iiii]= rules_array_list_b
                rules_array[iiii+1] = rules_array_list_a
                swapped = True
            print swapped
            iiii = iiii + 1
        if swapped == True:
            iiii = 0
    print "rules_array"
    print rules_array
    return rules_array

#scripting lines for test purposes
train_array = io.read_csv_into_array("train.B.csv")
io.convert_into_csv(find_high_value_comparison_pairs(train_array), "prioritized rules")
#io.convert_into_csv(compare_all_columns_in_array(train_array, 0), "comparison values")


# print extract_comparison_values_from_array(train_array[:,1])#how do I do default values?
#sample_array= ([34,65,32,76,2])
#io.plot_histogram_of_comparison(sample_array)
#io.plot_histogram_of_comparison(compare_columns_in_array(train_array))

#print np.array(compare_columns_in_array(train_array, 0))

#print train_array[0]
#convert_into_csv(train_array, "test_csv_file")
Titanic
=======

This is our collaborative project for the kaggle titanic contest.

gendermodel.py : legacy file illustrating the basics
predict.py : legacy file manually constructing survival tables and data analysis
randforest.py : legacy file with a first pass at random forests (without sibsp and parch)

Help_Explore_Explain.py : various help stuff


PrepareTitanicData.py : regularizes and floats the data. call function titandata() to get data, test8, or testdata
UPDATE: can now call original data in string format with "originaldata" argument.

predictions.py: contains several functions that create survival predictions on a given data set, compares
predictions to each other, and tests them on the training data. These will be called by higherlevel.py
and/or other files.
UPDATE: The function randomforests() will run many random forests and yield the average prediction for each passenger.

higherlevel.py: The main file for doing analysis work and playing with the data. The functions df() and dfrange()
are particularly powerful for filtering the data. showstats() presents the results for a data set in a nice format.
convertages() converts unknown ages to useful averages. Numerous cuts through the data are shown, t
o try to gain insight. Random forests are implemented. Several predictions are compared to each other.

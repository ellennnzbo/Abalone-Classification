import numpy as np
import pandas as pd
import csv

def import_for_classification(
        ifname, input_cols=None, target_col=None, classes=None):
    """
    Imports the abalone data-set, processes the sex and target columns, and generates exploratory plots

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    returns
    -------
    inputs -- the data as a numpy.array object
    targets -- the targets as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    classes -- ordered list of classes
    """
    # if no file name is provided then use synthetic data
    dataframe = pd.read_csv(ifname, header=None)
    dataframe.columns = ['sex', "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight",
                         "shell_weight", "rings"]
    # print("dataframe.columns = %r" % (dataframe.columns,))

    # process the data
    dataframe = process_data(dataframe)
    N = dataframe.shape[0]

    # if no target name is supplied we assume it is the last colunmn in the
    # data file
    if target_col is None:
        target_col = dataframe.columns[-1]
        potential_inputs = dataframe.columns[:-1]
    else:
        potential_inputs = list(dataframe.columns)
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    print("input_cols = %r" % (input_cols,))
    print("target_col = %r" % (target_col))
    if classes is None:
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
        classes = class_values.unique()
    else:
        # construct a 1d array of the rows to keep
        to_keep = np.zeros(N,dtype=bool)
        for class_name in classes:
            to_keep |= (dataframe[target_col] == class_name)
        # now keep only these rows
        dataframe = dataframe[to_keep]
        # there are a different number of dat items now
        N = dataframe.shape[0]
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
    print("classes = %r" % (classes,))
    inputs = dataframe[input_cols].as_matrix()
    targets = dataframe[target_col].as_matrix()
    return inputs, targets, input_cols, classes

def process_data(dataframe):
    # one hot encoding
    dataframe = pd.concat([pd.get_dummies(dataframe['sex'], prefix='sex'), dataframe], axis=1)
    dataframe.drop(['sex'], axis=1, inplace=True)
    # target variable encoding
    target_col = dataframe.columns[-1]
    ring_threshold = 10
    dataframe[target_col] = np.where(dataframe[target_col] > ring_threshold, 1, 0)
    return dataframe
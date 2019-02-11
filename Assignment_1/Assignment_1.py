# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:49:33 2019

@author: XXZ180012
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
#from sklearn import tree, metrics

#reader = csv.reader(open(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv', "r"))
#
#variables = reader

binary = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv')

def entropy(data, member):
    entropy = 0
    values = data[member].unique()
    for value in values:
        fraction =data[member].value_counts()[value]/len(data[member])
        entropy += -(fraction * np.log2(fraction))
    return entropy
    

def information_Gain(data, attribute):
    data_1 = data[data[attribute] == 1]
    data_0 = data[data[attribute] == 0]
    print(data_1)
    print(data_0)
    
    
    
    
    


#def test_split(index, value, dataset):
#	left, right = list(), list()
#	for row in dataset:
#		if row[index] < value:
#			left.append(row)
#		else:
#			right.append(row)
#	return left, right

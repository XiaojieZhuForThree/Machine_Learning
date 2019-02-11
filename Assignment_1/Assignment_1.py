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

class decisionTree:
    def __init__():
        

        
        
        
        

def entropy(data):
    entropy = 0
    values = data["Class"].unique()
    for value in values:
        fraction =data["Class"].value_counts()[value]/len(data["Class"])
        entropy += -(fraction * np.log2(fraction))
    return entropy
    

def information_Gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    for value in values:
        data_value = data[data[attribute] == value]
        total_entropy -= len(data_value)*entropy(data_value)/len(data)
    return total_entropy


def buildNode(data, attribute):
    subdata_1 = data[data[attribute] == 1]
    subdata_0 = data[data[attribute] == 0]
    return subdata_1, subdata_0

#    print(data_1)
#    print(data_0)
    
    
#for attribute in list(binary):
#    if attribute != "Class":
#        print("the information gain for " + str(attribute) + "=", information_Gain(binary, attribute))        
    
    


#def test_split(index, value, dataset):
#	left, right = list(), list()
#	for row in dataset:
#		if row[index] < value:
#			left.append(row)
#		else:
#			right.append(row)
#	return left, right

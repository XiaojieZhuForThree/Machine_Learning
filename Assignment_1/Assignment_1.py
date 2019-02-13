# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:49:33 2019

@author: XXZ180012
"""
#import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import csv
#from sklearn import tree, metrics

#reader = csv.reader(open(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv', "r"))
#
#variables = reader

binary = pd.read_csv(r'C:\Users\zxj\Desktop\Assignment_1\training_set.csv')

#class decisionTree:
#    def __init__(self, data):
#        self.data = data
#        self.spliter = None
#        
#    def buildTree():
#        maxGain = 0;
#        for attribute in set(data):
#            if information_Gain(data, attribute) > maxGain:
#                maxGain = information_Gain(data, attribute)
#                spliter = attribute
#        root = spliter
#        root.right, root.left = buildNode(data, spliter)
#        
#        
#        
#        

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

def findSplit(data, used):
    check = set(data) - used
    maxGain = None
    maxVal = 0
    for attribute in check:
        iG = information_Gain(data, attribute)
        if iG > maxVal:
            maxVal = iG
            maxGain = attribute
    return maxGain


def buildNode(data, attribute):
    subdata_1 = data[data[attribute] == 1]
    subdata_0 = data[data[attribute] == 0]
    return subdata_1, subdata_0


def buildTree(data, used, n):
    if len(data["Class"].unique()) == 1:
        print(data["Class"].unique()[0], end = '')
        return
    else: 
        rootVal = findSplit(data, used)
        if rootVal is not None:
            used.add(rootVal)
            leftSet, rightSet = buildNode(data, rootVal)
            print("\n" + "| " * n + rootVal +  " = 1: ", end = '')
            buildTree(leftSet, set(used), n+1)
            print("\n" + "| " * n + rootVal + " = 0: ", end = '')
            buildTree(rightSet, set(used), n+1)
            return
        return

initialSet = set(["Class"])   
buildTree(binary, initialSet, 0)
#binary[((binary["XJ"] == 0) & (binary["XO"] == 0) & (binary["XF"] == 0) & (binary["XQ"] == 0)
#         & (binary["XU"] == 0) & (binary["XI"] == 0))]
#    print(data_1)
#    print(data_0)
    
    
#for attribute in list(binary)[:-1]:
#    print("the information gain for " + str(attribute) + "=", information_Gain(binary, attribute))        
    
    


#def test_split(index, value, dataset):
#	left, right = list(), list()
#	for row in dataset:
#		if row[index] < value:
#			left.append(row)
#		else:
#			right.append(row)
#	return left, right

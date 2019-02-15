# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:49:33 2019

@author: XXZ180012
"""
#import os
import random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import csv
#from sklearn import tree, metrics

#reader = csv.reader(open(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv', "r"))
#
#variables = reader

binary = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets1\training_set.csv')

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

def varImpurity(data):
    part1 = 0
    part2 = 0
    values = data["Class"].unique()
    if 0 in values :
        part1 = data["Class"].value_counts()[0]/len(data)
    if 1 in values:
        part2 = data["Class"].value_counts()[1]/len(data)
    return part1 * part2

def impurityGain(data, attribute):
    VI = varImpurity(data)
    values = data[attribute].unique()
    for value in values:
        data_value = data[data[attribute] == value]
        VI -= len(data_value)*varImpurity(data_value)/len(data)
    return VI

def findSplit2(data, used):
    check = set(data) - used
    maxGain = None
    maxVal = 0
    for attribute in check:
        iG = impurityGain(data, attribute)
        if iG > maxVal:
            maxVal = iG
            maxGain = attribute
    return maxGain

def buildTree_1(data, used, n):
    if len(data["Class"].unique()) == 1:
        print(data["Class"].unique()[0], end = '')
        return
    else: 
        rootVal = findSplit(data, used)
        if rootVal is not None:
            used.add(rootVal)
            leftSet, rightSet = buildNode(data, rootVal)
            print("\n" + "| " * n + rootVal +  " = 1: ", end = '')
            buildTree_1(leftSet, set(used), n+1)
            print("\n" + "| " * n + rootVal + " = 0: ", end = '')
            buildTree_1(rightSet, set(used), n+1)
            return
        return
    
def buildTree_2(data, used, n):
    if len(data["Class"].unique()) == 1:
        print(data["Class"].unique()[0], end = '')
        return
    else: 
        rootVal = findSplit2(data, used)
        if rootVal is not None:
            used.add(rootVal)
            leftSet, rightSet = buildNode(data, rootVal)
            print("\n" + "| " * n + rootVal +  " = 1: ", end = '')
            buildTree_2(leftSet, set(used), n+1)
            print("\n" + "| " * n + rootVal + " = 0: ", end = '')
            buildTree_2(rightSet, set(used), n+1)
            return
        return

def post_Pruning(decisionTree, L, K):
    bestOne = decisionTree
    for i in range(1, L + 1):
        newTree = decisionTree
        M = random.randint(1, K+1)
        for j in range(1, M+1):
            N = nonLeafNode(newTree)
            P = random.randint(1, len(N)+1)
            theOne = N[P-1]
            if count(theOne, 0) > count(theOne, 1):
                theOne = LeafNode(0)
            else:
                theOne = LeafNode(1)
        if (accuracy(newTree) > accuracy(bestOne)):
           bestOne = newTree
    return bestOne

initialSet1 = set(["Class"])   
initialSet2 = set(['Class'])
print("First Tree\n")
buildTree_1(binary, initialSet1, 0)
print("\nSecond Tree")
buildTree_2(binary, initialSet2, 0)
binary[((binary["XK"] == 1) & (binary["XD"] == 1) & (binary["XS"] == 1) & (binary["XP"] == 1))]





#    print(data_1)
#    print(data_0)
    
#    
#for attribute in list(binary)[:-1]:
#    print("the impurity gain for " + str(attribute) + "=", impurityGain(binary, attribute))        
    
    


#def test_split(index, value, dataset):
#	left, right = list(), list()
#	for row in dataset:
#		if row[index] < value:
#			left.append(row)
#		else:
#			right.append(row)
#	return left, right

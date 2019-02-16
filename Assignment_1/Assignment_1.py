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

trainSet1 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets1\training_set.csv')
trainSet2 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets2\training_set.csv')
validateSet1 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets1\validation_set.csv')
validateSet2 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets2\validation_set.csv')
#    print(row)

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.pos = None
        self.neg = None

    def __str__(self):
        printTree(self)
        return ""
    
    class LeafNode:
        def __init__(self, value):
            self.value = value

def entropy(data):
    entropy = 0
    values = data["Class"].unique()
    for value in values:
        fraction =data["Class"].value_counts()[value]/len(data["Class"])
        entropy += -(fraction * np.log2(fraction))
    return entropy

def varImpurity(data):
    part1 = 0
    part2 = 0
    values = data["Class"].unique()
    if 0 in values :
        part1 = data["Class"].value_counts()[0]/len(data)
    if 1 in values:
        part2 = data["Class"].value_counts()[1]/len(data)
    return part1 * part2

def impurity_Gain(data, attribute):
    VI = varImpurity(data)
    values = data[attribute].unique()
    for value in values:
        data_value = data[data[attribute] == value]
        VI -= len(data_value)*varImpurity(data_value)/len(data)
    return VI    

def information_Gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    for value in values:
        data_value = data[data[attribute] == value]
        total_entropy -= len(data_value)*entropy(data_value)/len(data)
    return total_entropy

def findSplit(data, used, method):
    check = set(data) - used
    if len(check) == 0:
        return None
    maxGain = None
    maxVal = 0
    for attribute in check:
        gain = method(data, attribute)
        if gain > maxVal:
            maxVal = gain
            maxGain = attribute
    return maxGain

def split(data, attribute):
    subdata_1 = data[data[attribute] == 1]
    subdata_0 = data[data[attribute] == 0]
    return subdata_1, subdata_0


def buildTree(data, method, used = set(["Class"])):
    if len(data["Class"].unique()) == 1:
        return TreeNode.LeafNode(data['Class'].unique()[0])
    else: 
        rootVal = findSplit(data, used, method)
        if rootVal is not None:
            used.add(rootVal)
            posSet, negSet = split(data, rootVal)
            root = TreeNode(rootVal)
            if (not posSet.empty):
                root.pos = buildTree(posSet, method, set(used))
            if (not negSet.empty):
                root.neg = buildTree(negSet, method, set(used))
            return root
        return None
    
def printTree(root, n = 0):
    if (type(root) == None):
        return
    elif (type(root) == TreeNode.LeafNode):
        print(root.value, end = '')
    elif (type(root) == TreeNode):
        if (root.pos != None):
            print("\n" + "| " * n + root.name +  " = 1 : ", end = '')
            printTree(root.pos, n+1)
        if (root.neg != None):
            print("\n" + "| " * n + root.name +  " = 0 : ", end = '')
            printTree(root.neg, n+1) 
      
def nonLeafNode(root):
    if (type(root) == TreeNode.LeafNode):
        return []
    elif (type(root) == None):
        return []
    elif (type(root) == TreeNode):
        ans = [root]
        if (root.pos != None):
            ans += nonLeafNode(root.pos)
        if (root.neg != None):    
            ans += nonLeafNode(root.neg)
        return ans

def count(data, root, val):
    return data[root].value_counts()[val]
    
def accuracy(root, data):
    ans = 0
    for row in data.iterrows():
        test = row[1]
        ans += testValue(root, test)
    return ans / len(data)

def testValue(root, row):
    if (type(root) == TreeNode.LeafNode):
        if (root.value == row['Class']):
            return 1
        else:
            return 0
    elif (type(root) is None):
        return 0
    elif (type(root) == TreeNode):
        if (row[root.name] == 1):
            return (testValue(root.pos, row))
        else:
            return (testValue(root.neg, row))
    else:
        return 0

root1 = buildTree(trainSet1, information_Gain)
root2 = buildTree(trainSet2, information_Gain)
root3 = buildTree(trainSet1, impurity_Gain)
root4 = buildTree(trainSet2, impurity_Gain)
#print(root1)
#print(root2)
#print(root3)
#print(root4)
#def findSplit2(data, used):
#    check = set(data) - used
#    maxGain = None
#    maxVal = 0
#    for attribute in check:
#        iG = impurityGain(data, attribute)
#        if iG > maxVal:
#            maxVal = iG
#            maxGain = attribute
#    return maxGain       
    
#def buildTree_1(data, used, n):
#    if len(data["Class"].unique()) == 1:
#        print(data["Class"].unique()[0], end = '')
#        return
#    else: 
#        rootVal = findSplit(data, used)
#        if rootVal is not None:
#            used.add(rootVal)
#            leftSet, rightSet = buildNode(data, rootVal)
#            print("\n" + "| " * n + rootVal +  " = 1: ", end = '')
#            buildTree_1(leftSet, set(used), n+1)
#            print("\n" + "| " * n + rootVal + " = 0: ", end = '')
#            buildTree_1(rightSet, set(used), n+1)
#            return
#        return
#    
#def buildTree_2(data, used, n):
#    if len(data["Class"].unique()) == 1:
#        print(data["Class"].unique()[0], end = '')
#        return
#    else: 
#        rootVal = findSplit2(data, used)
#        if rootVal is not None:
#            used.add(rootVal)
#            leftSet, rightSet = buildNode(data, rootVal)
#            print("\n" + "| " * n + rootVal +  " = 1: ", end = '')
#            buildTree_2(leftSet, set(used), n+1)
#            print("\n" + "| " * n + rootVal + " = 0: ", end = '')
#            buildTree_2(rightSet, set(used), n+1)
#            return
#        return



#initialSet1 = set(["Class"])   
#initialSet2 = set(['Class'])
#print("First Tree\n")
#buildTree_1(binary, initialSet1, 0)
#print("\nSecond Tree")
#buildTree_2(binary, initialSet2, 0)
#trainSet2[((trainSet2["XI"] == 1) & (binary["XK"] == 1) & (binary["XD"] == 0) & (binary["XT"] == 1))]


def post_Pruning(decisionTree, training_data, validate_data, L, K):
    bestOne = decisionTree
    for i in range(1, L + 1):
        newTree = decisionTree
        M = random.randint(1, K+1)
        for j in range(1, M+1):
            N = nonLeafNode(newTree)
            P = random.randint(1, len(N)+1)
            theOne = N[P-1]
            if count(training_data, theOne, 0) > count(training_data, theOne, 1):
                theOne = TreeNode.LeafNode(0)
            else:
                theOne = TreeNode.LeafNode(1)
        if (accuracy(newTree, validate_data) > accuracy(bestOne, validate_data)):
           bestOne = newTree
    return bestOne

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

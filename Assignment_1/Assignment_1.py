# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:49:33 2019
@author: XXZ180012
"""
#import os
import random
import numpy as np
import pandas as pd
import copy
##import matplotlib.pyplot as plt
##import csv
##from sklearn import tree, metrics
#
##reader = csv.reader(open(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv', "r"))
##
##variables = reader
#
#trainSet1 = pd.read_csv('data_sets1\\training_set.csv', encoding='latin1')
#trainSet2 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets2\training_set.csv')
#validateSet1 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets1\validation_set.csv')
#validateSet2 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets2\validation_set.csv')
#testSet1 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets1\test_set.csv')
#testSet2 = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\data_sets2\test_set.csv')
##    print(row)
#
#
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.pos = None
        self.neg = None
        self.parent = None
        self.posClass = 0
        self.negClass = 0

        
class LeafNode:
    def __init__(self, value):
        self.value = value
            
class Tree(TreeNode):
    def __init__(self, data, method):
        self.tree = buildTree(data, method, used = set(["Class"]))
        self.posClass = 0
        self.negClass = 0
    def __str__(self):
        printTree(self.tree)
        return ""
    
def entropy(data):
    entropy = 0
    values = data["Class"].unique()
    if len(values) == 1:
        return 0
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

def findSplit(data, method, used):
    check = set(data) - used
    maxVal = float(0)
    maxGain = None
    for attribute in check:
        gain = method(data, attribute)
        if gain >= maxVal:
            maxVal = gain
            maxGain = attribute
    return maxGain

def split(data, attribute):
    subdata_1 = data[data[attribute] == 1]
    subdata_0 = data[data[attribute] == 0]
    return subdata_1, subdata_0


def buildTree(data, method, used = set(["Class"])):
    if data.empty:
        return None
    if len(data["Class"].unique()) == 1:
        return LeafNode(data['Class'].unique()[0])
    else: 
        rootVal = findSplit(data, method, used)
        if rootVal is not None:
            used.add(rootVal)
            posSet, negSet = split(data, rootVal)
            root = TreeNode(rootVal)
            root.posClass = data["Class"].value_counts()[1]
            root.negClass = data["Class"].value_counts()[0]
            if (not posSet.empty and not negSet.empty):
                posChild = buildTree(posSet, method, set(used))
                negChild = buildTree(negSet, method, set(used))
                root.pos = posChild
                root.neg = negChild
                negChild.parent = root
                posChild.parent = root
            return root
        return None
    
def printTree(root, n = 0):
    if (type(root) == None):
        return
    elif (type(root) == LeafNode):
        print(root.value, end = '')
    elif (type(root) == TreeNode):
        if (root.pos != None):
            print("\n" + "| " * n + root.name +  " = 1 : ", end = '')
            printTree(root.pos, n+1)
        if (root.neg != None):
            print("\n" + "| " * n + root.name +  " = 0 : ", end = '')
            printTree(root.neg, n+1) 
      
def nonLeafNode(root):
    if (type(root) == LeafNode):
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
        ans += testValue(root.tree, test)
    return ans / len(data)

def testValue(root, row):
    if (type(root) == LeafNode):
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


def post_Pruning(decisionTree, validate_data, L, K):
    bestOne = decisionTree
    for i in range(1, L):
        newTree = copy.deepcopy(decisionTree)
        M = random.randint(1, K)
        for j in range(1, M + 1):            
            N = nonLeafNode(newTree.tree)
            if (len(N) > 1):
                P = random.randint(1, len(N)-1)
                theOne = N[P]
                replace = None
                if theOne.posClass < theOne.negClass:
                    replace = LeafNode(0)
                else:
                    replace = LeafNode(1)
                if (theOne.parent == None):
                    newTree = replace
                elif (theOne.parent.pos == theOne):
                    theOne.parent.pos = replace
                else:
                    theOne.parent.neg = replace
        if (accuracy(newTree, validate_data) > accuracy(bestOne, validate_data)):
           bestOne = newTree
    return bestOne


def runProgram():
    L = int(input("Please input the first integer used for post-pruning: "))
    K = int(input("Please input the second integer used for post-pruning: "))
    trainingSet = pd.read_csv(input("Please provide the address of the training set: "), encoding='latin1')
    testSet = pd.read_csv(input("Please provide the address of the test set: "), encoding='latin1')
    validateSet = pd.read_csv(input("Please provide the address of the validation set: "), encoding='latin1')
    toPrint = input("to print the trees {yes,no}: ")
    if toPrint == 'yes':
        toPrint = True
    else:
        toPrint = False
    print("\n\nBuilding the trees...")
    Tree_InfoGain = Tree(trainingSet, information_Gain)
    Tree_ImpuGain = Tree(trainingSet, impurity_Gain)
    print("The accuracy of the tree with information gain on the testSet is: " + str(accuracy(Tree_InfoGain, testSet) * 100) + '%')
    print("The accuracy of the tree with impurity gain on the testSet is: " + str(accuracy(Tree_ImpuGain, testSet) * 100) + '%')
    print("\nPruning the trees...")
    PrunedInfoTree = post_Pruning(Tree_InfoGain, validateSet, L, K)
    PrunedImpuTree = post_Pruning(Tree_ImpuGain, validateSet, L, K)
    print("The accuracy of the tree with information gain on the validation set is: " + str(accuracy(Tree_InfoGain, validateSet) * 100) + '%')
    print("The accuracy of the pruned tree with information gain on the validation set is: " + str(accuracy(PrunedInfoTree, validateSet) * 100) + '%')
    print("The accuracy of the tree with impurity gain on the validation set is: " + str(accuracy(Tree_ImpuGain, validateSet) * 100) + '%')
    print("The accuracy of the pruned tree with impurity gain on the validation set is: " + str(accuracy(PrunedImpuTree, validateSet) * 100) + '%')
    if toPrint == True:
        print("\nThe Structure of the tree with information gain is : ")
        print(Tree_InfoGain)

        print("\nThe Structure of the pruned tree with information gain is : ")
        print(PrunedInfoTree)

        print("\nThe Structure of the tree with impurity gain is : ")
        print(Tree_ImpuGain)

        print("\nThe Structure of the pruned tree with impurity gain is : ")
        print(PrunedImpuTree)

#tree1 = Tree(trainSet1, information_Gain)
#tree2 = Tree(trainSet2, information_Gain)
#tree3 = Tree(trainSet1, impurity_Gain)
#tree4 = Tree(trainSet2, impurity_Gain)
#
#print("\nthe accuracy of tree1 on validateSet1 is " + str(accuracy(tree1, validateSet1) * 100)+ "%")
#x = post_Pruning(tree1, validateSet1, 20, 30)
#print("after pruning, the accuracy of tree1 on validateSet1 is " + str(accuracy(x, validateSet1) * 100)+ "%")
#
#print("\nthe accuracy of tree2 on validateSet2 is " + str(accuracy(tree2, validateSet2) * 100)+ "%")
#x = post_Pruning(tree2, validateSet2, 20, 30)
#print("after pruning, the accuracy of tree2 on validateSet2 is " + str(accuracy(x, validateSet2)* 100)+ "%")
#
#print("\nthe accuracy of tree3 on validateSet1 is " + str(accuracy(tree3, validateSet1) * 100)+ "%")
#x = post_Pruning(tree3, validateSet1, 20, 30)
#print("after pruning, the accuracy of tree1 on validateSet1 is " + str(accuracy(x, validateSet1) * 100)+ "%")
#
#print("\nthe accuracy of tree4 on validateSet2 is " + str(accuracy(tree4, validateSet2) * 100)+ "%")
#x = post_Pruning(tree4, validateSet2, 20, 30)
#print("after pruning, the accuracy of tree4 on validateSet2 is " + str(accuracy(x, validateSet2) * 100)+ "%")

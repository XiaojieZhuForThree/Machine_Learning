# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:49:33 2019

@author: XXZ180012
"""
import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import csv
#from sklearn import tree, metrics

#reader = csv.reader(open(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv', "r"))
#for row in reader:
#    print(row)

binary = pd.read_csv(r'C:\Users\XXZ180012\Desktop\Assignment_1\training_set.csv')
        
    
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

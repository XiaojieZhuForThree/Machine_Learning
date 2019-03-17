# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:51:15 2019

@author: xxz180012
"""
import numpy as np
spam = {}
ham = {}
def trainMultinomialNB(spam, ham, classes):
    # what kind of values do you want to return?
    for word in spam:
        

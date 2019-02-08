# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:49:33 2019

@author: XXZ180012
"""

import csv

with open ("C:\Users\Desktop\\training_set.csv","r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        print(row)
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:51:15 2019

@author: xxz180012
"""
import numpy as np
import pandas as pd
import os

spam = {}
ham = {}

hamAddr = r'C:\Users\zxj\Desktop\test\ham' 
spamAddr = r'C:\Users\zxj\Desktop\test\spam'
root = r'C:\Users\zxj\Desktop\test'   

classes = {'ham', 'spam'}

def extractVocabulary(folder):
    vocab = set()
    files = os.listdir(folder)
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                vocab.add(word)
    return vocab
    
        
def countVocabulary(folder):
    files = os.listdir(folder)
    dic = {}
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                if word not in dic:
                    dic[word] = 1
                dic[word] += 1
    return dic
#    return len(files)

def countDocs(folder):
    files = os.listdir(folder)
    return len(files)

def countTokensOfTerm(text, v):
    if v in text:
        return text[v]
    return 0


def trainMultinomialNB(classes):
    prior = {}
    V = extractVocabulary(hamAddr).union(extractVocabulary(spamAddr))
    N = countDocs(hamAddr) + countDocs(spamAddr)
    condprob = {}
    for t in V:
        condprob[t] = {}
    for cls in classes:
        N_cls = countDocs(root + '/' + cls)
        prior[cls] = N_cls / N
        text = countVocabulary(root + '/' + cls)
        total = 0
        for i in V:
            total += countTokensOfTerm(text, i) + 1
        for t in V:
            Tct = countTokensOfTerm(text, t)
            condprob[t][cls] = (Tct + 1) / total
    return V, prior, condprob 

V, prior, condprob = trainMultinomialNB(classes)



extractVocabulary(hamAddr)
extractVocabulary(spamAddr)
    
    
    
    
    
     
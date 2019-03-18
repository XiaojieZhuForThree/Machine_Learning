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

hamAddr = 'C:\Users\zxj\Desktop\test\ham' 
spamAddr = 'C:\Users\zxj\Desktop\test\spam'
root = r'C:\Users\zxj\Desktop\test'   

classes = {'ham', 'spam'}

def extraVocabulary(folder):
    vocab = set()
    files = os.listdir(folder)
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                vocab.add(word)
    return vocab
    
        
def countVocabulary(folder, dic):
    files = os.listdir(folder)
    
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                if word not in dic:
                    dic[word] = 1
                dic[word] += 1
    return
#    return len(files)

def countDocs(folder):
    files = os.listdir(folder)
    return len(files)

def countTokensOfTerm(text, v):
    if v in text:
        return text[v]
    return 0

extraVocabulary(r'C:\Users\zxj\Desktop\test\ham', ham)       
extraVocabulary(r'C:\Users\zxj\Desktop\test\spam', spam) 

def trainMultinomialNB(classes, documents):
    prior = {}
    V = extraVocabulary(hamAddr)
    N = countDocs(hamAddr) + countDocs(spamAddr)
    condprob = {}
    for cls in classes:
        N_cls = countDocs(root + "'\'" + cls)
        prior[cls] = N_cls / N
        for t in V:
            condprob[t] = {}
            Tct = countTokensOfTerm(text, t)
            condprob[t][cls] = (Tct + 1) / sum(text[i] + 1 for i in text)
    return V, prior, condprob        
    
    
    
    
    
    
    

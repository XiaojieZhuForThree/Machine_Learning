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

trainHamAddr = r'C:\Users\zxj\Desktop\train\ham'
trainSpamAddr = r'C:\Users\zxj\Desktop\train\spam'

testHamAddr = r'C:\Users\zxj\Desktop\test\ham' 
testSpamAddr = r'C:\Users\zxj\Desktop\test\spam'

testRoot = r'C:\Users\zxj\Desktop\test'   
trainRoot = r'C:\Users\zxj\Desktop\train'


classes = ['ham', 'spam']

def extractVocabulary(folder):
    vocab = set()
    files = os.listdir(folder)
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                vocab.add(word)
    return vocab

def extractTokensFromDoc(V, doc):
    tokens = set()
    for word in doc:
        if word in V:
            tokens.add(word)
    return tokens


    
        
def countVocabulary(folder):
    files = os.listdir(folder)
    dic = {}
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                if word not in dic:
                    dic[word] = 1
                else:
                    dic[word] += 1
    return dic


def countDocs(folder):
    files = os.listdir(folder)
    return len(files)

def countTokensOfTerm(text, v):
    if v in text:
        return text[v]
    return 0


def trainMultinomialNB(classes, trainRoot):
    prior = {}
    V = extractVocabulary(trainHamAddr).union(extractVocabulary(trainSpamAddr))
    N = countDocs(trainHamAddr) + countDocs(trainSpamAddr)
    condprob = {}
    for t in V:
        condprob[t] = {}
    for cls in classes:
        N_cls = countDocs(trainRoot + '/' + cls)
        prior[cls] = N_cls / N
        text = countVocabulary(trainRoot + '/' + cls)
        total = 0
        for i in V:
            total += countTokensOfTerm(text, i) + 1
        for t in V:
            Tct = countTokensOfTerm(text, t)
            condprob[t][cls] = (Tct + 1) / total
    return V, prior, condprob 


def applyMultinomialNB(classes, V, prior, condprob, doc):
    W = extractTokensFromDoc(V, doc)
    scores = {}
    for cls in classes:
        scores[cls] = np.log(prior[cls])
        for t in W:
            scores[cls] += np.log(condprob[t][cls])
    ans = classes[0]
    score = scores[ans]
    for key in scores:
        if scores[key] > score:
            score = scores[key]
            ans = key
    return ans

def testFunction_MultinomialNB(folder):
    right = 0
    numOfFiles = countDocs(testHamAddr) + countDocs(testSpamAddr)
    V, prior, condprob = trainMultinomialNB(classes, trainRoot)
    for cls in classes:
        files = os.listdir(folder + '/' + cls)
        for file in files:
            with open(folder + '/' + cls + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
                doc = doc.read().lower().replace('\n', ' ').split(' ')
                ans = applyMultinomialNB(classes, V, prior, condprob, doc)
                if ans == cls:
                    right += 1
    return right / numOfFiles
    
testFunction_MultinomialNB(testRoot)



    

V, prior, condprob = trainMultinomialNB(classes, trainRoot)

extractVocabulary(trainHamAddr)
extractVocabulary(trainSpamAddr)
    
    
    
    
    
     

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:51:15 2019

@author: xxz180012
"""
import numpy as np
import sys
import os

spam = {}
ham = {}

trainHamAddr = r'C:\Users\zxj\Desktop\train\ham'
trainSpamAddr = r'C:\Users\zxj\Desktop\train\spam'

testHamAddr = r'C:\Users\zxj\Desktop\test\ham' 
testSpamAddr = r'C:\Users\zxj\Desktop\test\spam'

testRoot = r'C:\Users\zxj\Desktop\test'   
trainRoot = r'C:\Users\zxj\Desktop\train'

stopwords = ['a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 "can't",
 'cannot',
 'could',
 "couldn't",
 'did',
 "didn't",
 'do',
 'does',
 "doesn't",
 'doing',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 "hadn't",
 'has',
 "hasn't",
 'have',
 "haven't",
 'having',
 'he',
 "he'd",
 "he'll",
 "he's",
 'her',
 'here',
 "here's",
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 "how's",
 'i',
 "i'd",
 "i'll",
 "i'm",
 "i've",
 'if',
 'in',
 'into',
 'is',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 "let's",
 'me',
 'more',
 'most',
 "mustn't",
 'my',
 'myself',
 'no',
 'nor',
 'not',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'ought',
 'our',
 'ours\tourselves',
 'out',
 'over',
 'own',
 'same',
 "shan't",
 'she',
 "she'd",
 "she'll",
 "she's",
 'should',
 "shouldn't",
 'so',
 'some',
 'such',
 'than',
 'that',
 "that's",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 "there's",
 'these',
 'they',
 "they'd",
 "they'll",
 "they're",
 "they've",
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 'very',
 'was',
 "wasn't",
 'we',
 "we'd",
 "we'll",
 "we're",
 "we've",
 'were',
 "weren't",
 'what',
 "what's",
 'when',
 "when's",
 'where',
 "where's",
 'which',
 'while',
 'who',
 "who's",
 'whom',
 'why',
 "why's",
 'with',
 "won't",
 'would',
 "wouldn't",
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves']

stopV = set(stopwords)

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
    dictionary = {}
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                if word not in dictionary:
                    dictionary[word] = 1
                else:
                    dictionary[word] += 1
    return dictionary


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

def testFunction_MultinomialNB():
    right_before = 0
    right_after = 0
    numOfFiles = countDocs(testHamAddr) + countDocs(testSpamAddr)
    V, prior, condprob = trainMultinomialNB(classes, trainRoot)
    noStopV = V - stopV
    
    for cls in classes:
        if cls == 'ham':
            testRoot = testHamAddr
        else:
            testRoot = testSpamAddr
            
        files = os.listdir(testRoot)
        for file in files:
            with open(testRoot + '/' + file, 'r'\
                      , encoding='utf-8', errors='ignore') as doc:
                doc = doc.read().lower().replace('\n', ' ').split(' ')
                ans_before = applyMultinomialNB(classes, V, prior, condprob, doc)
                ans_after = applyMultinomialNB(classes, noStopV, prior, condprob, doc)
                if ans_before == cls:
                    right_before += 1
                if ans_after == cls:
                    right_after += 1
    print ("Accuracy before filtering stop words: "+ str(right_before * 100 / numOfFiles) + '%')
    print ("Accuracy after filtering stop words: "+ str(right_after * 100 / numOfFiles) + '%')

def main(trainHamAddr, trainSpamAddr, testHamAddr, testSpamAddr):
    
    right_before = 0
    right_after = 0
    numOfFiles = countDocs(testHamAddr) + countDocs(testSpamAddr)
    V, prior, condprob = trainMultinomialNB(classes, trainRoot)
    noStopV = V - stopV
    
    for cls in classes:
        if cls == 'ham':
            testRoot = testHamAddr
        else:
            testRoot = testSpamAddr
            
        files = os.listdir(testRoot)
        for file in files:
            with open(testRoot + '/' + file, 'r'\
                      , encoding='utf-8', errors='ignore') as doc:
                doc = doc.read().lower().replace('\n', ' ').split(' ')
                ans_before = applyMultinomialNB(classes, V, prior, condprob, doc)
                ans_after = applyMultinomialNB(classes, noStopV, prior, condprob, doc)
                if ans_before == cls:
                    right_before += 1
                if ans_after == cls:
                    right_after += 1
    print ("Accuracy before filtering stop words: "+ str(right_before * 100 / numOfFiles) + '%')
    print ("Accuracy after filtering stop words: "+ str(right_after * 100 / numOfFiles) + '%')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    
testFunction_MultinomialNB()

#V, prior, condprob = trainMultinomialNB(classes, trainRoot)
#
#extractVocabulary(trainHamAddr)
#extractVocabulary(trainSpamAddr)
    
    
    
    
    
     
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:56:33 2019

@author: zxj
"""
import os
import sys
import numpy as np

trainHamAddr = r'C:\Users\zxj\Desktop\train\ham'
trainSpamAddr = r'C:\Users\zxj\Desktop\train\spam'

testHamAddr = r'C:\Users\zxj\Desktop\test\ham' 
testSpamAddr = r'C:\Users\zxj\Desktop\test\spam'

testRoot = r'C:\Users\zxj\Desktop\test'   
trainRoot = r'C:\Users\zxj\Desktop\train'

classes = ['ham', 'spam']
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
word_weights_before = {'theta0' : 0}
word_weights_after = {'theta0' : 0}

trainFiles = {}
testFiles = {}

vocabDic = {}

Vocabulary = []
noStop_Vocabulary = []

learning_rate = 0.001
iteration = 1000
lam = 0

def addFile(holder, address, cls):
    files = os.listdir(address)
    for file in files:
        with open(address + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ')
        holder[temp] = cls

def extractVocabulary(folder):
    vocab = set()
    files = os.listdir(folder)
    for file in files:
        with open(folder + '/' + file, 'r', encoding='utf-8', errors='ignore') as doc:
            temp = doc.read().lower().replace('\n', ' ').split(' ')
            for word in temp:
                vocab.add(word)
    return vocab

def countVocabulary(file):
    dictionary = {}
    for word in file.split(' '):
        if word not in dictionary:
            dictionary[word] = 1
        else:
            dictionary[word] += 1
    return dictionary

def getCondProb(cls, word_weights, file):
    words = countVocabulary(file)
    if cls == classes[0]:
        sumHam = word_weights['theta0']        
        for i in words:
            if i in word_weights:
                sumHam += word_weights[i] * words[i]
        return 1 / (1 + np.exp(sumHam))

    else:
        sumSpam = word_weights['theta0']
        for i in words:
            if i in word_weights:
                sumSpam += word_weights[i] * words[i]
        return np.exp(sumSpam) / (1 + np.exp(sumSpam))

def weightTrain(trainFiles, word_weights, iteration, lam):
    for x in range(iteration):
        for weight in word_weights:
            gain = 0
            for file in trainFiles:

                dic = vocabDic[file]
                if trainFiles[file] == classes[1]:
                    y = 1
                else:
                    y = 0

                if weight in dic:
                    gain += dic[weight] * (y - getCondProb(classes[1], word_weights, trainFiles[file]))
            word_weights[weight] += ((learning_rate * gain) - (learning_rate * lam * word_weights[weight]))

def logisticRegression(file, word_weights):
    scoreHam = getCondProb(classes[0], word_weights, file)
    scoreSpam = getCondProb(classes[1], word_weights, file)
    if scoreSpam > scoreHam:
        return classes[1]
    return classes[0]

#def test():
#    addFile(trainFiles, trainHamAddr, classes[0])
#    addFile(trainFiles, trainSpamAddr, classes[1])
#    addFile(testFiles, testHamAddr, classes[0])
#    addFile(testFiles, testSpamAddr, classes[1])
#    
#    V = extractVocabulary(trainHamAddr).union(extractVocabulary(trainSpamAddr))
#    noStopV = V - stopV
#    
#    for i in V:
#        word_weights_before[i] = 0    
#    
#    for i in noStopV:
#        word_weights_after[i] = 0
#    
#    for file in trainFiles:
#        vocabDic[file] = countVocabulary(file)
#    
#    weightTrain(trainFiles, word_weights_before, 8, 0.3) 
#    weightTrain(trainFiles, word_weights_after, 8, 0.3)
#    
#    right_before = 0
#    right_after = 0
#    for i in testFiles:
#        if logisticRegression(i, word_weights_before) == testFiles[i]:
#            right_before += 1  
#            
#        if logisticRegression(i, word_weights_after) == testFiles[i]:
#            right_after += 1             
#
#    print ("Accuracy before filtering stop words: "+ str(right_before * 100 / len(testFiles)) + '%')
#    print ("Accuracy after filtering stop words: "+ str(right_after * 100 / len(testFiles)) + '%')
    
def main(trainHamAddr, trainSpamAddr, testHamAddr, testSpamAddr, lamInp):
    
    addFile(trainFiles, trainHamAddr, classes[0])
    addFile(trainFiles, trainSpamAddr, classes[1])
    addFile(testFiles, testHamAddr, classes[0])
    addFile(testFiles, testSpamAddr, classes[1])
    
    V = extractVocabulary(trainHamAddr).union(extractVocabulary(trainSpamAddr))
    noStopV = V - stopV
    
    for i in V:
        word_weights_before[i] = 0    
    
    for i in noStopV:
        word_weights_after[i] = 0
    
    for file in trainFiles:
        vocabDic[file] = countVocabulary(file)
    
    weightTrain(trainFiles, word_weights_before, 8, 0.3) 
    weightTrain(trainFiles, word_weights_after, 8, 0.3)
    
    right_before = 0
    right_after = 0
    for i in testFiles:
        if logisticRegression(i, word_weights_before) == testFiles[i]:
            right_before += 1  
            
        if logisticRegression(i, word_weights_after) == testFiles[i]:
            right_after += 1             

    print ("Accuracy before filtering stop words: "+ str(right_before * 100 / len(testFiles)) + '%')
    print ("Accuracy after filtering stop words: "+ str(right_after * 100 / len(testFiles)) + '%')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
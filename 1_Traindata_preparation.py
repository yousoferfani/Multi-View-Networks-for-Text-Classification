"""
Created on Jul2017
My Implementation of the paper: End-to-End Multi-View Networks for Text Classification
@author: Yousof Erfani
"""
import glove 
from glove import Glove
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
import sklearn
from nltk.corpus import stopwords
import os
import pdb
import re
# folder addresses
train_neg_Address = 'dataset/train_neg'
train_pos_Address = 'dataset/train_pos'
# Parameters
embedding_length = 50
n_examples = 1875

def read_file(addr):
    lines = []
    for file_name in os.listdir(addr):
        #print file_name
        textfile = open (addr+'/'+file_name,'r')
        lines.append(textfile.readlines())
        textfile.close()
    return lines   

train_neg_lines  = read_file(train_neg_Address)
train_pos_lines  = read_file(train_pos_Address)
# load gloave model
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    leng = len(f.readlines())
    f.close()
    f = open(gloveFile,'r')
    cc = 0
    for line in f:
        cc = cc+1
        if cc % int(leng/100)==0:
            print(str(int(100*cc/leng))+'%')
        
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
model = loadGloveModel ('glove.6B/' + 'glove.6B.50d.txt')

def find_sentence_length(text):
    sentence_lengths = []
    for line in text:        
        sentence_lengths.append( len(line[0].split()) )        
    return sentence_lengths 
    
def find_embeddings(text, median_length):
    embeddings = []
    global model
    word_counts = pd.DataFrame({'word_name':['the'],'count':[1]})
    stop_words = stopwords.words('english')
    embeddings = np.zeros((len(text),median_length,embedding_length ), dtype = 'float64')

    for line_no, line in enumerate(text):
        print(line_no)
        cnt=0        
        for word_no, word in enumerate(line[0].split()):
            if cnt < median_length: 
                word = re.sub(r"['.:;!]",'',word)
                word = re.sub(r"<br",'',word)
                word = re.sub(r"[-)()\/*\#$\%\&\+\=\?\[\]\@\_]",' ',word)
                word = re.sub(r"  ",' ',word)
                word = re.sub(r'[",>]',' ',word)
                word = re.sub(r'[0-9]','',word)
                if word.endswith('.') | word.endswith(',')|word.endswith('?')|word.endswith('!')|word.endswith(':')|word.endswith('"')|word.endswith(')'):
                    word = word[:-1]
                if word.endswith("'s")|word.endswith("!!")|word.endswith("??")|word.endswith("?!") :
                    word = word[:-2]
                if word.endswith("n't") :
                    word = word[:-3]    
                if word.endswith(".<br") :
                    word = word[:-4]        
                if word.startswith('/>'):
                    word = word[2:]
                if word.startswith('"')|word.startswith('('):
                    word = word[1:]   
                    
                try:
                    w_unicode = np.unicode(word)              
                except:
                    print ('----Non Unocode----', word)   
                    
                try:
                    for subword in word.split(): 
                        if subword not in set(stopwords.words('english')):# and subword.lower() in model.keys():                                      		    
                            embeddings[line_no,cnt,:] = model[subword.lower()]  
                            new_row = pd.DataFrame({'word_name':[subword.lower()],'count':[1]})
                            word_counts = word_counts.append(new_row)
                            cnt += 1                                       
                except:
                    
                    print('NOt found for ',word.lower())                   

    word_counts_grouped=word_counts.groupby(word_counts.word_name).sum() 
    return  embeddings, word_counts_grouped

# dealing with the vairable length sentence input
setnece_length_neg= find_sentence_length(train_neg_lines)
sentence_length_pos= find_sentence_length(train_pos_lines)
setnece_length_neg.extend(sentence_length_pos)
med_sentence_leng  = int(np.median (setnece_length_neg))
print('median_sentence_leng_train:', med_sentence_leng )

embedding_train_neg , wcount_neg = find_embeddings(train_neg_lines[:n_examples],med_sentence_leng)
wcount_neg = wcount_neg.reset_index()
print ("Finished Computing the embeddings for negative  documents")

embedding_train_pos, w_count_pos  = find_embeddings(train_pos_lines[:n_examples],med_sentence_leng)
print ("Finished Computing the embeddings for positive  documents")
# find one hot encoding for labels
labels = np.concatenate((np.ones(embedding_train_pos.shape[0]),np.zeros(embedding_train_neg.shape[0])))
labels = np.reshape (labels,(-1,1))
labels = np.concatenate((labels, 1-labels),axis =1)
train_data = np.concatenate((embedding_train_pos,embedding_train_neg  ),axis =0 )

random_indices = np.random.permutation(train_data.shape[0])
train_data = train_data[random_indices,:,:]
labels = labels [random_indices]

del model
del train_pos_lines
del train_neg_lines
np.save('train_data_50.npy', train_data)
np.save('labels_50.npy', labels)
print('Finished')





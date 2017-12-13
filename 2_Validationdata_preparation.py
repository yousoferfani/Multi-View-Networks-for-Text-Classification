
"""
Created on Nov 2017
My implementation of the paper :End-to-End Multi-View Networks for Text Classification

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
valid_neg_Address = 'dataset/valid_neg'
valid_pos_Address = 'dataset/valid_pos'
embedding_length = 50
n_examples = 500

def read_file(addr):
    lines = []
    for file_name in os.listdir(addr):
        #print file_name
        textfile = open (addr+'/'+file_name,'r')
        lines.append(textfile.readlines())
        textfile.close()
    return lines   

valid_neg_lines  = read_file(valid_neg_Address)
valid_pos_lines  = read_file(valid_pos_Address)

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
print('HEREEE1')
def find_sentence_length(text):
    sentence_lengths = []
    for line in text:        
        sentence_lengths.append( len(line[0].split()) )        
    return sentence_lengths 
    
def find_embeddings(text, median_length):
    embeddings = []
    global model
    word_counts = pd.DataFrame({'word_name':['the'],'count':[1]})
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
                        if subword not in set(stopwords.words('english')):                                      		    
                            embeddings[line_no,cnt,:] = model[subword.lower()]  
                            new_row = pd.DataFrame({'word_name':[subword.lower()],'count':[1]})
                            word_counts = word_counts.append(new_row)
                            cnt += 1                                       
                except:
                    print('NOt found for ',word.lower())                   
    word_counts_grouped=word_counts.groupby(word_counts.word_name).sum() 
    return  embeddings, word_counts_grouped

# dealing with the vairable length sentence input by finding 
setnece_length_neg= find_sentence_length(valid_neg_lines)
sentence_length_pos= find_sentence_length(valid_pos_lines)
setnece_length_neg.extend(sentence_length_pos)
med_sentence_leng_valid = int(np.median (setnece_length_neg))
# From the train dataset
med_sentence_leng_valid = 174
print('median_sentence_leng_valid:', med_sentence_leng_valid)
embedding_valid_neg , wcount_neg_valid = find_embeddings(valid_neg_lines[:n_examples],med_sentence_leng_valid)
wcount_neg = wcount_neg_valid.reset_index()
print ("Finished Computing the embeddings for negative  documents")
embedding_valid_pos, w_count_pos_valid  = find_embeddings(valid_pos_lines[:n_examples],med_sentence_leng_valid)
print ("Finished Computing the embeddings for positive  documents")
# One hot encoding for labels
valid_labels = np.concatenate((np.ones(embedding_valid_pos.shape[0]),np.zeros(embedding_valid_neg.shape[0])))
valid_labels = np.reshape (valid_labels,(-1,1))
valid_labels = np.concatenate((valid_labels, 1-valid_labels),axis =1)
valid_data = np.concatenate((embedding_valid_pos,embedding_valid_neg  ),axis =0 )
# Shuffle the data
random_indices = np.random.permutation(valid_data.shape[0])
valid__data = valid_data[random_indices,:,:]
valid_labels = valid_labels[random_indices]

del model
del valid_pos_lines
del valid_neg_lines
np.save('valid_data_50.npy', valid_data)
np.save('valid_labels_50.npy', valid_labels)
print('Finished')





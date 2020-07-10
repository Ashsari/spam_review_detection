# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:49:05 2020

@author: Ashraf
"""

#importing the libraries
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical

from random import seed
seed(9001)
np.random.seed(205)

def clean_str(string):
    """
    Tokenization/string cleaning for all reviews.
    #replaces all none A-Z , a_z and 0-9 with a space. punctuations like ., !, ? and special characters are gone. 
    Original taken from "https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py ", I have changed some parts
    """
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def cleaning_reviews(reviews):
    
    vocabulary = []
    numWords = []
    review_vocab = []
    cleaned_reviews = []
    for r in reviews:
        rev = clean_str(r)
        split_rev = rev.split(' ') 
        vocabulary.extend(split_rev)
        numWords.append(len(split_rev))
        review_vocab.append(split_rev)
        cleaned_reviews.append(rev)
    
    return vocabulary, numWords, review_vocab, cleaned_reviews

def words_frequency(cleaned_reviews, vocabulary):
    '''
        returns the TF_IDF values for the review vocab
    '''
    tf_idf = TfidfVectorizer()
    tf_reviews = tf_idf.fit_transform(cleaned_reviews)

    feature_names = tf_idf.get_feature_names()
    dense = tf_reviews.todense()
    denselist = dense.tolist()
    df_TF_IDF = pd.DataFrame(denselist, columns=feature_names)
    
    missed_tfidfW = list((set(vocabulary).difference(feature_names)))    
    return df_TF_IDF, missed_tfidfW


#getting the embedding matrix for all the vocabularies
def create_embedding_matrix(filepath, vocabulary, embedding_dim):
    '''
    getting the embedding matrix for all the vocabularies
    filepath is the path to the glove pretrained word embedding dictionary 

    '''
    embedding_matrix = np.zeros((len(vocabulary), embedding_dim))
    with open(filepath , encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in vocabulary:
                idx = vocabulary.index(word)
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


def not_found_words(embedding_matrix):
    '''      
        returns nonzero_elements which are covered word vectors in the dictionary 
        returns not_foundWords and not_foundIndex of the vocabulary that are not found in the dictionary 
    '''
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print('The percentage of the covered words in the pretrained dictionary: ', nonzero_elements / len(vocabulary) )# 0.936484798000833
    not_foundIndex = np.where(~embedding_matrix.any(axis=1))[0]
    print('number of not found words is: ', len(not_foundIndex))
    not_foundWords= []
    for i in not_foundIndex:
        not_foundWords.append(vocabulary[i])
        
    return  nonzero_elements, not_foundWords, not_foundIndex


#importing the dataset
dataset = pd.read_csv(r'D:\AUniversity-2018\Winter2019\Spam-Detection-dataset\op_spam_v1.4\final.csv')
reviews = dataset.iloc[:,1]

#get labels and convert them to categorical format
y = dataset.iloc[:,2]
le = LabelEncoder()
y = le.fit_transform(y)
label = to_categorical(y, num_classes = 2)


#cleaning the vocabulary and tokenizing the reviews
vocabulary, numWords, review_vocab, cleaned_reviews = cleaning_reviews(reviews)

# keeping only unique words     
vocabulary = list(set(vocabulary))

# Here we provide the pretrained word embedding dictionary of glove with the vector dimension of 300
embedding_dim = 300
filepath ='C:/Users/Ashraf/Desktop/Keras_CNN/yelp_imdb_keras_data/data/glove_word_embeddings/glove.6B.300d.txt'

#Here we construct the embedding matrix of the fetched vocabulary from the reviews. 
#Each row of the matrix represents the corresponding word embedding vector
embedding_matrix = create_embedding_matrix(filepath, vocabulary, embedding_dim )

#We fetch the list of the words that are not covered by the pretrained dictionary and later eliminate them from the vocabulary list  
nonzero_elements, not_foundWords, not_foundIndex = not_found_words(embedding_matrix)

# Feature scaling make all features(the vectors values) between 0 and 1
X = embedding_matrix
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#calculate the TF-IDF of the vocabularies from the cleaned reviews 
df_TF_IDF, missed_tfidfW = words_frequency(cleaned_reviews, vocabulary)




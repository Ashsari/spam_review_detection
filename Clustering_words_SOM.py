# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 02:22:18 2020

@author: Ashraf
"""

from SRD_minisom import MiniSom
import matplotlib.pyplot as plt
from pylab import bone, pcolor, colorbar, plot, show, close
from random import seed
seed(9001)
rand = 205
np.random.seed(rand)


def count_cells_words(mappings, grid_dim):
    '''
    Counts each cells assigned words
    '''
    d = grid_dim*grid_dim
    count_matrix = np.zeros(d).reshape(grid_dim,grid_dim)
    for key in mappings:      
            count = len(mappings[key])
            indx = key
            #print(type(indx))
            #print(indx)
            count_matrix[key] = count_matrix[key] + count
    return count_matrix


def cell_words_dict(mappings, vocabulary):
    '''
    fetchs the vocabulary assigned to each cell on the grid map and creates a dictiony for them
    i.e: cell(1,0) has the vocbulary 'let', 'good' --> (1,0): ['let', 'good'] 
    '''
    vocab_dict = {}
    for key in mappings:
        #count = len(mappings[key])
        k = key
        values = mappings[k]
        voc= []
        for indx in values:
            voc.append(vocabulary[indx])
        vocab_dict[k] = voc
    return vocab_dict

def find_maped_words(vocab_dict, vocabulary_ham, vocabulary_spam ):
    '''
    creates 3 dictionaries of winning cell addresses: one for the common words on ham and spam, one for words on ham and one for words on spam
    '''
    common_wd ={}
    ham_wd ={}
    spam_wd={}
    for key in vocab_dict:
        voc_list = vocab_dict[key]
        cw = [] ; hw = [] ; sw = []        
        for word in voc_list:
            if (word in vocabulary_ham) & (word in vocabulary_spam):
                cw.append(word)
            elif (word in vocabulary_ham):
                hw.append(word)
            else:
                sw.append(word)
        if len(cw) != 0: 
            common_wd[key] = cw
        if len(hw) != 0:
            ham_wd[key] = hw
        if len(sw) != 0:
            spam_wd[key] = sw           
    return common_wd, ham_wd, spam_wd

def fetch_words_index(word_dict, vocabulary):
    map_index={}
    for key in word_dict:
        words = word_dict[key]
        indx= []
        for w in words:
            indx.append(vocabulary.index(w))
        map_index[key]= indx
    return map_index

        
#map_index = fetch_words_index(common_wd, vocabulary)        
        
def unique_val_frequency(count_matrix):   
    # Get a tuple of unique values & their frequency in numpy array
    uniqueValues, occurCount = np.unique(count_matrix, return_counts=True)
    #print("Unique Values : " , uniqueValues)#https://thispointer.com/python-find-unique-values-in-a-numpy-array-with-frequency-indices-numpy-unique/
    #print("Occurrence Count : ", occurCount)
    return uniqueValues, occurCount



def bar_plot(uniqueValues, occurCount):
    import matplotlib.pyplot as plt
    y_pos = uniqueValues
    plt.bar(uniqueValues,occurCount, align='center')
    plt.xticks(y_pos, uniqueValues)
    plt.xlabel('Unique Values')
    plt.ylabel("Occurrence Count")
    plt.title('count_matrix')
    plt.show()

def separate_ham_spam(reviews, y):
    ''' this function separates two classes using their lables
    '''
    import pandas as pd
    ham = []
    spam = []
    for l, r in zip(y, reviews):
        if l == 0:
            ham.append(r)
        else:
            spam.append(r)
    return ham, spam


###########################################################

#define the grid size for the SOM map by g. 
g = 20 
X_of_map = g
y_of_map = g

#the input length passed to the SOM is the embedding vectors dimension 
input_len = X.shape[1]

#defining the neghborhood radius sigma as s and the learning rate
s = 1; l = 1 # the radius

#We create a SOM object and train it on the word vectors that were prepared in the pre-processing step
som = MiniSom(x =X_of_map, y= y_of_map, input_len = input_len, sigma = s, learning_rate = l, random_seed = 8)
#Randomly initialize the weights that are assigne to each grid cells
som.random_weights_init(X)
#trining the SOM
som.train_random(data = X, num_iteration = 1200)
print('SOM training is started')




ham , spam =separate_ham_spam(reviews, y)

vocabulary_ham, numWords_ham, review_vocab_ham, cleaned_reviews_ham = cleaning_reviews(ham)
vocabulary_spam, numWords_spam, review_vocab_spam, cleaned_reviews_spam = cleaning_reviews(spam)

vocabulary_ham = list(set(vocabulary_ham))# 8986
vocabulary_spam = list(set(vocabulary_spam))#  9124

matches = list(set(vocabulary_spam) & set(vocabulary_ham))#5382

#visualizing the grid map trained by the SOM
mappings = som.win_map(X)

#map_cels = words_win_cel(mappings)


vocab_dict = cell_words_dict(mappings, vocabulary)
common_wd, ham_wd, spam_wd = find_maped_words(vocab_dict, vocabulary_ham, vocabulary_spam )  
 
count_matrix = count_cells_words(mappings, g)  
uniqueValues, occurCount = unique_val_frequency(count_matrix)

cw = count_cells_words(common_wd, g)
hw = count_cells_words(ham_wd, g)
sw = count_cells_words(spam_wd, g)
cw = cw.astype(int)
hw = hw.astype(int)
sw = sw.astype(int)

def plot_circled_distriution(vocabulary_ham, vocabulary_spam,fc, fh, fs):
    bone()#initialize the window that will contain the map
    pcolor(som.distance_map().T)# we add the info of the MID for all the winning nodes
    colorbar()#we use different colors corresponding to the different range values of the MID(Mean Interneuron Distances) to get All of these MIDs we can use the method of distance_map()
    for i, x in enumerate(X):
        w = som.winner(x)
        voc = vocabulary[i]
        ms_v = 0.12
        x_pos = [0.7, 0.8, 0.4]
        y_pos = [0.4, 0.8, 0.5]
        
        if fc == 1 & fh == 0 & fh == 0:
            cms = 0.2
            ms_v = cms
        if fc == 0 & fh == 1 & fh == 0:
            hms = 0.2 
            ms_v = hms
        if fc == 0 & fh == 0 & fh == 1:
            sms = 0.2
            ms_v = sms
                                   
        if  ( fc ==1 ) & (voc in vocabulary_ham) & (voc in vocabulary_spam) :
            count = cw[w]
            plot(w[0] + x_pos[0], 
                 w[1]+ y_pos[0],
                 'o', color=(0, 0, 1), ms = ms_v * count,
                 label='Spam and Ham Vocabulary')
            
        elif  (fh == 1) & (voc in vocabulary_ham):
            count = hw[w]
            plot(w[0] + x_pos[1], 
                 w[1]+ y_pos[1],
                 'o', color=(0, 1, 0), ms = ms_v * count,
                 label='Ham Vocabulary')
        elif (fs == 1) & (voc in vocabulary_spam):
            count = sw[w]
            plot(w[0] + x_pos[2], 
                 w[1]+ y_pos[2],
                 'o', color=(1, 0, 0), ms = ms_v * count,
                 label='Spam Vocabulary')
    

plot_circled_distriution(vocabulary_ham, vocabulary_spam,1,1,1)

     





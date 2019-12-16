
import glob
import os
import pandas as pd
import xlrd
import re
import nltk
from nltk.corpus import stopwords
from stemming.porter2 import stem
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.model_selection import KFold # import KFold
import numpy as np
from numpy import zeros
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, Input, concatenate, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from itertools import cycle, islice, dropwhile
from scipy.stats import multivariate_normal as mvn
import math
from utils import preprocessing, word_emdbedding_matrix, cross_validation, baseline_cnn, index_tiled_cnn, tiled_cnn, 
from utils import em_algorithm, get_dictionary, simple_hybrid_tiled_cnn, hybrid_tiled_cnn
SEED = 2000

path_train = 'C:/Users/Maria/Desktop/New folder/aclImdb/train/'
path_test = 'C:/Users/Maria/Desktop/New folder/aclImdb/test/'
embedding_matrix, embedding_matrix_sg, x_train, x_val, x_test, y_train, y_val, y_test, x_train_text, x_val_text, x_test_text, x, y = preprocessing(path_train, path_test)

# baseline CNN
# bigram
filter_size = [2, 3, 4]
acc_baseline_cnn = []
for i in filter_size:
    temp = baseline_cnn(x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, 1500, i, 'binary_crossentropy', 3, batch = 30, func_1 = 'relu', func_2 = 'tanh', func_3 = 'signoid')
    acc_baseline_cnn.append(temp)

# Tiled CNN
kernels = [2, 3]
acc_tiled_cnn = []
for i in kernels:
    for j in filter_size:
        temp = tiled_cnn(x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, 1500, j, 'binary_crossentropy', 3, i, batch = 30, func_1 = 'relu', func_2 = 'tanh', func_3 = 'signoid')
        acc_tiled_cnn.append(temp)

# Simple Hybrid Tiled CNN and Hybrid Tiled CNN
n_classes = [2, 3]
acc_simple_hybrid_tiled_cnn = []
acc_hybrid_tiled_cnn = []
for i in n_classes:
    ll, pis, mus, sigmas, ws, l2 = em_algorithm(embedding_matrix_sg, i, 20)
    for j in filter_size:
        temp1 = simple_hybrid_tiled_cnn(x_train, y_train, x_val, y_val, x_test, y_test, ws, embedding_matrix, 15000, j, 'binary_crossentropy', 3, i, l2, x, embedding_matrix_sg, batch = 30, func_1 = 'relu', func_2 = 'tanh', func_3 = 'sigmoid')
        acc_simple_hybrid_tiled_cnn.append(temp1)
        temp2 = hybrid_tiled_cnn(x_train, y_train, x_val, y_val, x_test, y_test, ws, embedding_matrix, 1500, j, 'binary_crossentropy', 3, i, l2, x, embedding_matrix_sg, batch = 30, func_1 = 'relu', func_2 = 'tanh', func_3 = 'signoid'):
        acc_hybrid_tiled_cnn.append(temp2)

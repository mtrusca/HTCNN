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
SEED = 2000

def preprocessing(path_train, path_test):
    train_pos = glob.glob(os.path.join(path_train, 'pos', "*.txt"))
    train_neg = glob.glob(os.path.join(path_train, 'neg', "*.txt"))
    test_neg = glob.glob(os.path.join(path_test, 'neg', "*.txt"))
    test_pos = glob.glob(os.path.join(path_test, 'pos', "*.txt"))
    corpus_train_pos = []
    corpus_train_neg = []
    corpus_test_pos = []
    corpus_test_neg = []
    for i in train_pos:
        with open(i, encoding = 'utf8') as f_input:
            corpus_train_pos.append(f_input.read())
    for i in train_neg:
        with open(i, encoding = 'utf8') as f_input:
            corpus_train_neg.append(f_input.read())
    for i in test_pos:
        with open(i, encoding = 'utf8') as f_input:
            corpus_test_pos.append(f_input.read())
    for i in test_neg:
        with open(i, encoding = 'utf8') as f:
            corpus_test_neg.append(f.read())
    l_pos = [1] * 12500
    l_neg = [0] * 12500
    test_label = ['test'] * 25000
    train_label = ['train']  * 25000
    corpus_train_pos = pd.DataFrame(corpus_train_pos, columns = ['x'])
    corpus_train_neg = pd.DataFrame(corpus_train_neg, columns = ['x'])
    corpus_test_pos = pd.DataFrame(corpus_test_pos, columns = ['x'])
    corpus_test_neg = pd.DataFrame(corpus_test_neg, columns = ['x'])
    l_pos = pd.DataFrame(l_pos, columns = ['y'])
    l_neg = pd.DataFrame(l_neg, columns = ['y'])
    test_label = pd.DataFrame(test_label, columns = ['label'])
    train_label = pd.DataFrame(train_label, columns = ['label'])
    corpus_train_pos = corpus_train_pos.join(l_pos)
    corpus_train_neg = corpus_train_neg.join(l_neg)
    corpus_test_pos = corpus_test_pos.join(l_pos)
    corpus_test_neg = corpus_test_neg.join(l_neg)
    corpus_train = pd.concat([corpus_train_pos, corpus_train_neg], ignore_index = True)
    corpus_test = pd.concat([corpus_test_pos, corpus_test_neg], ignore_index = True)
    corpus_train = corpus_train.join(train_label)
    corpus_test = corpus_test.join(test_label)
    corpus = pd.concat([corpus_train, corpus_test], ignore_index = True)
    corpus.x = [re.sub(r"http\S+", "", i) for i in corpus.x]
    corpus.x = [re.sub(r"\n", "", i) for i in corpus.x]
    corpus.x = [re.sub(r"br", "", i) for i in corpus.x]
    corpus.x = [re.sub(r">", "", i) for i in corpus.x]
    corpus.x = [re.sub(r"""[%$%&*#;:.,\/!?+-=(){}[]|<>"]""", "", i) for i in corpus.x]
    corpus.x = [re.sub(r"[0-9]", "", i) for i in corpus.x]
    corpus.x = [i.lower() for i in corpus.x]
    corpus.x = [re.sub(r" +", " ", i) for i in corpus.x]
    corpus.x = [i.lstrip() for i in corpus.x]
    stop = stopwords.words("english")
    corpus.x = [[word for word in i.split(" ")] for i in corpus.x]
    corpus.x = [[word for word in i if word not in stop] for i in corpus.x]
    corpus.x = [" ".join(i) for i in corpus.x]
    corpus.x = [re.sub(r" +", " ", i) for i in corpus.x]
    corpus.x = [[stem(word) for word in i.split(" ")] for i in corpus.x]
    corpus.x = [" ".join(i) for i in corpus.x]
    x = corpus.x
    y = corpus.y 
    embedding_matrix, embedding_matrix_sg = word_emdbedding_matrix(x, y)
    x_train, x_val, x_test, y_train, y_val, y_test, x_train_text, x_val_text, x_test_text = cross_validation(x, y, 4)
    return embedding_matrix, embedding_matrix_sg, x_train, x_val, x_test, y_train, y_val, y_test, x_train_text, x_val_text, x_test_text, x, y

def word_emdbedding_matrix(x, y):
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=SEED)
    def labelize_tweets_ug(tweets,label):
        result = []
        prefix = label
        for i, t in zip(tweets.index, tweets):
            result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
        return result
    #all_x = pd.concat([x_train, x_test])
    all_x = x
    all_x_w2v = labelize_tweets_ug(all_x, 'all')
    cores = multiprocessing.cpu_count()
    model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2,
                             workers=cores, alpha=0.065, min_alpha=0.065)
    model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])
    for epoch in range(30):
        model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),
                            total_examples=len(all_x_w2v), epochs=1)
        model_ug_cbow.alpha -= 0.002
        model_ug_cbow.min_alpha = model_ug_cbow.alpha
    model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2,
                           workers=cores, alpha=0.065, min_alpha=0.065)
    model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
    for epoch in range(30):
        model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),
                          total_examples=len(all_x_w2v), epochs=1)
        model_ug_sg.alpha -= 0.002
        model_ug_sg.min_alpha = model_ug_sg.alpha

    ### word embeddings matrix
    embeddings_index = {}
    for w in model_ug_cbow.wv.vocab.keys():
        embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
    tokenizer = Tokenizer(num_words=100000, oov_token = True)
    #tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x)
    num_words = 100000
    embedding_matrix = np.zeros((num_words, 200))
    embedding_matrix_sg = np.zeros((num_words, 100))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        embedding_vector_sg = model_ug_sg.wv[w]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        if embedding_vector_sg is not None:
            embedding_matrix_sg[i] = embedding_vector_sg
    return embedding_matrix, embedding_matrix_sg

def cross_validation(x, y, splits):
    x_tr_val, x_test_text, y_tr_val, y_test = train_test_split(x, y, test_size=.2, random_state=SEED)
    x_tr_val = np.array(x_tr_val, dtype=pd.Series)
    x_test_text = np.array(x_test_text, dtype=pd.Series)
    y_tr_val = np.array(y_tr_val, dtype=pd.Series)
    y_test = np.array(y_test, dtype=pd.Series)
    x_train_text, x_val_text, y_train, y_val = [], [], [], []
    skf = KFold(n_splits=splits, shuffle=True, random_state=0)
    for train_index, val_index in skf.split(x_tr_val, y_tr_val):
        x_train_text.append(x_tr_val[train_index])
        x_val_text.append(x_tr_val[val_index])
        y_train.append(y_tr_val[train_index])
        y_val.append(y_tr_val[val_index])
    x_train, x_val = [], []
    tokenizer = Tokenizer(num_words=100000, oov_token = True)
    tokenizer.fit_on_texts(x)
    for i in range(4):
        sequence_tr = tokenizer.texts_to_sequences(x_train_text[i])
        x_train_seq = pad_sequences(sequence_tr, maxlen = 1500)
        x_train.append(x_train_seq)
        sequence_val = tokenizer.texts_to_sequences(x_val_text[i])
        x_val_seq = pad_sequences(sequence_val, maxlen = 1500)
        x_val.append(x_val_seq)
    sequences_test = tokenizer.texts_to_sequences(x_test_text)
    x_test = pad_sequences(sequences_test, maxlen=1500)
    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_text, x_val_text, x_test_text

def baseline_cnn(x_train, y_train, x_val, y_val, x_test, y_test, emb_matrix, length, filter_size, loss, no_epochs, batch = 30, func_1 = 'relu', func_2 = 'relu', func_3 = 'softmax'):
    iters = len(x_train)
    m, n = emb_matrix.shape
    acc = []
    for iter in range(iters):
        model = Sequential()
        e = Embedding(m, n, weights=[emb_matrix], input_length=length, trainable=True)
        model.add(e)
        model.add(Conv1D(filters=100, kernel_size=filter_size, padding='valid', activation=func_1, strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation=func_2))
        model.add(Dense(1, activation=func_3))
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        model.fit(x_train[iter], y_train[iter], validation_data=(x_val[iter], y_val[iter]), epochs=no_epochs, batch_size=batch, verbose=2)
        #model.predict(x_test)
        test_acc = model.evaluate(x = x_test, y = y_test)[1]
        acc.append(test_acc)
    final_acc = np.avearge(acc)
    return final_acc

def index_tiled_cnn(m, n, k):
    dec, int = math.modf(m/(k+n-1))
    temp = int * (k+n-1)
    index = []
    length = []
    for i in range(3):
        if (i+1)/k <= dec:
            index.append(np.int(temp + n+i-1))
        else:
            index.append(np.int(temp + i -k+1))
        length.append(index[i] - i)
    return index,  length


def tiled_cnn(x_train, y_train, x_val, y_val, x_test, y_test, emb_matrix, length, filter_size, loss, no_epochs, kernels, batch = 30, func_1 = 'relu', func_2 = 'relu', func_3 = 'softmax'):
    iters = len(x_train)
    dim1, dim2 = emb_matrix.shape
    acc = []
    for i in range(iters):
        def input_change(matrix):
            matrix_transpose = []
            for j in range(kernels):
                temp = matrix.transpose()
                temp = temp[0:index_tiled_cnn(length, filter_size, kernels)[0][j]]
                temp = temp.transpose()
                matrix_transpose.append(temp)
            return matrix_transpose
        x_train = input_change(x_train[i])
        x_val = input_change(x_val[i])
        x_test = input_change(x_test[i])
        inputs = []
        e_branches = []
        for j in range(kernels):
            input = Input(shape=(index_tiled_cnn(length, filter_size, kernels)[1][j],))
            e_encoder = Embedding(dim1, dim2, weights=[emb_matrix], input_length=index_tiled_cnn(length, filter_size, kernels)[1][j], trainable=True)(input)
            e_branch = Conv1D(filters=100, kernel_size=filter_size, padding='valid', activation=func_1, strides=kernels)(e_encoder)
            e_branches.append(e_branch)
            inputs.append(input)
        merged = e_branches[0]
        for j in range(kernels):
            if j == 0:
                continue
            merged = concatenate([merged, e_branches[j]], axis=1)
        merged = GlobalMaxPooling1D()(merged)
        merged = Dense(256, activation=func_2)(merged)
        merged = Dense(1)(merged)
        output = Activation(func_3)(merged)
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train[i], validation_data = (x_val, y_val[i]), batch_size=batch, epochs=no_epochs)
        #model.predict(x_test)
        test_acc = model.evaluate(x = x_test, y = y_test)
        acc.append(test_acc)
    final_acc = np.avearge(acc)
    return final_acc

def em_algorithm(embedding_matrix_sg, n_classes, max_iter):
    np.random.seed(8)
    pis = np.random.random(n_classes)
    pis /= pis.sum()
    n, p = embedding_matrix_sg.shape
    mus = np.random.random((n_classes,p))
    sigmas = np.array([np.eye(p)] * n_classes) * embedding_matrix_sg.std()
    e = np.where(~embedding_matrix_sg.any(axis=1))[0]
    embedding_matrix_sg = np.delete(embedding_matrix_sg, e, axis = 0)
    l = list(range(100000))
    #zeros
    l1 = e.tolist()
    #non-zeros
    l2 = [x for x in l if x not in l1]
    def em_gmm_orig(xs, pis, mus, sigmas, tol=0.01):
        k = len(pis)
        ll_old = 0
        for i in range(max_iter):
            print('\nIteration: ', i)
            print()
            exp_A = []
            exp_B = []
            ll_new = 0
            # E-step
            ws = np.zeros((k, n))
            for j in range(len(mus)):
                for i in range(n):
                    ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
                    if i % 1000 == 0:
                        print(i)
            ws /= ws.sum(0)
            # M-step
            pis = np.zeros(k)
            for j in range(len(mus)):
                for i in range(n):
                    pis[j] += ws[j, i]
            pis /= n
            mus = np.zeros((k, p))
            for j in range(k):
                for i in range(n):
                    mus[j] += ws[j, i] * xs[i]
                mus[j] /= ws[j, :].sum()        
            sigmas = np.zeros((k, p, p))
            for j in range(k):
                for i in range(n):
                    ys = np.reshape(xs[i]- mus[j], (p,1))
                    sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
                sigmas[j] /= ws[j,:].sum()        
            # update complete log likelihoood
            ll_new = 0.0
            for i in range(n):
                s = 0
                for j in range(k):
                    s += pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
                ll_new += np.log(s)
            print(f'log_likelihood: {ll_new:3.4f}')
            if np.abs(ll_new - ll_old) < tol:
                break
            ll_old = ll_new
         return ll_new, pis, mus, sigmas, ws
    ll, pis, mus, sigmas, ws = em_gmm_orig(embedding_matrix_sg, pis, mus, sigmas)
    return ll, pis, mus, sigmas, ws, l2


def get_dictionary(l2, ws, x, embedding_matrix_sg, n_classes):
    clusters = [[] for _ in range(n_classes)]
    m, n = embedding_matrix_sg.shape
    ws_t = np.transpose(ws)
    for i in range(m - 1):
        clusters[np.argmax(ws_t[i])].append(l2[i])
    tokenizer = Tokenizer(num_words=100000, oov_token = True)
    tokenizer.fit_on_texts(x)
    word_clusters = []
    for cluster in clusters:
        index = clusters.index(cluster)
        word_cluster = []
        word_cluster = [w for w, c in tokenizer.word_index.items() if c in cluster and w != True]
        word_clusters.append(word_cluster)
    dicts = [[] for _ in range(n_classes)]
    dicts_1 = [[] for _ in range(n_classes)]
    for i in range(n_classes):
        dicts[i] = Tokenizer(num_words=100000, oov_token = True)
        dicts[i].fit_on_texts(x)
        dicts_1[i] = Tokenizer(num_words=100000, oov_token = True)
        dicts_1[i].fit_on_texts(x)
    for i in range(n_classes):
        for w in word_clusters[i]:
            j = [k for k in range(n_classes) if k != i]
            for l in j:
                del dicts[l].word_index[w]
                del dicts[l].word_docs[w]
                del dicts[l].word_counts[w]
    return word_clusters, clusters, dicts, dicts_1


def simple_hybrid_tiled_cnn(x_train, y_train, x_val, y_val, x_test, y_test, ws, emb_matrix, length, filter_size, loss, no_epochs, n_classes, l2, x, embedding_matrix_sg, batch = 30, func_1 = 'relu', func_2 = 'relu', func_3 = 'softmax'):
    iters = len(x_train)
    dim1, dim2 = emb_matrix.shape
    acc = []
    _, _, dicts, _ = get_dictionary(l2, ws, x, embedding_matrix_sg, n_classes)
    for i in range(iters):
        x_train = []
        def input_change(matrix):
            matrix = []
            for j in range(n_classes):
                sequences_cluster = dicts[j].texts_to_sequences(x_train_text[i])
                temp = pad_sequences(sequences_cluster, maxlen=length)
                matrix.append(temp)
            return matrix
        x_train = input_matrix(x_train[i]) 
        x_val = input_matrix(x_val[i])     
        x_test = input_matrix(x_test[i])  
        inputs = []
        e_branches = []
        for j in range(kernels):
            input = Input(shape=(length,))
            e_encoder = Embedding(dim1, dim2, weights=[emb_matrix], input_length=length, trainable=True)(input)
            e_branch = Conv1D(filters=100, kernel_size=filter_size, padding='valid', activation=func_1, strides=kernels)(e_encoder)
            e_branches.append(e_branch)
            inputs.append(input)
        merged = e_branches[0]
        for j in range(kernels):
            if j == 0:
                continue
            merged = concatenate([merged, e_branches[j]], axis=1)
        merged = GlobalMaxPooling1D()(merged)
        merged = Dense(256, activation=func_2)(merged)
        merged = Dense(1)(merged)
        output = Activation(func_3)(merged)
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train[i], validation_data = (x_val, y_validation[i]), batch_size=batch, epochs=no_epochs)
        test_acc = model.evaluate(x = x_test, y = y_test)
        acc.append(test_acc)
    final_acc = np.avearge(acc)
    return final_acc


def hybrid_tiled_cnn(x_train, y_train, x_val, y_val, x_test, y_test, ws, emb_matrix, length, filter_size, loss, no_epochs, n_classes, l2, x, emb_matrix_sg, batch = 30, func_1 = 'relu', func_2 = 'relu', func_3 = 'softmax'):
    iters = len(x_train)
    dim1, dim2 = emb_matrix.shape
    acc = []
    _, _, dicts, dicts_1 = get_dictionary(l2, ws, x, emb_matrix_sg, n_classes)
    for i in range(iters):
        x_train = []
        def input_change(matrix):
            matrix = []
            matrix_neighbor = []
            for j in range(n_classes):
                sequences_cluster = dicts[j].texts_to_sequences(x_train_text[i])
                temp = pad_sequences(sequences_cluster, maxlen=length)
                matrix.append(temp)
                sequences_cluster = dicts_1[j].texts_to_sequences(x_train_text[i])
                temp = pad_sequences(sequences_cluster, maxlen=length)
                matrix_neighbor.append(temp)
            return matrix, matrix_neighbor
        x_train = input_matrix(x_train[i])[0] 
        x_val = input_matrix(x_val[i])[0]     
        x_test = input_matrix(x_test[i])[0]
        x_train_neighbor = input_matrix(x_train[i])[1] 
        x_val_neighbor = input_matrix(x_val[i])[1]     
        x_test_neighbor = input_matrix(x_test[i])[1]       
        def neighbor_change(matrix, matrix_neighbor):
            index = set(list(range(length)))
            n_doc = matrix[0].shape[0]
            for j in range(n_doc):
            neighbors = []
            for c in n_classes:
                ind = [k for k, x in enumerate(matrix[c][j]) if x != 1]
                for c1 in range(n_classes):
                    if c1 = 0:
                        continue
                    temp2 = [x + c1 if x < length-c1 else x for x in ind]
                    temp3 = [x - c1 if x > length+c1 else x for x in ind]
                    ind = list(set().union(ind, temp2, temp3))
                diff = index - set(ind)
                diff = list(diff)
                matrix_neighbor[c][j][diff1] = 0
                neighbors.append(matrix_neighbor)
            return neighbors
        x_train_neighbor = neighbor_change(x_train, x_train_neighbor)
        x_val_neighbor = neighbor_change(x_val, x_val_neighbor)
        x_test_neighbor = neighbor_change(x_test, x_test_neighbor)
        inputs = []
        e_branches = []
        for j in range(n_classes):
            input = Input(shape=(length,))
            e_encoder = Embedding(dim1, dim2, weights=[emb_matrix], input_length=length, trainable=True)(input)
            e_branch = Conv1D(filters=100, kernel_size=filter_size, padding='valid', activation=func_1, strides=kernels)(e_encoder)
            e_branches.append(e_branch)
            inputs.append(input)
        merged = e_branches[0]
        for j in range(n_classes):
            if j == 0:
                continue
            merged = concatenate([merged, e_branches[j]], axis=1)
        merged = GlobalMaxPooling1D()(merged)
        merged = Dense(256, activation=func_2)(merged)
        merged = Dense(1)(merged)
        output = Activation(func_3)(merged)
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        model.fit(x_train_neighbor, y_train[i], validation_data = (x_val_neighbor, y_validation[i]), batch_size=batch, epochs=no_epochs)
        test_acc = model.evaluate(x = x_test_neighbor, y = y_test)
        acc.append(test_acc)
    final_acc = np.avearge(acc)
    return final_acc

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:34:08 2021

@author: Wei Zhao @ Metis, 02/12/2021
"""
import pickle
import string
import re
from collections import defaultdict
from itertools import chain
from math import sqrt

from textblob import Word
from nltk import word_tokenize
from nltk import pos_tag

from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
#%%
#--------------------------------------------------------
def save_as_pickle(fn, data):
    """
    Function to save data as a pickled file
    Parameters
    ----------
    fn : str
        File directory.
    data : any type, but recommend dictionary
        data to save.
    Returns
    -------
    None.
    """
    with open(fn, 'wb') as to_write:
        pickle.dump(data, to_write)
    print('Saved data to "' + fn + '"')

#--------------------------------------------------------
def read_from_pickle(fn):
    """
    Function to read data from a pickled file
    Parameters
    ----------
    fn : str
        File directory.
    data : any type, but recommend dictionary
        data to read.
    Returns
    -------
    data : same as data
        Read in this variable.
    """
    with open(fn,'rb') as read_file:
        data = pickle.load(read_file)
    print('Read data from "' + fn + '"')
    return data

#--------------------------------------------------------
def remove_punctuation(txt):
    """
    Function to remove punctuation in sensences.

    Parameters
    ----------
    text : str
        text input with punctuation.

    Returns
    -------
    words_wo_punct : str
        text output with no punctuation.

    """

    words_wo_punct = re.sub('[%s]' % re.escape(string.punctuation+'≥'+'≤'),
                            ' ', txt)

    return words_wo_punct
#--------------------------------------------------------
def remove_num(txt):
    """
    remove numbers in text

    Parameters
    ----------
    txt : str
        text with numbers.

    Returns
    -------
    txt : str
        text without numbers.

    """
    txt = re.sub('\w*\d\w*', '', txt)
    return txt
#%%
def paragraph_lemma(txt):
    """
    Lemmatize a paragraph.

    Parameters
    ----------
    txt : str
        Texts after removing numbers and punctuations.

    Returns
    -------(pd.Series(nltk.ngrams(words, 2)).value_counts())[:10]
    lemma_txt : str
        lemmatized paragraph with
        each word being lowercase.

    """
    token_word = word_tokenize(txt)
    lemma_txt = ' '.join([Word(word.lower()).lemmatize()
                      for word in token_word
                      if len(Word(word.lower()).lemmatize()) > 1])
    return lemma_txt
#--------------------------------------------------------
def get_nouns(txt):
    '''Given a string of text, tokenize the text and
    pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(txt)
    all_nouns = [word for (word, pos) in pos_tag(tokenized)
                 if is_noun(pos)]
    return ' '.join(all_nouns)
#--------------------------------------------------------
def get_nouns_adj(text):
    '''Given a string of text, tokenize the text and
    pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized)
                 if is_noun_adj(pos)]
    return ' '.join(nouns_adj)
#--------------------------------------------------------
def display_topics(model, feature_names, no_top_words,
                   no_topic_to_plot=20, topic_names=None,
                   display=False):
    """
    Display and store topic words in a dictionary

    Parameters
    ----------
    model : NMF model
        An NMF model.
    feature_names : list
        get feature from vesterizer.
    no_top_words : int
        number of topic words.
    no_topic_to_plot : int, optional
        number of topic to plot. The default is 20.
    topic_names : list, optional
        topic name. The default is None.
    display : bool, optional
        Display or not display with generating
        the topic word dictionary only.
        The default is False.

    Returns
    -------
    output : dictionary
        Topic word dictionary.

    """
    output = defaultdict(list)
    if display:
        for ix, topic in enumerate(model.components_):
            if not topic_names or not topic_names[ix]:
                print("\nTopic ", ix)
            else:
                print("\nTopic: '",topic_names[ix],"'")

            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
            if ix >= no_topic_to_plot-1:
                break

    for ix, topic in enumerate(model.components_):
        output[ix] = ", ".join([feature_names[i]
                                for i in topic.argsort()[:-no_top_words - 1:-1]])

    return output
#--------------------------------------------------------
def del_abbreviation(txt):
    """
    Delete abbreviations

    Parameters
    ----------
    txt : str
        Texts with abbreviations.

    Returns
    -------
    txt : str
        Texts without abbreviations..

    """
    while 'mtbi' in txt:
        txt.remove('mtbi')
    while 'tbi' in txt:
        txt.remove('tbi')
    while 'icp' in txt:
        txt.remove('icp')
    while 'gc' in txt:
        txt.remove('gc')
    while 'ci' in txt:
        txt.remove('ci')
    return txt
#--------------------------------------------------------
def merge_ngrams(word_list, dict_ngrams, exception_dict):
    """
    Merge words to ngrams.

    Parameters
    ----------
    word_list : list
        List of single words.
    dict_ngrams : dictionary
        dict_ngrams includes bi, tri, and quangrams.
        Each ngram is a list.
    exception_dict : list
        A list of words to be excluded.

    Returns
    -------
    word_list_ngrams_merged : list
        word list with ngrams.

    """
    exception = dict_ngrams['quagrams'] \
                + dict_ngrams['trigrams'] \
                + dict_ngrams['bigrams'] \
                + stopwords.words('english') \
                + exception_dict

    exception = list(set(chain
                         .from_iterable([e.split(' ')
                                         for e in exception])))

    word_list_ngrams_merged = []
    i = 0
    while i < len(word_list):
        uni = word_list[i]
        bi = ' '.join(word_list[i:i+2])
        tri = ' '.join(word_list[i:i+3])
        qua = ' '.join(word_list[i:i+4])

        if qua in dict_ngrams['quagrams']:
            word_list_ngrams_merged.append(qua)
            i+=4
        elif tri in dict_ngrams['trigrams']:
            word_list_ngrams_merged.append(tri)
            i += 3
        elif bi in dict_ngrams['bigrams']:
            word_list_ngrams_merged.append(bi)
            i += 2
        else:
            if uni not in exception:
                word_list_ngrams_merged.append(uni)
            i += 1
    return word_list_ngrams_merged
#--------------------------------------------------------
def plot_top_words(model, feature_names,
                   n_top_words, no_docs,
                   figsize, topic_name):
    fig, axes = plt.subplots(5, 4, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]

        ax.barh(top_features, weights, height=0.7)
        ax.set_title('{0}\n ({1} papers)'
                     .format(topic_name[topic_idx],
                             no_docs.values[no_docs.index[topic_idx]]),
                     fontdict={'fontsize': 25})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=25)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
#--------------------------------------------------------
def dummy(doc):
    """
    Dummy tokenizer to avoid tokenization
    when vectorizing the data
    """
    doc2 = doc.split(',')
    return doc2
#--------------------------------------------------------
def store_topic_words(model_dict, vectorizer_dict):
    """
    Store topic words in a list.

    Parameters
    ----------
    model_dict : dictionary
        A dictionary with all NMF models stored.
    vectorizer_dict : dictionary
        A dictionary with all vecterizers stored.

    Returns
    -------
    topic_words : dictionary
        A dictionary with all topic words stored.

    """
    years = range(1991, 2022)
    topic_words=defaultdict(list)
    for y in years:
        words = vectorizer_dict[y].get_feature_names()
        topic_word_yearly = model_dict[y].components_.argsort(axis=1)[:,-1:-11:-1]
        topic_words[y] = [[words[e] for e in l] for l in topic_word_yearly]
    return topic_words
#--------------------------------------------------------
# REF: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
def evaluate_arima_model(X, arima_order, training_portion=0.66):
    """
    Function to evaluate arima model.

    Parameters
    ----------
    X : array or list
        All dataset.
    arima_order : tuple
        The (p,d,q) order of the model for
        the number of AR parameters,
        differences, and MA parameters to use.
    training_portion : float, optional
        portion of training data. The default is 0.66.

    Returns
    -------
    rmse : float
        root mean squared error.

    """
    # prepare training dataset
    train_size = int(len(X) * training_portion)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset,
                    p_values,
                    d_values,
                    q_values,
                    training_portion=0.66):
    """
    Fucntion to tune hyperparameters of arima model

    Parameters
    ----------
    dataset : array or list
        All dataset.
    p_values : int
        parameter for autoregression.
    d_values : int
        parameter for differentiation.
    q_values : int
        parameter for moving average.
    training_portion : float, optional
        portion of training data. The default is 0.66.

    Returns best_cfg, best_score
    -------
    best_cfg: best configuration of p, d, and q.
    best score: smallest rmse value.

    """
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order, training_portion=training_portion)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
#                     print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return best_cfg, best_score

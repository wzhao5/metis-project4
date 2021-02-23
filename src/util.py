#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:34:08 2021

@author: Wei Zhao @ Metis, 02/12/2021
"""
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob, Word
from nltk import word_tokenize, pos_tag
import re
from collections import defaultdict
from itertools import chain
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
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
    
    # no_punct = [words for words in text
    #             if words not in string.punctuation]
    # words_wo_punct = ''.join(no_punct)
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
# def dummy(doc):
    # """
    # Dummy tokenizer to avoid tokenization
    # when vectorizing the data
    # """
    # return [doc]

#--------------------------------------------------------
def plot_top_words(model, feature_names,
                   n_top_words, title, no_docs,
                   figsize):
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        
        ax.barh(top_features, weights, height=0.7)
        ax.set_title('Topic {0} ({1} papers)'
                     .format(topic_idx+1,
                             no_docs.values[no_docs.index[topic_idx]]),
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=25)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=35)
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
    years = range(1991, 2022)
    topic_words=defaultdict(list)
    for y in years:
        words = vectorizer_dict[y].get_feature_names()
        topic_word_yearly = model_dict[y].components_.argsort(axis=1)[:,-1:-11:-1]
        topic_words[y] = [[words[e] for e in l] for l in topic_word_yearly]
    return topic_words       
#%%
if __name__ == '__main__':
    import spacy
    sp = spacy.load('en_core_web_sm')
    
    sp('The striped bats are hanging on their feet for best"').lemma_

    #%%
    import nltk
    from nltk.stem import WordNetLemmatizer 
    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    lemmatizer.lemmatize('was')





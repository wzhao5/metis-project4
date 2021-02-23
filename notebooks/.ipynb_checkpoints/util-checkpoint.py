#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:34:08 2021

@author: Wei Zhao @ Metis, 02/12/2021
"""
import pickle
import string
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
def remove_punctuation(text):
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
    
    no_punct = [words for words in text
                if words not in string.punctuation]
    
    words_wo_punct = ''.join(no_punct)
    
    return words_wo_punct
#--------------------------------------------------------





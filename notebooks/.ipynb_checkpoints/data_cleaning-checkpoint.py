#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is to query and download data from PubMed

@author: Wei Zhao @ Metis, 02/12/2021
"""
#%%
import string
string.punctuation
#%%
def remove_punctuation(text):
    
    no_punct = [words for words in text
                if words not in string.punctation]
    
    words_wo_punct = ''.join(no_punct)
    
    return words_wo_punct
    

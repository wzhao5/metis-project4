#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is for data cleaning

@author: Wei Zhao @ Metis, 02/12/2021
"""
#%%
import pandas as pd
from util import remove_punctuation
from util import read_from_pickle
#%%
def get_first_author(txt):
    '''
    Get full name of the first authors

    Parameters
    ----------
    txt : dictionary
        Name dictionary.

    Returns
    -------
    name : str
        Full name of the first author.

    '''
    first = txt[0]['firstname']
    last = txt[0]['lastname']
    if first is None:
        first = "nan"
    if last is None:
        last = 'nan'
    name = first + ' ' + last
    return name
#-----------------------------------------------------
def get_last_author(txt):
    '''
    Get full name of the last authors.
    Last author typically is corresponding author.

    Parameters
    ----------
    txt : list of dictionary
        Name dictionary.

    Returns
    -------
    name : str
        Full name of the last author.

    '''
    first = txt[-1]['firstname']
    last = txt[-1]['lastname']
    if first is None:
        first = "nan"
    if last is None:
        last = 'nan'
    name = first + ' ' + last
    return name
#-----------------------------------------------------
def get_publication_year(data):
    '''
    Function to get publication year

    Parameters
    ----------
    data : pandas date time structure
        date in raw data.

    Returns
    -------
    year : int
        year.

    '''
    if hasattr(data, 'year'):
        year = data.year
    else:
        year = int(data)
    return year
#-----------------------------------------------------
def clean_journal_title(txt):
    '''
    Function to clean journal title so that
    only main title remains

    Parameters
    ----------
    txt : str
        Journal title in raw data.

    Returns
    -------
    journal_title : str
        clean journal title.

    '''

    if ':' in txt:
        txt_split = txt.split(':')
        return txt_split[0]
    elif '(' in txt:
        txt_split = txt.split('(')
        return txt_split[0]

    return txt

#-----------------------------------------------------
def data_cleaning(df):

    # delete article_id, publication_type,
    # not useful in the analysis
    df = df.drop('article_id', axis=1)
    df = df.drop('publication_type', axis=1)
    idx = df[df['authors'].str.len() == 0].index
    df = df.drop(idx, inplace=False)
    df['first_author'] = (df['authors']
                          .apply(get_first_author))
    df['last_author'] = (df['authors']
                          .apply(get_last_author))
    df = df.drop('authors', axis=1)
    df['year'] = (df['publication_date']
                  .apply(get_publication_year))
    idx = df[df['year'] < 1991].index
    df = df.drop(idx, inplace=False)
    df = df.drop('publication_date', axis=1)
    df.dropna(subset=['abstract', 'title'], axis=0, inplace=True)

    df['title'] = df['title'].apply(remove_punctuation)
    df['journal'] = df['journal'].apply(clean_journal_title)

    return df
#%%
def main(df):
    df = data_cleaning(df)
    return df

if __name__ == '__main__':
    fn = '../data/data.pickle'
    data = read_from_pickle(fn)
    df = pd.DataFrame(data)
    df = main(df)

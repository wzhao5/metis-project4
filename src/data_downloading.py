#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is to query and download data from PubMed

@author: Wei Zhao @ Metis, 02/12/2021
"""
#%%
from util import *
from pymed import PubMed
from collections import defaultdict

#%%
pubmed = PubMed(tool="MyTool", email="")
query = ['((traumatic brain injury) ' + 
         'OR (concussion) ' + 
         'OR (brain biomechanics)) ' +
         'AND ("1991/01/01"[Date - Create] : "3000"[Date - Create])' + 
         'AND (english[Language])'
         ]
results = pubmed.query(query, max_results=12000000)

#%%
def download_data(results):
    # Loop over the retrieved articles
    data_dict = defaultdict(list)
    c = 0
    for article in results:
        c += 1
        if c%1200 == 0:      
            print(c)
        # Extract and format information from the article
        data_dict['article_id'].append( article.pubmed_id)
        #-----------------------------------------------------
        data_dict['title'].append(article.title)
        #-----------------------------------------------------
        data_dict['authors'].append(article.authors)
        #-----------------------------------------------------
        if not hasattr(article, 'journal'):
            data_dict['journal'].append('Nan')
        else:
            data_dict['journal'].append(article.journal)
        #-----------------------------------------------------
        if not hasattr(article, 'keywords'):
           data_dict['keywords'].append('Nan')
        else:
            if article.keywords:
                if None in article.keywords:
                    article.keywords.remove(None)
            data_dict['keywords'].append('", "'.join(article.keywords))
        #-----------------------------------------------------    
        data_dict['publication_date'].append(article.publication_date)
        #-----------------------------------------------------
        data_dict['abstract'].append(article.abstract)
        #-----------------------------------------------------
        if not hasattr(article, 'publication_type'):
            data_dict['publication_type'].append('Nan')
        else:
            data_dict['publication_type'].append(article.publication_type)

    #-----------------------------------------------------    
    fn = '../data/data.pickle'
    save_as_pickle(fn, data_dict)

#%%
def main(results):
    download_data(results)

if __name__ == '__main__':
    main(results)

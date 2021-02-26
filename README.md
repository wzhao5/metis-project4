# Metis Data Science Bootcamp | Project 4

---

## **Investigation of Research Topic Development on Traumatic Brain injury**

**Topic Modeling**

Project timeline: 2.5 weeks

Final presentation is posted [here](https://github.com/weizhao-BME/metis-project2/blob/main/presentation/presentation_project2.pdf).

------------

### **Introduction** 

When writing a research proposal, the first thing is to choose a research topic. Therefore, it is important to systematically investigate the development of research topics and understand how the research field has been evolving. In this analysis, the evolution of research topics related to traumatic brain injury was investigated using an natural language processing approach so that academic researchers can better understand the historical and current situation of this field. 

***********************

### **Methods**

#### Data acquisition

The data was queried on PubMed using a python toolkit "pymed". The querying keywords include "traumatic brain injury, brain biomechanics, and concussion" with time period limited from1991 to 2021. The contents downloaded included article title, abstract, authors, publication year, and journal title.  

#### Exploratory analysis

The number of publications and top 10 journals with most publications from 1991 to 2021 were queried using Pandas. 

#### Data cleaning and preprocessing

First, article titles and abstracts were combined to maximize the information included in the text and words. Second, punctuations were removed from all the texts. Third, all the texts were tokenized using "word_tokenizer" in natural language toolkit (nltk). All nouns and adjectives were identified and retained. Further, all the retained words were lemmatized to group together the inflected form of a word. Last, a dictionary of n-grams based on domain knowledge were generated to maintain specific phrases in the research field. For example, words like glasgow coma scale, blood brain barrier, white matter, blood flow, or ct scan were included. In addition, stopword dictionary from was expanded by adding more unnecessary words and phrases for the topic modeling in this analysis. For example, words like day, month, area, value, or article were excluded.



#### Topic modeling

First, term frequency–inverse document frequency (TF-IDF) vectorizer were used to transform the texts into vectors to scale down the impact of tokens that occur very frequently in a given corpus [REF]([sklearn.feature_extraction.text.TfidfTransformer — scikit-learn 0.24.1 documentation (scikit-learn.org)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)). The min_df and max_df were defined as 2, and 0.95 to ignore the vocabulary terms that have a document frequency lower and higher than the given thresholds, respectively [REF](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py). Second, an NMF model with 20 components (representing 20 topics) were established. The generalized Kullback-Leibler divergence were used with the "beta_loss" keyword. This is equivalent to a probabilistic latent semantic indexing. Other definition of parameters were referred to ([REF](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)). Next, an ARIMA model was established based on the yearly counts of publications for a topic normalized by the total counts of publications in that year regardless of topics. A grid search cross validation (66% training vs. 34% testing data) was conducted to tune the best configuration of AR parameters, differences, and MA parameters. The root mean squared error was minimized in the cross-validations. Finally, using the tuned ARIMA models the normalized counts of publications were forecasted for each topic. The figure below shows the workflow of data preprocessing and topic modeling. 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/topic_modeling_workflow.png" alt="Figure 1" width="700"/>

---

### Results and Discussion

The results show five examples of topic development for illustration. 



<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.2.png" alt="Figure 2" width="700"/>



<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.3.png" alt="Figure 3" width="700"/>



<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.4.png" alt="Figure 4" width="700"/>



<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.5.png" alt="Figure 5" width="700"/>



<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.6.png" alt="Figure 6" width="700"/>





---

### Conclusions

XX








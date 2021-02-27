# Metis Data Science Bootcamp | Project 4

---

## **Investigation of Research Topic Development on Traumatic Brain injury**

**Topic Modeling**

Project timeline: 2.5 weeks

Final presentation is posted [here](https://github.com/weizhao-BME/metis-project2/blob/main/presentation/presentation_project2.pdf).

------------

### **Introduction** 

When writing a research proposal, the first thing is to choose a research topic, which many junior researchers are sill struggling with. Some may believe that reading sufficient system review papers would help them target a research topic. But it is not sufficient at all. When finding a research topic, it is important to systematically investigate the development of research topics and understand how the research field has been evolving. Identifying hot research topics, clustering research articles with a hot topic, and reading these clustered articles will be more efficient and more focused for those junior researchers. For illustration of this method, the evolution of research topics related to traumatic brain injury was investigated using a natural language processing approach so that academic researchers can use this method to find hot research topics that they are interested in. 

***********************

### **Methods**

#### Data acquisition

The data was queried on PubMed using a python toolkit "pymed". The querying keywords include "traumatic brain injury, brain biomechanics, and concussion" with time period limited from1991 to 2021. The contents downloaded included article title, abstract, authors, publication year, and journal title.  

#### Exploratory analysis

The number of publications and top 10 journals with most publications from 1991 to 2021 were queried using Pandas. 

#### Data cleaning and preprocessing

First, article titles and abstracts were combined to maximize the information included in the text and words. Second, punctuations were removed from all the texts. Third, all the texts were tokenized using "word_tokenizer" in natural language toolkit (nltk). All nouns and adjectives were identified and retained. Further, all the retained words were lemmatized to group the inflected form of a word. Last, a dictionary of n-grams based on domain knowledge were generated to maintain specific phrases in the research field. For example, words like glasgow coma scale, blood brain barrier, white matter, blood flow, or ct scan were included. In addition, stopword dictionary was expanded by adding more unnecessary words and phrases for the topic modeling in this analysis. For example, words like day, month, area, value, or article were excluded.

#### Topic modeling

First, term frequency–inverse document frequency (TF-IDF) vectorizer were used to transform the texts into vectors to scale down the impact of tokens that occur very frequently in a given corpus ([REF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)). The min_df and max_df were defined as 2, and 0.95 to ignore the vocabulary terms that have a document frequency lower and higher than the given thresholds, respectively ([REF)](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py). Second, an NMF model with 20 components (representing 20 topics) was established. The generalized Kullback-Leibler divergence was used with the "beta_loss" keyword. This is equivalent to a probabilistic latent semantic indexing. Other definition of parameters were referred to ([REF](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)). Next, an ARIMA model was established based on the yearly counts of publications for a topic normalized by the total counts of publications in that year regardless of topics. A grid search cross validation (66% training vs. 34% testing data) was conducted to tune the best configuration of AR parameters, differences, and MA parameters. The root mean squared error was minimized in the cross-validations. Finally, using the tuned ARIMA models the normalized counts of publications were forecasted for each topic. The figure below shows the workflow of data preprocessing and topic modeling. 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/topic_modeling_workflow.png" alt="Figure 1" width="700"/>

---

### Results and Discussion

The figure below shows the total counts of publications each year, demonstrating that this research field is expanding. Note that 2021 just started when this repo was written.  

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/counts_of_publications.svg" alt="counts of pub" width="700"/>

Top 10 most popular journals have been identified. The most popular journal is Journal of Neurotrauma (IF of 4.056). Three of my research papers published there were selected as front cover pages. 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/top_10_most_pop_journals_over_30_years.svg" alt="top 10" width="700"/>

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/cover_pages_j_neurotrauma.png" alt="cover" width="700"/>

The results show five examples of topic development for illustration. First, for the topic of brain biomechanics, it is not as hot as it was in 1990s,  but is likely to maintain a stable development trend in the next five years. 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.2.png" alt="Figure 2" width="700"/>

Researchers become less interested in the intracranial pressure now compared with in 1990s. In the past, it was believed the pressure gradient could distort the brain and caused concussion ([King et al., 2013](https://smf.org/docs/articles/hic/King_IRCOBI_2003.pdf)), but more recently it was believed that the brain pressure which is induced by linear acceleration did not  induce brain deformation in TBI. This is because linear acceleration is not significantly correlated with brain deformation ([Ji et al., 2014](https://link.springer.com/article/10.1007%2Fs10237-014-0562-z)).  

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.3.png" alt="Figure 3" width="700"/>

The topic of head kinematics was very hot in 2000s. But more recently, the number of publications kept decreasing. This might be because the studies converged to a small and specific scope. In addition, head kinematics do not directly inform brain deformation. Another possible reason could be that the NMF model believed that some articles about head kinematics were related to sports concussion and assigned these articles to that topic. 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.4.png" alt="Figure 4" width="700"/>

Sports concussion is a very hot research topic currently, and will remain a hot topic in next five years. More and more recent research findings have opened a door for more research opportunities, e.g.,  ([Zhao et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28710533/)). In addition, the NFL foundation has opened more funding opportunities for researchers ([REF](https://www.nflfoundation.org/applications/grant_programs)). 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.5.png" alt="Figure 5" width="700"/>

Similar to sports concussion, blast injury has become a hot topic now, and is likely to remain hot in the future. One possible reason could be that the DoD has opened more funding programs than before ([REF](https://blastinjuryresearch.amedd.army.mil/index.cfm/brain_health_program/maximize_research)). More and more research articles were published towards the understanding of the biomechanical mechanisms behind blast injury, e.g., [Aravind et al., 2020](https://pubmed.ncbi.nlm.nih.gov/33013653/) and [Kim et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32614198/). 

<img src="https://github.com/weizhao-BME/metis-project4/blob/main/figures/fig.6.png" alt="Figure 6" width="700"/>

---

### Conclusions

In the context of traumatic brain injury, this analysis presented an approach for academic researchers to know the evolution of hot research topics that they are interested in. Therefore, I suggest running a topic modeling as proposed in this analysis,  clustering articles with identified research topics, choosing a hot topic, which will remain hot in the future, and reading the clustered articles before roaming in the sea of articles. 


# covid-19-challenge

In this repository, [Carlos Gomes](https://github.com/CarlosGomes98) and I work on the [Kaggle COVID-19 Open Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). The data can be downloaded from Kaggle. 

In short, the aim of the challenge is to attempt to find answers to multiple COVID-19 related questions in a corpus of biomedical articles published on the topic.

To replicate, clone this repository and create a conda environment with the necessary packages by running ```conda env create -f conda_env.yml```. Activate the environment with ```conda activate ml```.  Download the data from Kaggle and **place these files in a folder named 'data'**.


1) **data_preprocessing.ipynb** aggregates the articles into a single file and performs some pre-processing on the text.
2) **LDA.ipynb** performs Latent Dirichlet Allocation (LDA) topic modelling on the abstracts of the papers.
3) **LDA-answer-finding.ipynb** tries to answer some of the questions using the topics from LDA.
4) **Embeddings.ipynb** tries to answer the questions using different types of word embeddings.

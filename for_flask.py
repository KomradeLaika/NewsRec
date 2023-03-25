import numpy as np 
import pandas as pd 
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recomendation(idx, no_of_news_article):
    ds = pd.read_csv('export_cleaned_ds.csv', sep=';')
    tf = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.8, min_df=0.0, use_idf=True,
                         ngram_range=(1, 3))
    tfidf_matrix = tf.fit_transform(ds['cleaned_desc'])
    pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    results = {}
    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
        results[row['id']] = similar_items[1:]
    similarity_score = list(enumerate(cosine_similarities[idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:no_of_news_article + 1]

    output_df = pd.DataFrame(data=[i[1] for i in similarity_score], columns=['Similarity'])
    news_list = []
    dates_list = []
    titles_list = []
    links_list = []

    news_indices = [i[0] for i in similarity_score]
    for i in range(len(news_indices)):
        news_list.append(str(news_indices[i]))
        dates_list.append(ds['date'].iloc[news_indices[i]])
        titles_list.append(ds['title'].iloc[news_indices[i]])
        links_list.append(ds['link'].iloc[news_indices[i]])

    output_df['IDX'] = news_list
    output_df['Dates'] = dates_list
    output_df['Titles'] = titles_list
    output_df['Links'] = links_list
    return output_df

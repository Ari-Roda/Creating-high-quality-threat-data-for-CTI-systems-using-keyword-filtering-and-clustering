import pickle
import pandas as pd
import numpy
import re
import os
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score , davies_bouldin_score

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", -1)

stop = stopwords.words('english')
stop.extend(['like', 'would', 'use', 'x200b', 'get', 'know', 'one', 'using',  'also',   'want', 'could',  'something', 'need', 'see', 'used', 'anyone', 'even', 'etc', 'really'])

df = pd.read_csv("dataset", header=1 , names=['Body','Platform'])
df.drop(columns=['Platform'], inplace=True)

##########################include same preprocessing process used for specific dataset###########################################################################


d2v_model = Doc2Vec.load("doc2vec model")

clusters = 2
iterations = 100

kmeans_model = KMeans(n_clusters=clusters, init='k-means++', max_iter=iterations, random_state = 2) 
X = kmeans_model.fit(d2v_model.docvecs.vectors_docs)
l = kmeans_model.fit_predict(d2v_model.docvecs.vectors_docs)
labels = kmeans_model.labels_.tolist()

df["clusters"] = labels

cluster_list = []

for i in range(clusters):
	df_temp = df[df["clusters"]==i]
	cluster_words = Counter(" ".join(df_temp["Body"].str.lower()).split()).most_common(20)
	print ("cluster ",i)
	[cluster_list.append(x[0]) for x in cluster_words]
	print (cluster_list)
	print (" ")
	cluster_list.clear()

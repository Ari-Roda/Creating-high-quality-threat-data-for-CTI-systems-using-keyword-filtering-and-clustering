import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.models import phrases

df = pd.read_csv("dataset",header=1,names=["A","B","C","D"])

##########################use same preprocessing as specific dataset#######################################################################################################

bigrams = phrases.Phrases(sentences)

epoch_list = [50]
size_list = [10]

for x in epoch_list:
	for y in size_list:

		vec_size = y
		max_epochs = x
		mode = 0
		model = Word2Vec(size=vec_size, sg=mode, iter=max_epochs)
		model.build_vocab(bigrams[sentences])
		model.train(bigrams[sentences],total_examples=model.corpus_count,epochs=model.epochs)
		model.save("10vector50epoch_new_tweet_word2vec_model")

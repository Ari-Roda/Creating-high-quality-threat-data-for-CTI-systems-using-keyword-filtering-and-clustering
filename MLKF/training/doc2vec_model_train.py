import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

df = pd.read_csv("dataset",header=1,names=["A","B"])
stop = stopwords.words('english')

##########################use same preprocessing as specific dataset#######################################################################################################

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(df["A"])]

epoch_list = [50,100,200,300,400,500]
size_list = [100,200,400,500,100,2000]

for x in epoch_list:
	for y in size_list:

		vec_size = y
		max_epochs = x
		mode = 0
		subsampling = 1e-5

		model = Doc2Vec(vector_size=vec_size, dm=mode, sample=subsampling,epochs=max_epochs,min_count=1)
		model.build_vocab(tagged_data)
		model.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)
		model.save("clustering")
		print("Model Saved " +str(y)+"s_"+str(x))




import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from time import sleep
from nltk import word_tokenize 
from nltk.util import ngrams
import nltk as nltk
from gensim.models.word2vec import Word2Vec
import csv 

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", -1)
stop = stopwords.words('english')

df = pd.read_csv("choose_a_dataset",header=0,names=["A","B","C","D"]) # import dataset

modelw1v = Word2Vec.load("w2v_model") #import trained word2vec model for dataset

def make_ngram_list(item):
	unigrams = set(item.split(" "))
	token = nltk.word_tokenize(item)
	bigrams = list(ngrams(token, 2))
	bigrams = [" ".join(x) for x in bigrams]
	bigrams = set(bigrams)
	unigrams_and_bigrams = list(unigrams) + list(bigrams)
	return (unigrams_and_bigrams)

def writetocsv(entry):
	with open('fppf.csv','a') as f:
    		writer=csv.writer(f)
    		writer.writerow(entry["A"])

##############################################################################################################################################
#basic preprocessing
##############################################################################################################################################
df["A"] = df["A"].str.lower()
df['A'] = df['A'].replace(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', regex=True)
df['A'] = df['A'].replace(r'\B@\w+', '', regex=True)
df['A'] = df['A'].replace(r'[^a-zA-Z0-9]', ' ', regex=True)
df['A'] = df['A'].replace(r'\s+', ' ', regex=True)
df["A"] = df['A'].apply(lambda x: [snowball_stemmer.stem(item) for item in str(x).split() if item not in stop and len(item) > 2 and len(item) < 40])
df["A"] = df["A"].str.join(" ")
##############################################################################################################################################

reddit_broad_filter = "bootcamp"            #string version of main filter
reddit_broad_filter_list = ["bootcamp"]     #list version of main filter
reddit_double_meaning_words = ["bootcamp" ] #double meaning word filter for each/full dataset


df = df[df.A.str.contains(r"\b(bootcamp)\b")] #keep entries which contain term from main filter

double_meaning_words_filter = reddit_double_meaning_words
filter_list = reddit_broad_filter_list

for item in df["A"]:
	double_meaning_words_set = set(make_ngram_list(item)).intersection(double_meaning_words_filter)
	original_words_set = set(make_ngram_list(item)).intersection(filter_list) # check again if it contains word from filter list if it does go through
	if bool(double_meaning_words_set) == True: #does the entry contain a word from double meaning words list
		if len(list(double_meaning_words_set)) > 1 or len(list(original_words_set)) > 1: #if it contains more than 1 word from double meaning word list assume its related and let it through. otherwise process
			continue
		else:
			most_similar_words_list1 = modelw1v.wv.most_similar(positive=[list(double_meaning_words_set)[0].replace(" ","_")], topn=100)
			similar_words = [x[0].replace("_"," ") for x in most_similar_words_list1]
			#updated_associated_words = set(similar_words) - double_meaning_words_set #get the associated words list minus term
			tweet_minus_word = set(make_ngram_list(item)) - double_meaning_words_set #remove that term from the tweet also
			if bool(tweet_minus_word.intersection(similar_words)) == True: # check if the tweets other terms match our associated words list, if yes there fine, else remove them
				continue 										   													   
			else:
				fal_po = df.loc[df['A'] == item] #specific entry that have a term from double meaning list but dont contain any associated words in them
				df = df[~df.A.str.contains(item)] # remove that entry from datasets
				writetocsv(fal_po)
	elif bool(double_meaning_words_set) == False and bool(original_words_set) == False:
		fal_po = df.loc[df['A'] == item] #specific entry that have a term from double meaning list but dont contain any associated words in them
		df = df[~df.A.str.contains(item)] # remove that entry from datasets
		writetocsv(fal_po)
	else:
		continue

df.to_csv("fina.csv", index=False)




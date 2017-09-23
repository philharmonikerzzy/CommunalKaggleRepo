import glob, os
import cntk as C
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse.csr

def BuildCorpusAndID(train_text_path, test_text_path):
	
	corpus = []
	train_corpus = []
	test_corpus = []
	TrainID = []
	TestID = []
	with open(train_text_path, encoding="utf-8") as f:
		isFirstLine = True
		for line in f:
			if isFirstLine:
				isFirstLine = False
			else:
				corpus.append(line.strip().split("||")[1])
				train_corpus.append(line.strip().split("||")[1])
				TrainID.append(line.strip().split("||")[0])
	with open(test_text_path, encoding = "utf-8") as g:
		isFirstLine = True
		for line in g:
			if isFirstLine:
				isFirstLine = false
			else:
				corpus.append(line.strip().split("||")[1])
				test_corpus.append(line.strip().split("||")[1])
				TestID.append(line.strip().split("||")[0])
	return train_corpus, test_corpus, corpus
			


def GetCountVectorizedCorpus(corpus):
	
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
					max_features = 100000,
					stop_words = 'english')
	
	return tf_vectorizer.fit_transform(corpus)


def GetCountVectorizedData(text_data, corpus):
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features = 100000, stop_words = 'english')
	
	tf_vectorizer.fit(corpus)
	return tf_vectorizer.transform(text_data)


def HandleCategoricalFeatures(dataframe):
	
	for col in (dataframe.select_dtypes(include=['object'])).columns:
		dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[col])],axis=1)
		dataframe = dataframe.drop(col,axis=1)
	return dataframe


if __name__ == "__main__":
	
	train_text_path = "../raw_data/training_text"
	test_text_path = "../raw_data/test_text"
	train_variant_path = "../raw_data/training_variant.csv"
	test_variant_path = "../raw_data/test_variant.csv"
	train_corpus, test_corpus, corpus = BuildCorpusAndID(train_text_path, test_text_path)
	train_tfidf = GetCountVectorizedData(train_corpus, corpus)
	test_tfidf = GetCountVectorizedData(test_corpus, corpus)
	trainvardf = pd.read_csv(train_variant_path)
	
	trainvardf = HandleCategoricalFeatures(trainvardf)
	
	testvardf = pd.read_csv(test_variant_path)
	#after we have the tfidf/count based vector of the text, we use some type of embedding,
	#could directly use the embedding 
	labels=traindf['Class']
	traindf=traindf.drop('Class',axis=1)
	labels=
	x=C.sequence.input_variable(vocab_dim)
	y=C.sequence.input_variable(label_dim)
	
	
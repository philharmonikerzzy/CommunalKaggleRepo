import cntk as C
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing
import scipy
import numpy as np

def convertNumpyType(nparray, new_type):

	y=nparray.view(new_type)
	y[:]= nparray
	return y

def combineTextDF(dataframe, tf_idfdata):
	
	return scipy.sparse.hstack((np.asarray(dataframe).astype(float),tf_idfdata.astype(float)))

def createOneHotLabels(multiclasslabel):
	
	lb=preprocessing.LabelBinarizer()
	return lb.fit_transform(multiclasslabel)

def createReader(dataarray, labels):

	return C.io.MinibatchSourceFromData(dict(features=scipy.sparse.csr_matrix(dataarray),label=scipy.sparse.csr_matrix(labels)))

def createReaderDoubleLabels(dataarray, labels):

	return C.io.MinibatchSourceFromData(dict(features=scipy.sparse.csr_matrix(dataarray),label=labels))	
	
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
				isFirstLine = False
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

def GetTFIDF_train_test(traintext, testtext, corpus):
		tf_idf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,2), stop_words='english')
		tf_idf_vectorizer.fit(corpus)
		return tf_idf_vectorizer.transform(traintext), tf_idf_vectorizer.transform(testtext)
	
def GetCountVectorizedData(text_data, corpus):
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features = 100000, stop_words = 'english')
	
	tf_vectorizer.fit(corpus)
	return tf_vectorizer.transform(text_data)


def HandleCategoricalFeatures(dataframe):
	
	for col in (dataframe.select_dtypes(include=['object'])).columns:
		dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[col])],axis=1)
		dataframe = dataframe.drop(col,axis=1)
	return dataframe

import sys
sys.path.insert(0, '../pylib/')
import sklearn 
from sklearn import cross_validation
import pandas as pd
import numpy as np
import scipy
import PreprocessTextDataLib as lib
import InitiateCNTKLib as clib
import xgboost as xgb
import xgblib 
 

train_text_path = "../raw_data/CancerTreatment/training_text"
test_text_path = "../raw_data/CancerTreatment/test_text"
train_variant_path = "../raw_data/CancerTreatment/training_variants.csv"
test_variant_path = "../raw_data/CancerTreatment/test_variants.csv"
traincp, testcp, cp = lib.BuildCorpusAndID(train_text_path, test_text_path)

print ("successfully created text corpi for all data")

trainvec, testvec = lib.GetTFIDF_train_test(traincp,testcp, cp)

trainvec = trainvec.astype(np.float32)
testvec = testvec.astype(np.float32)

print ("successfully generated tfidf data for training text")

traindf = pd.read_csv(train_variant_path)
testdf = pd.read_csv(test_variant_path)
testID = testdf['ID']

label = traindf['Class']
label = label-1

labels = lib.createOneHotLabels(traindf['Class']).astype(float)

traindf = traindf.drop(['Class', 'ID'], axis=1)
testdf = testdf.drop('ID',axis=1)
combineddf = traindf.append(testdf)

combineddf = lib.HandleCategoricalFeatures(combineddf)

traindf = combineddf.iloc[:traindf.shape[0]]
testdf = combineddf.iloc[traindf.shape[0]:]

print ("successfully processed training variant data frame")

train = lib.combineTextDF(traindf, trainvec)
train = train.tocsr()
test = (lib.combineTextDF(testdf, testvec)).tocsr()

print ("successfully combined training variant and text")


train_test_split = cross_validation.StratifiedShuffleSplit(label, n_iter=1,test_size=0.1)

for train_idx,test_idx in train_test_split:
		X_train,X_test=train[train_idx,:],train[test_idx,:]
		trainlabels,testlabels = label.iloc[train_idx],label.iloc[test_idx]
		

		
pred = xgblib.xgboost_pred(X_train,trainlabels,X_test,testlabels,test)

pred = pd.DataFrame(pred, index = testID)
pred.to_csv("../xgb_baseline_tfidf.csv",index=False)




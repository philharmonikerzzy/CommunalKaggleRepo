import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb
import time 
import math

def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
     
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())



def xgboost_pred(train,labels,test,test_labels,final_test):
    params = {}
    params["objective"] = "reg:linear"
    params["eval_metric"]="rmse"
    params["eta"] = 0.02 #0.02 
    params["min_child_weight"] = 6
    params["subsample"] = 0.9 #although somehow values between 0.25 to 0.75 is recommended by Hastie
    params["colsample_bytree"] = 0.7
    #params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 8
    params["alpha"]=0.05
    plst = list(params.items())

    num_rounds = 20000
    xgtest = xgb.DMatrix(final_test)

    xgtrain = xgb.DMatrix(train, label=labels)
    xgval = xgb.DMatrix(test, label=test_labels)
 
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30)

    valpred = model.predict(xgval,ntree_limit=model.best_iteration)
    print "current rmsle is " + str(rmsle(np.array(valpred), np.array(test_labels)))

    print 'ready to generate test data'


    return model.predict(xgtest,ntree_limit=model.best_iteration)



def HandleNanData(dataFrame, categoricalFeatureList):
	
	nanColumns = dataFrame.isnull().any()
	nanColumnidx = nanColumns.index[nanColumns]
	for nancol in nanColumnidx:
		if nancol in categoricalFeatureList:
			dataFrame[nancol].fillna(dataFrame[nancol].mode().ix[0], inplace=True)
		else:
			dataFrame[nancol].fillna(-2,inplace=True)
	
	nanColumns = dataFrame.isnull().any()
	print str(len(nanColumns[nanColumns]))+" is still NaN"
	return dataFrame



def HandleCategoricalData(dataFrame, categoricalFeatureList):
	
	for catfeat in categoricalFeatureList:
		print catfeat + " includes "+str(len(dataFrame[catfeat].unique()))+" different categorical values"
		dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame[catfeat],      sparse=False)],axis=1)
		dataFrame = dataFrame.drop([catfeat],axis=1)
	
	return dataFrame		

def SplitTrainValData(data, label, val_size=0.2):
	stratifiedShuffle = cross_validation.ShuffleSplit(len(data), n_iter=1,test_size=val_size, random_state=0)
	for trainidx, validx in stratifiedShuffle:
		train, val = data.iloc[trainidx],data.iloc[validx]
		trainlabel, vallabel = label.iloc[trainidx], label.iloc[validx]
	return train, val, trainlabel, vallabel



if __name__=="__main__":

	traindf = pd.read_csv("../raw_data/RussianRealEstate/train.csv")

	testdf =pd.read_csv("../raw_data/RussianRealEstate/test.csv")

	macro = pd.read_csv("../raw_data/RussianRealEstate/macro.csv")

	trainLabel = traindf['price_doc']

	
	traindf.drop(['price_doc'], axis=1)

	unifiedData = traindf.append(testdf)


	unifiedData = pd.merge(unifiedData, macro, how = 'left', on=['timestamp'])
	
	unifiedData['year']=pd.DatetimeIndex(unifiedData['timestamp']).year
	unifiedData['month']=pd.DatetimeIndex(unifiedData['timestamp']).month
	unifiedData['dayofweek']=pd.DatetimeIndex(unifiedData['timestamp']).dayofweek

	unifiedData=unifiedData.drop(['timestamp'],axis=1)
	categoricalFeatures = list(unifiedData.select_dtypes(exclude=['float64','int64']))
	
	unifiedData = HandleNanData(unifiedData, categoricalFeatures)
	

	unifiedData = HandleCategoricalData(unifiedData,categoricalFeatures)

	traindata = unifiedData.iloc[:len(traindf)-1]
	testdata = unifiedData.iloc[len(traindf):]

	trainData, valData, trainlabel, vallabel = SplitTrainValData(traindata,trainLabel)


	trainData = np.array(trainData)
	valData = np.array(valData)
	testData = np.array(testdata)
	#trainlabel = np.log1p(trainlabel)
	#vallabel = np.log1p(vallabel)

	pred=xgboost_pred(trainData, trainlabel, valData, vallabel, testData)
	OutputPredictions = pd.DataFrame(pred, index = testdata['id'])
	OutputPredictions.columns = [ 'price_doc']
	OutputPredictions.to_csv('submission'+ time.strftime("%Y%m%d-%H%M")+".csv")

	








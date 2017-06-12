import pandas as pd
import numpy as np

traindf = pd.read_csv("../raw_data/train.csv")

testdf =pd.read_csv("../raw_data/test.csv")

macro = pd.read_Csv("../raw_data/macro.csv")


categoricalTrainFeatures = list(traindf.select_dtypes(exclude=['float64','int64']))

categoricalTestFeatures = list(testdf.select_dtypes(exclude=['float64','int64']))

traindf = HandleNanData(traindf, categoricalTrainFeatures)











def HandleNanData(dataFrame, categoricalFeatureList):
	
	nanColumns = dataFrame.isnull().any()
	for nancol in nanColumns:
		if nancol in categoricalFeatureList:
			dataFrame[nancol].fillna(dataFrame[nancol].mode().ix[0], inplace=True)
		else:
			dataFrame[nancol].fillna(-2,inplace=True)
	return dataFrame	
		


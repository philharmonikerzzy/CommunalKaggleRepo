import xgboost as xgb

def xgboost_pred(train,labels,test,test_labels,final_test):
    params = {}
    params["objective"] = "multi:softprob"
    params["eval_metric"]="mlogloss"
    params["eta"] = 0.01 #0.02 
    params["min_child_weight"] = 6
    params["subsample"] = 0.9 #although somehow values between 0.25 to 0.75 is recommended by Hastie
    params["colsample_bytree"] = 0.9
    params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 8
    params["num_class"]=9
    
    plst = list(params.items())


    num_rounds = 20000
    xgtest = xgb.DMatrix(final_test)

    xgtrain = xgb.DMatrix(train, label=labels)
    xgval = xgb.DMatrix(test, label=test_labels)
 
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30)
    #create a train and validation dmatrices 

    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
#     train = train[::-1,:]
#     labels = np.log(labels[::-1])
# 
    print ('ready to generate test data')

    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    return model.predict(xgtest,ntree_limit=model.best_iteration)
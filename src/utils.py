import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics  import r2_score

from src.exception import customException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise customException(e, sys)
        

def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))): #go trough each and evry models

            model=list(models.values())[i] # I'm getting each  model

            model.fit(x_train, y_train) # I do the fit on my x_train and y_train 

            y_train_pred = model.predict(x_train) # do the prediction on x_train

            y_test_pred = model.predict(x_test) # do the prediction on x_test

            train_model_score=r2_score(y_train, y_train_pred) # compute the r2_score for the train
            #print("y_test:"+str(len(y_test)))
            #print("y_test_pred:"+str(len(y_test_pred)))
            test_model_score= r2_score(y_test, y_test_pred)  # compute the r2_score for the test
            #print("test_model_score:"+str(test_model_score))
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise customException(e, sys)
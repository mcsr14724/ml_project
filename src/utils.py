import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        logging.info("dumping is intiated")
        with open(file_path,'wb') as f:
            dill.dump(obj,f)  
        logging.info('dumping is completed')     
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model_name=list(models.keys())[i]
            model=list(models.values())[i]
            param=params[model_name]

            if param:
                randomcv=RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param,
                    scoring='r2',
                    n_iter=50,
                    n_jobs=-1,
                    cv=5,
                    verbose=0,
                    random_state=1
                )
                randomcv.fit(x_train,y_train)
                y_train_pred=randomcv.predict(x_train)
                y_test_pred=randomcv.predict(x_test)
            else:
                model.fit(x_train,y_train)
                y_train_pred=model.predict(x_train)
                y_test_pred=model.predict(x_test)

            train_score=r2_score(y_train,y_train_pred)
            test_score=r2_score(y_test,y_test_pred)

            report[model_name]=test_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
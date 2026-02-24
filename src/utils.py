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
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    report = {}
    trained_models = {}

    for model_name, model in models.items():

        param = params[model_name]

        if param:
            randomcv = RandomizedSearchCV(
                estimator=model,
                param_distributions=param,
                scoring='r2',
                n_iter=50,
                n_jobs=-1,
                cv=5,
                random_state=1
            )

            randomcv.fit(x_train, y_train)
            best_model = randomcv.best_estimator_

        else:
            model.fit(x_train, y_train)
            best_model = model

        y_test_pred = best_model.predict(x_test)
        test_score = r2_score(y_test, y_test_pred)

        report[model_name] = test_score
        trained_models[model_name] = best_model

    return report, trained_models
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
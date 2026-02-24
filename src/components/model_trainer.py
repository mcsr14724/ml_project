import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info('splitting input and output')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'linear regression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'SVR':SVR(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=0,train_dir='logs\catboost_info')
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('no best model is found')
            
            logging.info('found best model')

            save_obj(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return best_model_score
        except Exception as e:
            raise CustomException(e,sys)
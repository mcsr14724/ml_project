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

            params = {
                "linear regression": {},

                "Lasso": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    "max_iter": [1000, 5000, 10000],
                    "tol": [1e-4, 1e-3, 1e-2]
                },

                "Ridge": {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["auto", "svd", "lsqr", "saga"],
                    "tol": [1e-4, 1e-3, 1e-2]
                },

                "ElasticNet": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "max_iter": [1000, 5000],
                    "tol": [1e-4, 1e-3]
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [3,5,7,9,11,15],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"],
                    "p": [1,2]
                },

                "SVR": [

                    {
                        "kernel": ["linear"],
                        "C": [0.1, 1, 10, 100],
                        "epsilon": [0.01, 0.1, 0.2, 0.5]
                    },

                    {
                        "kernel": ["rbf"],
                        "C": [0.1, 1, 10, 100],
                        "epsilon": [0.01, 0.1, 0.2, 0.5],
                        "gamma": ["scale", "auto"]
                    },

                    {
                        "kernel": ["poly"],
                        "C": [0.1, 1, 10],
                        "epsilon": [0.01, 0.1, 0.2],
                        "gamma": ["scale", "auto"],
                        "degree": [2,3,4]
                    }
                ],

                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,4],
                    "max_features": ["sqrt", "log2", None]
                },

                "RandomForestRegressor": {
                    "n_estimators": [100,200,300],
                    "max_depth": [None,10,20,30],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,4],
                    "max_features": ["sqrt","log2"],
                    "bootstrap": [True, False]
                },

                "AdaBoostRegressor": {
                    "n_estimators": [50,100,200,300],
                    "learning_rate": [0.01,0.05,0.1,1],
                    "loss": ["linear","square","exponential"]
                },

                "GradientBoostingRegressor": {
                    "n_estimators": [100,200,300],
                    "learning_rate": [0.01,0.05,0.1],
                    "max_depth": [3,5,7],
                    "subsample": [0.8,1.0],
                    "min_samples_split": [2,5],
                    "max_features": ["sqrt","log2",None]
                },

                "XGBRegressor": {
                    "n_estimators": [100,200,300],
                    "learning_rate": [0.01,0.05,0.1],
                    "max_depth": [3,5,7,10],
                    "subsample": [0.7,0.8,1],
                    "colsample_bytree": [0.7,0.8,1],
                    "gamma": [0,0.1,0.3],
                    "reg_alpha": [0,0.01,0.1],
                    "reg_lambda": [1,1.5,2],
                    "min_child_weight": [1,3,5]
                },

                "CatBoostRegressor": {
                    "iterations": [100,200,300],
                    "learning_rate": [0.01,0.05,0.1],
                    "depth": [4,6,8,10],
                    "l2_leaf_reg": [1,3,5,7,9],
                    "bagging_temperature": [0,1,3,5],
                    "random_strength": [1,5,10]
                }
            }

            model_report,trained_models = evaluate_model(x_train, y_train, x_test, y_test, models, params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=trained_models[best_model_name]

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
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsibe for data transformation
        '''
        try:
            num_cols=[
                'reading_score',
                'writing_score'
            ]

            cat_cols=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            logging.info('numerical pipeline is intiated')
            num_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='median')),
                    ('Sclaer',StandardScaler())
                ]
            )
            logging.info('numerical pipeline is completed')

            logging.info('categorical pipeline is intiated')
            cat_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(drop='first'))
                ]
            )
            logging.info('categorical pipeline is completed')

            logging.info('column transfomer is intiated')
            preprocessor=ColumnTransformer(
                transformers=[
                    ('numerical_pipeline',num_pipeline,num_cols),
                    ('categorical_pipeline',cat_pipeline,cat_cols)   
                ],
                remainder='passthrough'
            )
            logging.info('column transformer is completed')
        except Exception as e:
            raise CustomException(e,sys)
        
        return preprocessor
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)
            logging.info('read train and test data completed')

            logging.info('obtaining preprocessing object')
            preprocess_obj=self.get_data_transformer_object()

            target_col='math_score'
            num_cols=[
                'reading_score',
                'writing_score'
            ]

            cat_cols=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            x_train=df_train.drop(target_col,axis=1)
            y_train=df_train[target_col]
            x_test=df_test.drop(target_col,axis=1)
            y_test=df_test[target_col]

            logging.info('applying preprocessing object on training and testing dataframes')
            x_train_arr=preprocess_obj.fit_transform(x_train)
            x_test_arr=preprocess_obj.transform(x_test)

            train_arr=np.c_[x_train_arr,np.array(y_train)]
            test_arr=np.c_[x_test_arr,np.array(y_test)]

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_path,obj=preprocess_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)
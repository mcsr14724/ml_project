import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
import dill

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
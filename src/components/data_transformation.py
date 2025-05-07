import os
import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logging import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass

class DataTransformationConfig:
    preprocessor_file_path=os.path.join("Artifacts", "Preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformation()
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation Initiated!!")
            nums_cols=['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed', 'rainfall']
            
            logging,info("PipeLine Initiated!!")
            
            nums_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            preprocessor=ColumnTransformer([
                ('num pipeline', nums_pipeline, nums_cols)
            ] 
            )
            
            return preprocessor
        except Exception as e:
            logging.info("Exception occured while data transformation")
            raise CustomException(e,sys)
    
    
    
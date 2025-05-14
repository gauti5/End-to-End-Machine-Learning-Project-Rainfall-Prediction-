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
from src.utils import save_object

from dataclasses import dataclass

@dataclass

class DataTransformationConfig:
    preprocessor_file_path=os.path.join("Artifacts", "Preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation Initiated!!")
            nums_cols=['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']
            
            logging.info("PipeLine Initiated!!")
            
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
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preproccesor_obj=self.get_data_transformation()
            
            logging.info("read the train and test data")
            
            logging.info(f"train dataframe head : \n{train_df.head(5).to_string()}")
            logging.info(f"test dataframe head : \n{test_df.head(5).to_string()}")
            
            input_features_train_df=train_df.drop('rainfall', axis=1)
            target_features_train_df=train_df['rainfall']
            
            input_features_test_df=test_df.drop('rainfall', axis=1)
            target_features_test_df=test_df['rainfall']
            
            input_features_train_arr=preproccesor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preproccesor_obj.transform(input_features_test_df)
            
            logging.info("applying the preprocessor object in training and testing data")
            
            train_arr=np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr=np.c_[input_features_test_arr, np.array(target_features_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preproccesor_obj
            )
            logging.info("Preprocessor pickle file saved!!!")
            
            return(
                train_arr, test_arr
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
            
    
    

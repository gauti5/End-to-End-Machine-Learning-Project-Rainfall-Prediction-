import os
import sys
from pathlib import Path
import pandas as pd

from src.logging import logging 
from src.exception import CustomException

from src.utils import load_object

class predict_pipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            preprocessor_path=os.path.join('Artifacts', 'Preprocessor.pkl')
            model_path=os.path.join('Artifacts', 'Model.pkl')
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            data_scaled=preprocessor.transform(features)
            
            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self,
                 id:int,
                 day:int,
                 pressure:int,
                 maxtemp:float,
                 temparature:float,
                 mintemp:float,
                 dewpoint:float,
                 humidity:float,
                 cloud:float,
                 sunshine:float,
                 winddirection:float,
                 windspeed:float
                 ):
        self.id=id
        self.day=day
        self.pressure=pressure
        self.maxtemp=maxtemp
        self.temparature=temparature
        self.mintemp=mintemp
        self.dewpoint=dewpoint
        self.humidity=humidity
        self.cloud=cloud
        self.sunshine=sunshine
        self.winddirection=winddirection
        self.windspeed=windspeed
        
    def get_data_as_a_dataframe(self):
        try:
            custom_data_input_dict={
                'id':[self.id],
                'day':[self.day],
                'pressure':[self.pressure],
                'maxtemp':[self.maxtemp],
                'temparature':[self.temparature],
                'mintemp':[self.mintemp],
                'dewpoint':[self.dewpoint],
                'humidity':[self.humidity],
                'cloud':[self.cloud],
                'sunshine':[self.sunshine],
                'winddirection':[self.winddirection],
                'windspeed':[self.windspeed]
            }
            df=pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            logging.info("Exception Occured during the prediction piepline")
            raise CustomException(e,sys)
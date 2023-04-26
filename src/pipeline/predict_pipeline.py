import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Airline: str,
        Source: str,
        Destination: str,
        Total_Stops: int,
        Day: int,
        Month: int,
        Dep_hour: int,
        Dep_minute: int,
        Duration_hour: int,
        Duration_minute: int):

        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Total_Stops = Total_Stops
        self.Day = Day
        self.Month = Month
        self.Dep_hour = Dep_hour
        self.Dep_minute = Dep_minute
        self.Duration_hour = Duration_hour
        self.Duration_minute = Duration_minute

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Airline": [self.Airline],
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Total_Stops": [self.Total_Stops],
                "Day": [self.Day],
                "Month": [self.Month],
                "Dep_hour": [self.Dep_hour],
                "Dep_minute": [self.Dep_minute],
                "Duration_hour": [self.Duration_hour],
                "Duration_minute": [self.Duration_minute]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
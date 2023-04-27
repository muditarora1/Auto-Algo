import os
import sys
from src.logger import logging
from src.exception import CustomException 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_Transformation import DataTransformation
from src.components.data_Transformation import DataTransformationConfig
from src.utils import read_data

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#Config will give the path of any inputs needed in this file
@dataclass                                              # dataclass is used when u want to make class variable directly (without init)
class DataIngestionConfig:                              # For providing all the input thing for data ingestion
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()    #the above three paths will be saved in this class variable

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #df=pd.read_csv('Notebook/raw_clean_data.csv')       #Read the file
            df = read_data()
            print(df)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')
            train_set,test_set = train_test_split(df,test_size=.2,random_state=5)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print('\n',modeltrainer.initiate_model_trainer(train_arr,test_arr))


#set PYTHONPATH=%PYTHONPATH%;add_your_path(C:\Users\Admin\Desktop\Ineuron assignments\Project\Flight_Price_ML_project)

import os
import sys

import mysql.connector as conn
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging

def read_data():
    try:
        mydb = conn.connect(host="localhost",user="root",passwd="0000")
        cursor=mydb.cursor()
        cursor.execute("use ml_project")
        cursor.execute("Select * from flight")
        df = pd.DataFrame(cursor.fetchall(), columns=['sno','Airline','Source','Destination','Total_Stops','Price','Day','Month','Dep_hour','Dep_minute','Duration_hour','Duration_minute'],)
        return df
    
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            #gs = GridSearchCV(model,para,cv=5)
            gs = RandomizedSearchCV(model,para,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            logging.info(f"Model Name: {list(models.keys())[i]}  Test score: {test_model_score}  Train score: {train_model_score}")
            logging.info(f"Best Parameters: {gs.best_params_}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
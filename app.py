from flask import Flask,request,render_template
from flask_cors import cross_origin
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_minute = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        Duration_hour = abs(Arrival_hour - Dep_hour)
        Duration_minute = abs(Arrival_min - Dep_minute)

        # Total Stops
        Total_Stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        # AIR ASIA = 0 (not in column)
        Airline=request.form['airline']

        # Source
        # Banglore = 0 (not in column)
        Source = request.form["Source"]

        # Destination
        # Banglore = 0 (not in column)
        Destination = request.form["Destination"]


        data=CustomData(Airline,Source,Destination,Total_Stops,Day,Month,Dep_hour,Dep_minute,Duration_hour,Duration_minute)
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        output=round(results[0],2)
        print(output)
        return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)   
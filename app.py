from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os 

app = Flask(__name__)

file_path = os.path.abspath("E:/My_Project/Cellula Technologies/First Task/Booking EDA 2/Random_model.pkl")
with open(file_path, 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        lead_time = int(request.form['lead_time'])
        average_price = float(request.form['average_price'])
        special_requests = int(request.form['special_requests'])
        month = int(request.form['Month'])
        day_of_week = int(request.form['Day_of_week'])
        days_of_reservation = int(request.form['Days_of_reservation'])
        market_segment_type_online = int(request.form['market_segment_type_Online'])
        
        # Preprocess input data
        features = preprocess_input(lead_time, average_price, special_requests, month, day_of_week, days_of_reservation, market_segment_type_online)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return prediction result
        result = 'Booking should be canceled' if prediction == 1 else 'Booking should not be canceled'
        return render_template('result.html', result=result)
    except:
        return render_template('error.html')


def preprocess_input(lead_time, average_price, special_requests, month, day_of_week, days_of_reservation, market_segment_type_online):
    features = np.array([lead_time, average_price, special_requests, month, day_of_week, days_of_reservation, market_segment_type_online]).reshape(1, -1)
    return features

if __name__ == '__main__':
    app.run(debug=True)

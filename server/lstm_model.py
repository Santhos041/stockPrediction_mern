import time
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Preload API Key and Model
API_KEY = 'GCCTOW8QJYH58I3Q'
MODEL_FILE = "stock_model.h5"


def fetch_stock_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    return data


def preprocess_data(data):
    data = data.iloc[::-1]  # Reverse the data for chronological order
    target = data['4. close']
    features = data.drop(columns=['4. close', '5. volume'])
    return features, target


def train_model(features, target, epochs=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    X = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, target.values, test_size=0.2, random_state=42)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1)
    model.save(MODEL_FILE)

    return model, X_test, y_test


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request data
        data = request.json
        symbol = data.get("symbol")
        days = int(data.get("days", 1))  # Default to 1 day if not provided

        # Fetch and preprocess data
        stock_data = fetch_stock_data(symbol)
        if stock_data.empty:
            raise ValueError("Stock data is empty!")

        features, target = preprocess_data(stock_data)

        # Train the model
        model, X_test, y_test = train_model(features, target, epochs=5)

        # Scale and reshape features for predictions
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        X_test_scaled = np.reshape(scaled_features[-len(y_test):], (len(y_test), 1, features.shape[1]))

        # Predict for the test set
        y_pred = model.predict(X_test_scaled).flatten()

        # Plot actual vs predicted prices
        fig, ax = plt.subplots()
        ax.plot(range(len(y_test)), y_test, label="Actual Prices", color="blue")
        ax.plot(range(len(y_pred)), y_pred, label="Predicted Prices", color="red")
        ax.set_title(f"Actual vs Predicted Prices for {symbol}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()

        # Save the plot to BytesIO
        img_io_actual_vs_predicted = io.BytesIO()
        plt.savefig(img_io_actual_vs_predicted, format='png')
        img_io_actual_vs_predicted.seek(0)
        actual_vs_predicted_graph = base64.b64encode(img_io_actual_vs_predicted.getvalue()).decode('utf-8')
        img_io_actual_vs_predicted.close()

        # Prepare future predictions
        X_new = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))
        predictions = []
        for _ in range(days):
            next_prediction = model.predict(X_new[-1].reshape(1, 1, -1))
            predictions.append(float(next_prediction[0][0]))
            new_row = np.append(X_new[-1][0][1:], next_prediction[0][0])
            X_new = np.vstack([X_new, new_row.reshape(1, 1, -1)])

        # Plot predictions for future days
        fig, ax = plt.subplots()
        ax.plot(range(1, days + 1), predictions, label='Predicted Prices', color='red', marker='o')
        ax.set_title(f"{days}-Day Prediction for {symbol}")
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Predicted Price")
        ax.legend()

        # Save the future predictions plot
        img_io_predictions = io.BytesIO()
        plt.savefig(img_io_predictions, format='png')
        img_io_predictions.seek(0)
        predictions_graph = base64.b64encode(img_io_predictions.getvalue()).decode('utf-8')
        img_io_predictions.close()

        return jsonify({
            "predictions": predictions,
            "predictions_graph": predictions_graph,
            "actual_vs_predicted_graph": actual_vs_predicted_graph
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta

@st.cache_resource
def load_resources():
    model = load_model('nuclear_stock_model.h5')
    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    return model, scalers

model, scalers = load_resources()

st.title("Nuclear Energy Stock Price Predictor")
st.markdown("""
This app predicts future prices of nuclear energy stocks using a deep learning model trained on historical data and uranium prices.
""")
st.markdown("""
Disclaimer: This is for educative purposes, use at your own risk.
""")

def create_sequence(input_data, scalers, features, sequence_length=60):
    scaled_data = []
    for feature in features:
        scaler = scalers[feature]
        scaled_value = scaler.transform(np.array(input_data[feature]).reshape(-1, 1))[0, 0]
        scaled_data.append(scaled_value)
    
    sequence = np.array([scaled_data])
    sequence = np.repeat(sequence, sequence_length, axis=0)
    return sequence.reshape(1, sequence_length, len(features))

def predict_future_streamlit(model, last_sequence, scalers, features, days_to_predict=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        next_day_close_pred = model.predict(current_sequence)[0, 0]
        future_predictions.append(next_day_close_pred)
        
        new_row = np.zeros(len(features))
        
        close_idx = features.index('Close')
        open_idx = features.index('Open')
        high_idx = features.index('High')
        low_idx = features.index('Low')
        volume_idx = features.index('Volume')
        uranium_price_idx = features.index('UraniumPrice')
        
        new_row[close_idx] = next_day_close_pred
        new_row[open_idx] = current_sequence[0, -1, close_idx]
        new_row[high_idx] = next_day_close_pred * 1.01
        new_row[low_idx] = next_day_close_pred * 0.99
        new_row[volume_idx] = current_sequence[0, -1, volume_idx]
        new_row[uranium_price_idx] = current_sequence[0, -1, uranium_price_idx]
        
        current_sequence = np.append(current_sequence[:, 1:, :], [new_row.reshape(1, -1)], axis=1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scalers['Close'].inverse_transform(future_predictions)
    
    return future_predictions

st.sidebar.header("User Input Parameters")

def get_user_input():
    default_values = {
        'Open': 50.0,
        'High': 51.0,
        'Low': 49.0,
        'Close': 50.5,
        'Volume': 1000000,
        'UraniumPrice': 50.0
    }
    
    st.sidebar.subheader("Current Market Data")
    open_price = st.sidebar.number_input("Open Price", value=default_values['Open'], min_value=0.0, step=0.1)
    high = st.sidebar.number_input("Daily High", value=default_values['High'], min_value=0.0, step=0.1)
    low = st.sidebar.number_input("Daily Low", value=default_values['Low'], min_value=0.0, step=0.1)
    close = st.sidebar.number_input("Closing Price", value=default_values['Close'], min_value=0.0, step=0.1)
    volume = st.sidebar.number_input("Volume", value=default_values['Volume'], min_value=0, step=1000)
    uranium_price = st.sidebar.number_input("Uranium Price", value=default_values['UraniumPrice'], min_value=0.0, step=0.1)
    
    days_to_predict = st.sidebar.slider("Days to Predict", 1, 60, 30)
    
    return {
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
        'UraniumPrice': uranium_price
    }, days_to_predict

def main():
    st.subheader("Current Market Data")
    
    input_data, days_to_predict = get_user_input()
    
    st.write("### Input Parameters:")
    st.write(pd.DataFrame([input_data]))
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'UraniumPrice']
    sequence = create_sequence(input_data, scalers, features)
    
    if st.button("Predict Future Prices"):
        with st.spinner('Making predictions...'):
            predictions = predict_future_streamlit(model, sequence, scalers, features, days_to_predict)
            
            last_date = datetime.now()
            future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
            
            results = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predictions.flatten()
            })
            
            st.write("### Prediction Results:")
            st.write(results)
            
            st.write("### Price Prediction Chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(results['Date'], results['Predicted Price'], marker='o')
            ax.set_title(f"Nuclear Stock Price Prediction for Next {days_to_predict} Days")
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Price ($)")
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.write("### Prediction Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Starting Price", f"${input_data['Close']:.2f}")
                st.metric("Predicted High", f"${np.max(predictions):.2f}")
                
            with col2:
                st.metric("Predicted End Price", f"${predictions[-1][0]:.2f}")
                st.metric("Predicted Low", f"${np.min(predictions):.2f}")
            
            price_change = predictions[-1][0] - input_data['Close']
            pct_change = (price_change / input_data['Close']) * 100
            st.metric("Overall Change", 
                     f"${price_change:.2f}", 
                     f"{pct_change:.2f}%")

if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import joblib
from datetime import datetime, timedelta
import yfinance as yf

st.set_page_config(page_title="Nuclear Stock Predictor", layout="wide")

# Load assets with caching
@st.cache_resource
def load_model_assets():
    model = load_model('nuclear_stock_model.h5')
    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open('model_assets.pkl', 'rb') as f:
        model_assets = pickle.load(f)
    sample_data = joblib.load('sample_data.pkl')
    return model, scalers, model_assets, sample_data

model, scalers, model_assets, sample_data = load_model_assets()

# App title
st.title("🌍 Nuclear Energy Stock Price Predictor")
st.markdown("""
Predict future prices of nuclear energy stocks using our LSTM neural network model.
""")

# Sidebar controls
st.sidebar.header("Controls")
ticker_map = {
    'AtkinsRealis': 'ATRL.TO',
    'BWX Technologies': 'BWXT',
    'Cameco': 'CCO.TO',
    'Centrus Energy': 'LEU',
    'GE Vernova': 'GE',
    'NexGen Energy': 'NXE.TO'
}

selected_company = st.sidebar.selectbox(
    "Select Company",
    list(ticker_map.keys())
)

days_to_predict = st.sidebar.slider(
    "Days to Predict",
    min_value=5,
    max_value=60,
    value=30
)

# Prediction function
def predict_future(model, last_sequence, scalers, features, days_to_predict):
    future_predictions = []
    current_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    for _ in range(days_to_predict):
        next_day_close_pred = model.predict(current_sequence, verbose=0)[0, 0]
        future_predictions.append(next_day_close_pred)
        
        # Create new row for the sequence
        new_row = np.zeros(len(features))
        
        # Update each feature (adapt this to match your original logic)
        close_idx = features.index('Close')
        new_row[close_idx] = next_day_close_pred
        # ... [rest of your feature update logic] ...
        
        current_sequence = np.append(current_sequence[:, 1:, :],
                                    [new_row.reshape(1, -1)],
                                    axis=1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scalers['Close'].inverse_transform(future_predictions)
    return future_predictions.flatten()

# Main app logic
if st.sidebar.button("Run Prediction"):
    st.subheader(f"Prediction Results for {selected_company}")
    
    with st.spinner("Fetching data and making predictions..."):
        # Get ticker symbol
        ticker = ticker_map[selected_company]
        
        # Get recent stock data (last 60 days)
        stock_data = yf.download(ticker, period="3mo")
        
        # Here you would need to:
        # 1. Get current uranium prices (you might need an API)
        # 2. Combine and preprocess the data
        # 3. Create the input sequence
        
        # For demo purposes, we'll use the sample data
        last_sequence = sample_data['last_sequence']
        
        # Make predictions
        predictions = predict_future(
            model,
            last_sequence,
            scalers,
            model_assets['features'],
            days_to_predict
        )
        
        # Generate future dates
        last_date = sample_data['test_dates'][-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_dates, predictions, 'r-', label='Predicted Price')
        ax.set_title(f"Predicted Stock Prices for Next {days_to_predict} Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
        # Show prediction table
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": predictions
        })
        st.dataframe(pred_df.style.format({"Predicted Price": "${:.2f}"}))

# Add model info section
with st.expander("Model Information"):
    st.markdown("""
    **Model Architecture:**
    - Bidirectional LSTM neural network
    - 4 LSTM layers with dropout
    - Trained on historical stock and uranium price data
    
    **Input Features:**
    - Open, High, Low, Close prices
    - Trading Volume
    - Uranium Spot Prices
    """)
    
    st.image("model_architecture.png", caption="Model Architecture Diagram")

# Add disclaimer
st.markdown("---")
st.caption("""
*Note: This is for demonstration purposes only. Stock market predictions are inherently uncertain and this app should not be used for actual investment decisions.*
""")
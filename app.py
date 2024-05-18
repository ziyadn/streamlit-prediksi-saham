import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px

# Memuat model yang telah dilatih
pipeline = joblib.load('stock_price_pipeline_updated.pkl')

# Fungsi untuk menghitung fitur tambahan
def calculate_features(data):
    data['MA_3'] = data['Close'].rolling(window=3).mean().shift(1)
    data['MA_5'] = data['Close'].rolling(window=5).mean().shift(1)
    data['MA_10'] = data['Close'].rolling(window=10).mean().shift(1)
    data['Return'] = data['Close'].pct_change().shift(1)
    data['Volatility'] = data['Return'].rolling(window=10).std().shift(1)
    data = data.fillna(0)
    return data

# Judul aplikasi
st.title('Stock Price Prediction')

# Input ticker saham
st.header('Input Ticker Saham')
ticker = st.text_input('Ticker', value='AAPL')

# Tombol untuk mengambil data dan melakukan prediksi
if st.button('Predict'):
    # Mendapatkan data saham 3 bulan terakhir
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=360)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if not stock_data.empty:
        # Menghitung fitur tambahan untuk seluruh data
        stock_data = calculate_features(stock_data)
        
        # Mengambil data terbaru setelah menghitung fitur tambahan
        latest_data = stock_data.iloc[-1]
        Open = latest_data['Open']
        High = latest_data['High']
        Low = latest_data['Low']
        Close = latest_data['Close']
        Volume = latest_data['Volume']

        # Membuat DataFrame untuk prediksi
        input_data = pd.DataFrame({
            'Open': [Open],
            'High': [High],
            'Low': [Low],
            'Close': [Close],
            'Volume': [Volume],
            'MA_3': [latest_data['MA_3']],
            'MA_5': [latest_data['MA_5']],
            'MA_10': [latest_data['MA_10']],
            'Return': [latest_data['Return']],
            'Volatility': [latest_data['Volatility']]
        })
        
        # Melakukan prediksi menggunakan pipeline
        prediction = pipeline.predict(input_data)

        # Menampilkan grafik garis harga penutupan 3 bulan terakhir
        fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'1 Year Close Prices for {ticker}')
        st.plotly_chart(fig)

        # Menampilkan harga penutupan terbaru dalam format Rupiah
        st.subheader(f'Latest Close Price for {ticker}: Rp{Close:.2f}')

        # Menampilkan hasil prediksi dalam format Rupiah
        st.subheader(f'Predicted Close Price for {ticker}: Rp{prediction[0]:.2f}')
    else:
        st.error(f'No data found for ticker: {ticker}')

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. Page Configuration ---
st.set_page_config(page_title="QuantWays AI", layout="wide")

st.title("QuantWays: Intelligent Market Analytics")
st.markdown("*> Ask the AI to analyze any asset (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')*")

# --- 2. Sidebar Controls (The "Quant" Settings) ---
with st.sidebar:
    st.header("Model Parameters")
    lookback = st.slider("Lookback Period (Days)", 30, 365, 180)
    ma_window = st.number_input("Moving Average Window", value=50)
    st.info("This panel adjusts the sensitivity of the QuantWays trend algorithm.")

# --- 3. The Quant Logic Engine ---
def get_market_data(ticker, period):
    """Fetches live data and calculates quant metrics."""
    start_date = datetime.now() - timedelta(days=period)
    df = yf.download(ticker, start=start_date, progress=False)
    # Fix: Flatten MultiIndex columns if they exist (e.g. Price -> Close -> AAPL becomes just Close)
    if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None
        
    # Calculate Moving Average (Simple Quant Indicator)
    df[f'MA_{ma_window}'] = df['Close'].rolling(window=ma_window).mean()
    
    # Calculate Daily Returns & Volatility
    df['Returns'] = df['Close'].pct_change()
    volatility = df['Returns'].std() * (252 ** 0.5) # Annualized Volatility
    
    return df, volatility

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Processing
if prompt := st.chat_input("Enter an asset ticker (e.g., NVDA)..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Response (Simulated Agent)
    with st.chat_message("assistant"):
        with st.spinner(f"QuantWays Agent is analyzing {prompt}..."):
            data, vol = get_market_data(prompt, lookback)
            
            if data is not None:
                # 1. Text Insight
                response_text = (
                    f"### Analysis for **{prompt.upper()}**\n"
                    f"- **Current Price**: ${data['Close'].iloc[-1]:.2f}\n"
                    f"- **Annualized Volatility**: {vol:.2%}\n"
                    f"- **Trend Indicator**: The price is {'above' if data['Close'].iloc[-1] > data[f'MA_{ma_window}'].iloc[-1] else 'below'} "
                    f"the {ma_window}-day moving average."
                )
                st.markdown(response_text)
                
                # 2. Interactive Chart (Plotly)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma_window}'], 
                                         line=dict(color='orange', width=2), name=f'{ma_window}-Day MA'))
                fig.update_layout(title=f"{prompt.upper()} Price Action", xaxis_title="Date", yaxis_title="Price", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save context
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                error_msg = f"⚠️ Could not find data for ticker: **{prompt}**. Please try a valid symbol."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

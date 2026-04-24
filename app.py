import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta

# --- 1. Page Configuration & AI Setup ---
st.set_page_config(page_title="QuantWays AI", layout="wide")

st.title("QuantWays: Intelligent Market Analytics")
st.markdown("*> Ask the AI to analyze any asset (e.g., 'AAPL', 'BTC-USD', 'NVDA')*")

# Initialize Gemini from Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your Streamlit Secrets.")
    st.stop()

# Model Selection (Using your confirmed available model)
try:
    # We use gemini-2.0-flash for high-speed quant reasoning
    # old: model = genai.GenerativeModel('models/gemini-2.0-flash')
    model = genai.GenerativeModel('models/gemini-1.5-flash')
except Exception as e:
    st.error(f"AI Model Initialization failed: {e}")
    model = None

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("Quant Parameters")
    lookback = st.slider("Lookback Period (Days)", 30, 365, 180)
    ma_window = st.number_input("Moving Average Window", value=50, min_value=5, max_value=200)
    st.divider()
    st.info("The QuantWays engine uses these parameters to calculate trend strength and volatility.")

# --- 3. The Quant Logic Engine ---
def get_market_data(ticker, period):
    """Fetches live data and fixes MultiIndex column issues."""
    start_date = datetime.now() - timedelta(days=period)
    
    # Download data
    df = yf.download(ticker, start=start_date, progress=False)
    
    if df.empty:
        return None, None

    # CRITICAL FIX: Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Standardize column naming (ensure 'Close' exists)
    if 'Close' not in df.columns:
        return None, None
        
    # Quant Calculations
    df[f'MA_{ma_window}'] = df['Close'].rolling(window=ma_window).mean()
    df['Returns'] = df['Close'].pct_change()
    volatility = df['Returns'].std() * (252 ** 0.5) # Annualized
    
    return df, volatility

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Enter a ticker symbol..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"QuantWays Agent is analyzing {prompt.upper()}..."):
            data, vol = get_market_data(prompt, lookback)
            
            if data is not None and model is not None:
                # Extract specific values for the AI
                current_price = float(data['Close'].iloc[-1])
                ma_val = float(data[f'MA_{ma_window}'].iloc[-1]) if not pd.isna(data[f'MA_{ma_window}'].iloc[-1]) else 0
                trend_status = "ABOVE" if current_price > ma_val else "BELOW"
                
                # Construct AI Prompt
                ai_prompt = (
                    f"You are a Senior Quant Analyst for the QuantWays platform. "
                    f"Analyze the asset: {prompt.upper()}. "
                    f"Data: Price ${current_price:.2f}, Annual Volatility {vol:.2%}. "
                    f"The price is currently {trend_status} its {ma_window}-day Moving Average (${ma_val:.2f}). "
                    f"Provide a concise professional risk report and trend outlook."
                )

                try:
                    # Generate AI response
                    response = model.generate_content(ai_prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"⚠️ AI Reasoning Error: {str(e)}"
                
                # Display text report
                st.markdown(response_text)
                
                # Render Technical Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], 
                    name='Price Action'
                ))
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[f'MA_{ma_window}'], 
                    line=dict(color='orange', width=2), 
                    name=f'{ma_window}-Day MA'
                ))
                fig.update_layout(
                    title=f"{prompt.upper()} - Technical Analysis View",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store in history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                error_msg = f"⚠️ Could not retrieve data for **{prompt}**. Please check the ticker symbol."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta


# --- 1. Page Configuration & AI Setup ---
st.set_page_config(page_title="QuantWays AI", layout="wide")

# Initialize Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your Streamlit Secrets.")
    st.stop() # This prevents the crash and shows a helpful message instead
model = genai.GenerativeModel('gemini-1.5-flash') 

st.title("QuantWays: Intelligent Market Analytics")
st.markdown("*> Ask the AI to analyze any asset (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')*")

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("Model Parameters")
    lookback = st.slider("Lookback Period (Days)", 30, 365, 180)
    ma_window = st.number_input("Moving Average Window", value=50)
    st.info("This panel adjusts the sensitivity of the QuantWays trend algorithm.")

# --- 3. The Quant Logic Engine ---
def get_market_data(ticker, period):
    start_date = datetime.now() - timedelta(days=period)
    df = yf.download(ticker, start=start_date, progress=False)
    
    if df.empty:
        return None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df[f'MA_{ma_window}'] = df['Close'].rolling(window=ma_window).mean()
    df['Returns'] = df['Close'].pct_change()
    volatility = df['Returns'].std() * (252 ** 0.5) 
    
    return df, volatility

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter an asset ticker (e.g., NVDA)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"QuantWays Agent is analyzing {prompt}..."):
            data, vol = get_market_data(prompt, lookback)
            
            if data is not None:
                # Prepare data for Gemini
                current_price = data['Close'].iloc[-1]
                ma_val = data[f'MA_{ma_window}'].iloc[-1]
                trend = "ABOVE" if current_price > ma_val else "BELOW"
                
                # Gemini Analysis
                ai_prompt = (
                    f"You are a professional Quant Analyst for QuantWays. "
                    f"The user wants an analysis for {prompt.upper()}. "
                    f"Current Stats: Price ${current_price:.2f}, Volatility {vol:.2%}, "
                    f"Price is {trend} the {ma_window}-day Moving Average. "
                    f"Provide a brief, professional summary of the risk and trend."
                )
                
                response = model.generate_content(ai_prompt)
                response_text = response.text
                
                st.markdown(response_text)
                
                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma_window}'], 
                                         line=dict(color='orange', width=2), name=f'{ma_window}-Day MA'))
                fig.update_layout(title=f"{prompt.upper()} Technical View", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                error_msg = f"⚠️ No data found for ticker: **{prompt}**."
                st.error(error_msg)

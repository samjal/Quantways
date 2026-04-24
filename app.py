import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta

# --- 1. Page Configuration & AI Setup ---
st.set_page_config(page_title="QuantWays AI", layout="wide")

st.title("QuantWays: Intelligent Market Analytics")
st.markdown("*> Ask the AI to analyze any asset (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')*")

# Initialize Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your Streamlit Secrets.")
    st.stop()

# Diagnostic & Model Setup
try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    st.sidebar.write("✅ Models found:", available_models)
    # Using the production path for the model
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.sidebar.error(f"AI Setup failed: {e}")
    model = None

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
            
            if data is not None and model is not None:
                # Prepare stats
                current_price = float(data['Close'].iloc[-1])
                ma_val = float(data[f'MA_{ma_window}'].iloc[-1])
                trend = "ABOVE" if current_price > ma_val else "BELOW"
                
                # Gemini Analysis
                ai_prompt = (
                    f"You are a professional Quant Analyst for QuantWays. "
                    f"Analyze {prompt.upper()}. Price: ${current_price:.2f}, "
                    f"Volatility: {vol:.2%}, Trend: {trend} {ma_window}-day MA. "
                    f"Provide a brief, professional risk summary."
                )

                try:
                    response = model.generate_content(ai_prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"⚠️ AI Analysis failed: {e}"
                
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
                error_msg = f"⚠️ No data found or AI offline for: **{prompt}**."
                st.error(error_msg)

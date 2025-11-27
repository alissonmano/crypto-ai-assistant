import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
import requests
import os
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from dotenv import load_dotenv

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Crypto AI Commander", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Load environment variables (for local dev)
load_dotenv()

# --- 2. SESSION STATE (MEMORY) ---
# This keeps your trade logs and settings alive between clicks
if 'trade_log' not in st.session_state:
    st.session_state['trade_log'] = []
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = time.time()

# --- 3. HELPER FUNCTIONS ---
def get_secret(key):
    """Securely retrieve secrets from Streamlit Cloud or Environment."""
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

def format_currency(value):
    return f"${value:,.2f}"

# --- 4. ROBUST CLASS DEFINITIONS ---

class MarketData:
    """Handles data fetching with caching and error resilience."""
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None

    @st.cache_data(ttl=300) # Cache data for 5 minutes to save bandwidth/speed
    def fetch_data(_self, ticker, period="2y", interval="1d"):
        try:
            # Internal helper to attempt download
            def try_download(symbol):
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                if not data.empty:
                    # Handle MultiIndex (yfinance structure)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    return data
                return None

            # Generate list of ticker variations to try
            # Yahoo Finance prefers "BTC-USD" but users often type "BTC/USDT" or "BTC-USDT"
            candidates = [
                ticker,                                             # 1. Exact match
                ticker.replace("USDT", "USD"),                      # 2. Try USD instead of USDT
                ticker.replace("/", "-"),                           # 3. Try Dash instead of Slash
                ticker.replace("/", "-").replace("USDT", "USD"),    # 4. Try Dash AND USD
                f"{ticker}-USD"                                     # 5. Try appending -USD (e.g. for "BTC")
            ]
            
            df = None
            for symbol in candidates:
                df = try_download(symbol)
                if df is not None:
                    break

            # Validation: Check if empty
            if df is None or df.empty:
                return None
            
            # Validation: Check for required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                return None

            return df
        except Exception as e:
            st.error(f"Data Fetch Error: {str(e)}")
            return None

    def process_indicators(self, df):
        """Adds technical indicators using the robust 'ta' library."""
        try:
            df = df.copy()
            
            # RSI (Momentum)
            rsi_indicator = RSIIndicator(close=df["Close"], window=14)
            df["RSI"] = rsi_indicator.rsi()

            # SMA (Trend)
            df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
            df["SMA_200"] = SMAIndicator(close=df["Close"], window=200).sma_indicator()

            # ATR (Volatility for Stop Loss)
            atr_indicator = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
            df["ATR"] = atr_indicator.average_true_range()

            # AI Target: Did price go up next day? (1 = Yes, 0 = No)
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

            # Drop NaNs created by indicators (e.g., first 200 rows for SMA_200)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            st.error(f"Indicator Calculation Error: {e}")
            return None

class AIBrain:
    """The Machine Learning Model."""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
        self.features = ['RSI', 'SMA_50', 'SMA_200', 'ATR']

    def train_and_predict(self, df):
        # Validation
        if len(df) < 250:
            return 0.5, 0.0, "Insufficient Data"

        X = df[self.features]
        y = df['Target']

        # Train/Test Split (Time-series aware: strictly past data predicts future)
        split = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Train
        self.model.fit(X_train, y_train)
        
        # Validate
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Predict Future (Today's Close -> Tomorrow)
        latest_features = X.iloc[[-1]]
        probability_up = self.model.predict_proba(latest_features)[0][1]
        
        return probability_up, accuracy, "Success"

class TradeManager:
    """Handles Exchange Connection and Execution."""
    def __init__(self, exchange_id, api_key, secret, dry_run):
        self.dry_run = dry_run
        self.exchange = None
        
        if not dry_run:
            if not api_key or not secret:
                st.warning("⚠️ API Keys missing. Switched to Dry Run.")
                self.dry_run = True
            else:
                try:
                    exchange_class = getattr(ccxt, exchange_id)
                    self.exchange = exchange_class({
                        'apiKey': api_key,
                        'secret': secret,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'} 
                    })
                    # Lightweight check
                    self.exchange.check_required_credentials()
                except Exception as e:
                    st.error(f"❌ API Connection Failed: {e}")
                    self.dry_run = True

    def _resolve_symbol(self, symbol):
        """Helper to fix ticker formats (e.g. BTC-USD -> BTC/USDT)"""
        # 1. Normalize separator
        std_symbol = symbol.replace("-", "/") # e.g. BTC/USDT
        
        # 2. Build candidate list intelligently
        candidates = [std_symbol]
        
        # Common crypto mis-matches
        if std_symbol.endswith("/USD"):
            candidates.append(std_symbol.replace("/USD", "/USDT"))
        elif std_symbol.endswith("/USDT"):
            candidates.append(std_symbol.replace("/USDT", "/USD"))
            
        # 3. Try candidates
        for cand in candidates:
            price = self._fetch_price_safe(cand)
            if price > 0:
                return cand, price
        
        return None, 0.0

    def _fetch_price_safe(self, symbol):
        """Internal method to try fetching price without crashing."""
        # Try CCXT first (Primary Source)
        exchange_ref = self.exchange if (self.exchange and not self.dry_run) else ccxt.binance()
        try:
            return exchange_ref.fetch_ticker(symbol)['last']
        except Exception as e:
            # Fallback for Paper Trading: Try fetching from Yahoo Finance if Exchange fails
            # This is useful if running locally with IP blocks on Binance
            if self.dry_run:
                try:
                    # Robust Conversion for YFinance Fallback
                    if symbol.endswith("/USDT"):
                        yf_sym = symbol.replace("/USDT", "-USD")
                    elif symbol.endswith("/USD"):
                        yf_sym = symbol.replace("/", "-")
                    else:
                        yf_sym = symbol.replace("/", "-")
                        if "USD" not in yf_sym:
                             yf_sym += "-USD"
                    
                    data = yf.Ticker(yf_sym).history(period="1d")
                    if not data.empty:
                        return data['Close'].iloc[-1]
                except:
                    pass
            return 0.0

    def get_ticker_price(self, symbol):
        """Public method to get price (for display)."""
        _, price = self._resolve_symbol(symbol)
        return price

    def get_balance(self, currency="USDT"):
        """Checks wallet balance."""
        if self.dry_run: return 10000.0 # Fake $10k
        try:
            bal = self.exchange.fetch_balance()
            return bal['total'].get(currency, 0.0)
        except Exception as e:
            st.error(f"Balance Check Error: {e}")
            return 0.0

    def execute_order(self, symbol, side, amount_usd):
        # 1. Resolve Symbol (Auto-fix USD -> USDT)
        valid_symbol, price = self._resolve_symbol(symbol)
        
        if not valid_symbol or price <= 0:
            return f"❌ Error: Could not verify price for {symbol}. Exchange API might be unreachable for this pair. Try refreshing or checking your network."

        amount_coin = amount_usd / price

        # 2. Dry Run Logic
        if self.dry_run:
            log_entry = f"📝 [PAPER] {side.upper()} {amount_coin:.5f} {valid_symbol} @ {format_currency(price)}"
            st.session_state['trade_log'].insert(0, f"{datetime.now().strftime('%H:%M:%S')} - {log_entry}")
            return log_entry

        # 3. Live Logic
        try:
            # Balance Check
            quote_currency = valid_symbol.split("/")[1] # e.g. USDT
            balance = self.get_balance(quote_currency)
            
            if side == 'buy' and balance < amount_usd:
                return f"❌ Insufficient {quote_currency}. Have: {balance:.2f}, Need: {amount_usd:.2f}"

            # Execute
            order = self.exchange.create_market_order(valid_symbol, side, amount_coin)
            
            log_entry = f"🚀 [LIVE] {side.upper()} {valid_symbol} | ID: {order['id']}"
            st.session_state['trade_log'].insert(0, f"{datetime.now().strftime('%H:%M:%S')} - {log_entry}")
            return f"✅ SUCCESS: {log_entry}"

        except Exception as e:
            return f"❌ Execution Error: {str(e)}"

# --- 5. UI IMPLEMENTATION ---

# Sidebar
with st.sidebar:
    st.title("🦅 Crypto Commander")
    
    # Asset Selection with Dropdown
    popular_assets = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "LTC-USD", "Custom"]
    selected_asset = st.selectbox("Select Asset", popular_assets)
    
    if selected_asset == "Custom":
        ticker_input = st.text_input("Enter Ticker", value="BTC-USD", help="Yahoo Finance Ticker Format").upper()
    else:
        ticker_input = selected_asset
    
    # Mode Toggle
    st.divider()
    mode = st.radio("Trading Mode", ["Paper Trading (Safe)", "Live Trading (Real Money)"])
    dry_run_active = True if "Paper" in mode else False
    
    if not dry_run_active:
        st.warning("⚠️ YOU ARE USING REAL MONEY")
    
    # API Keys (Collapsible for cleaner mobile view)
    with st.expander("🔐 API Credentials"):
        api_key = st.text_input("API Key", value=get_secret("EXCHANGE_API_KEY"), type="password")
        api_secret = st.text_input("API Secret", value=get_secret("EXCHANGE_SECRET"), type="password")
        exchange_id = st.selectbox("Exchange", ["binance", "coinbase", "kraken"], index=0)

    # Discord
    with st.expander("🔔 Notifications"):
        webhook = st.text_input("Discord Webhook URL", type="password")

# Main Content
if ticker_input:
    # 1. Initialize Classes
    market = MarketData(ticker_input)
    trader = TradeManager(exchange_id, api_key, api_secret, dry_run=dry_run_active)
    
    # 2. Fetch Data
    raw_df = market.fetch_data(ticker_input)
    
    if raw_df is not None:
        # 3. Process & AI
        df = market.process_indicators(raw_df)
        ai = AIBrain()
        prob_up, accuracy, msg = ai.train_and_predict(df)
        
        current_price = df['Close'].iloc[-1]
        
        # 4. Dashboard Header
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", format_currency(current_price))
        
        # AI Signal Color Logic
        signal_color = "off"
        if prob_up > 0.60: 
            signal_text = "STRONG BUY"
            signal_color = "normal" # Green in standard theme usually
        elif prob_up < 0.40:
            signal_text = "STRONG SELL"
            signal_color = "inverse"
        else:
            signal_text = "NEUTRAL"
        
        c2.metric("AI Signal", signal_text, delta=f"{prob_up:.1%} Bullish")
        c3.metric("Model Accuracy", f"{accuracy:.1%}")
        c4.metric("Volatility (ATR)", format_currency(df['ATR'].iloc[-1]))

        # 5. Charts Area
        tab_chart, tab_trade, tab_log = st.tabs(["📈 Market Vision", "⚡ Execution Deck", "📜 Logs"])
        
        with tab_chart:
            # Advanced Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
            fig.update_layout(height=450, margin=dict(l=10, r=10, b=10, t=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_trade:
            st.subheader("Manual Override")
            col_amt, col_btn = st.columns([1, 1])
            
            with col_amt:
                amount = st.number_input("Trade Amount (USD)", min_value=10.0, value=100.0, step=10.0)
            
            with col_btn:
                # Buy / Sell Buttons with Callback
                b1, b2 = st.columns(2)
                if b1.button("🟢 BUY NOW", use_container_width=True):
                    res = trader.execute_order(ticker_input, 'buy', amount)
                    if "✅" in res: st.success(res)
                    else: st.error(res)
                    
                if b2.button("🔴 SELL NOW", use_container_width=True):
                    res = trader.execute_order(ticker_input, 'sell', amount)
                    if "✅" in res: st.success(res)
                    else: st.error(res)

            st.caption("AI Auto-Trading is currently disabled. Use buttons above to execute based on AI signals.")

        with tab_log:
            st.write("Recent Activity:")
            for log in st.session_state['trade_log']:
                st.text(log)
                
    else:
        st.error(f"Could not load data for {ticker_input}. Please check the ticker symbol.")
else:
    st.info("👈 Enter a crypto ticker in the sidebar to begin.")

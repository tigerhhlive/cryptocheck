import os
import time
import logging
import requests
import threading
import pandas as pd
from flask import Flask
from datetime import datetime

app = Flask(__name__)

CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

ADX_THRESHOLD = 20
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.2
TP1_MULTIPLIER = 1.8
TP2_MULTIPLIER = 2.8
MIN_PERCENT_RISK = 0.03
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
MONITOR_INTERVAL = 120
SLEEP_HOURS = (0, 7)
MIN_ATR = 0.001
SIGNAL_COOLDOWN = 1800

last_signals = {}
daily_signal_count = 0
daily_hit_count = 0
last_report_day = None
open_positions = {}
tp1_count = 0
tp2_count = 0
sl_count = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

def test_api(symbol):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        'fsym': symbol[:-4],  # Assuming the symbol is in format SYMBOLUSDT
        'tsym': 'USDT',
        'limit': 60,
        'aggregate': 15,  # 15-minute aggregate
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        json_data = response.json()

        # Log full API response to check the returned data
        logging.info(f"API response for {symbol}: {json_data}")

        # If the response contains valid data, print a sample
        if "Data" in json_data and "Data" in json_data["Data"]:
            logging.info(f"Valid data received for {symbol}: {json_data['Data']['Data'][:5]}")  # Show the first 5 rows
        else:
            logging.warning(f"Invalid data or no data received for {symbol}")
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")

# Test for a specific symbol (e.g., BTCUSDT)
test_api('BTCUSDT')

def get_data(timeframe, symbol):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    aggregate = 5 if timeframe == '5m' else 15
    limit = 60
    fsym, tsym = symbol[:-4], "USDT"
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'aggregate': aggregate,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        json_data = res.json()
        data = json_data.get("Data", {}).get("Data", [])
        if not data or not isinstance(data, list):
            logging.warning(f"⚠️ No valid data received for {symbol} in {timeframe}. Raw: {json_data}")
            return None
        df = pd.DataFrame(data)
        if df.empty or df.isnull().all().any():
            logging.warning(f"⚠️ DataFrame is empty or all null for {symbol} in {timeframe}")
            return None
        
        # Correct the column names and calculate volume
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumefrom'] + df['volumeto']  # Sum volumefrom and volumeto for volume

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol}: {e}")
        return None

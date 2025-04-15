import os
import requests
import time
import logging
import threading
import pandas as pd
import pandas_ta as ta
from flask import Flask
from datetime import datetime

app = Flask(__name__)

# تنظیمات
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# متغیرها برای ردیابی پوزیشن‌ها
open_positions = {}
daily_signal_count = 0
daily_hit_count = 0
tp_count = 0
sl_count = 0

# تنظیمات گزارش روزانه
SLEEP_HOURS = (0, 7)  # ساعات خواب
HEARTBEAT_INTERVAL = 7200  # فاصله بین هر پیام حیات
CHECK_INTERVAL = 600  # فاصله چک کردن هر سیگنال

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ارسال پیام به تلگرام
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

# تابع برای دریافت داده‌ها از CryptoCompare
def get_data(timeframe, symbol):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    aggregate = 5 if timeframe == '5m' else 15 if timeframe == '15m' else 30 if timeframe == '30m' else 60  # روزانه (1d)
    limit = 60
    fsym, tsym = symbol[:-4], "USDT"
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'aggregate': aggregate,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    res = requests.get(url, params=params, timeout=10)
    data = res.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['volume'] = df['volumeto']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# تابع برای بررسی وضعیت پوزیشن‌ها و مدیریت آن‌ها
def monitor_positions():
    global tp_count, sl_count, daily_signal_count, daily_hit_count
    while True:
        for symbol, position in open_positions.items():
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            current_price = get_data('5m', symbol)['close'].iloc[-1]
            
            # بررسی TP و SL
            if current_price >= take_profit:
                tp_count += 1
                position['status'] = 'TP Hit'
                logging.info(f"{symbol} TP Hit")
            elif current_price <= stop_loss:
                sl_count += 1
                position['status'] = 'SL Hit'
                logging.info(f"{symbol} SL Hit")
            open_positions[symbol] = position
        
        # گزارش روزانه قبل از رفتن به حالت خواب
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            send_telegram_message(f"✅ Daily Report\nTotal Signals: {daily_signal_count}\nTP Hits: {tp_count}\nSL Hits: {sl_count}")
            time.sleep(60)  # یک دقیقه منتظر بمون تا دوباره چک کنه

        time.sleep(CHECK_INTERVAL)

# تابع اصلی برای بررسی سیگنال‌ها
def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count
    df = get_data(timeframe, symbol)
    if len(df) < 30:
        return None, None

    # محاسبات اندیکاتورها
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])

    candle = df.iloc[-2]
    rsi_val = df['rsi'].iloc[-2]
    entry_price = df['close'].iloc[-2]
    stop_loss = entry_price - (df['ATR'].iloc[-2] * 1.2)  # حد ضرر
    take_profit = entry_price + (df['ATR'].iloc[-2] * 2.8)  # حد سود

    # ذخیره اطلاعات پوزیشن
    open_positions[symbol] = {'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'status': 'Open'}

    message = f"🚨 *Signal for {symbol}*\nEntry: {entry_price}\nStop Loss: {stop_loss}\nTake Profit: {take_profit}"
    send_telegram_message(message)

    daily_signal_count += 1
    return "BUY" if rsi_val < 30 else "SELL", message

def monitor():
    global daily_signal_count
    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT"
    ]

    while True:
        for symbol in symbols:
            try:
                msg, _ = analyze_symbol(symbol, '15m')
                if msg:
                    daily_hit_count += 1
            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

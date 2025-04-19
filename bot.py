import os
import time
import logging
import requests
import threading
import pandas as pd
import pandas_ta as ta
from flask import Flask
from datetime import datetime

app = Flask(__name__)

# Environment
API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT = os.environ.get('TELEGRAM_CHAT_ID')

# Strategy params
EMA_FAST = 9
EMA_SLOW = 15
RSI_LEN = 14
ATR_LEN = 14
OB_LOOKBACK = 10     # Swing lookback for OB detection
SIGNAL_COOLDOWN = 1800
CHECK_INTERVAL = 600
HEARTBEAT_INTERVAL = 7200
SLEEP_HOURS = (0, 7)

# State
last_signal_bar = {}
open_positions = {}
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def send_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = { 'chat_id': TELEGRAM_CHAT, 'text': text, 'parse_mode': 'Markdown' }
    requests.post(url, json=payload)


def get_data(symbol, agg=15):
    url = 'https://min-api.cryptocompare.com/data/v2/histominute'
    params = { 'fsym': symbol[:-4], 'tsym': 'USDT', 'limit': 100, 'aggregate': agg, 'api_key': API_KEY }
    r = requests.get(url, params=params).json()['Data']['Data']
    df = pd.DataFrame(r)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open','high','low','close','volumefrom','volumeto']]


def detect_swing_highs(df):
    df['is_swing_high'] = ((df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high']))
    return df[df['is_swing_high']]


def detect_swing_lows(df):
    df['is_swing_low'] = ((df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low']))
    return df[df['is_swing_low']]


def detect_rsi_divergence(df):
    # bearish: price swing highs up + RSI swing highs down
    sh = detect_swing_highs(df)
    if len(sh) < 2: return None
    last2 = sh.iloc[-2:]
    if last2['high'].iloc[1] > last2['high'].iloc[0] and last2['RSI'].iloc[1] < last2['RSI'].iloc[0]:
        return 'bear'
    # bullish divergence
    sl = detect_swing_lows(df)
    if len(sl) < 2: return None
    last2l = sl.iloc[-2:]
    if last2l['low'].iloc[1] < last2l['low'].iloc[0] and last2l['RSI'].iloc[1] > last2l['RSI'].iloc[0]:
        return 'bull'
    return None


def analyze(symbol):
    df = get_data(symbol, agg=15)
    # Indicators
    df['EMA9'] = ta.ema(df['close'], length=EMA_FAST)
    df['EMA15'] = ta.ema(df['close'], length=EMA_SLOW)
    df['RSI'] = ta.rsi(df['close'], length=RSI_LEN)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_LEN)

    # Detect OB
    sh = detect_swing_highs(df)
    if sh.empty: return
    ob_top = sh['high'].iloc[-1]
    # OB bottom = lowest low in lookback window before top
    i = sh.index[-1]
    bottom_window = df.loc[i - pd.Timedelta(minutes=OB_LOOKBACK*15):i]
    ob_bot = bottom_window['low'].min()

    last = df.iloc[-1]
    # RSI divergence
    div = detect_rsi_divergence(df)
    # Entry conditions
    if div=='bear' and last['close'] <= ob_top and last['close'] >= ob_bot:
        bar_idx = df.index[-1]
        key = f"{symbol}_short"
        if last_signal_bar.get(key) == bar_idx: return
        last_signal_bar[key] = bar_idx
        # Build message
        sl = ob_top + 0.5*(ob_top-ob_bot)
        tp1 = last['close'] - last['ATR']*1
        tp2 = last['close'] - last['ATR']*2
        msg = (f"🚨 *AI SIGNAL Alert*\n"
               f"*{symbol}* - 🔴 SELL@{last['close']:.4f}\n"
               f"SL: {sl:.4f} TP1: {tp1:.4f} TP2: {tp2:.4f}\n"
               f"OB Zone: [{ob_bot:.4f}–{ob_top:.4f}]  RSI Divergence: *Bearish*")
        send_msg(msg)
    # bullish
    sls = detect_swing_lows(df)
    if sls.empty: return
    ob_bot2 = sls['low'].iloc[-1]
    top_window = df.loc[ sls.index[-1]: sls.index[-1] + pd.Timedelta(minutes=OB_LOOKBACK*15) ]
    ob_top2 = top_window['high'].max()
    if div=='bull' and last['close'] >= ob_bot2 and last['close'] <= ob_top2:
        bar_idx = df.index[-1]
        key = f"{symbol}_long"
        if last_signal_bar.get(key) == bar_idx: return
        last_signal_bar[key] = bar_idx
        sl = ob_bot2 - 0.5*(ob_top2-ob_bot2)
        tp1 = last['close'] + last['ATR']*1
        tp2 = last['close'] + last['ATR']*2
        msg = (f"🚨 *AI SIGNAL Alert*\n"
               f"*{symbol}* - 🟢 BUY@{last['close']:.4f}\n"
               f"SL: {sl:.4f} TP1: {tp1:.4f} TP2: {tp2:.4f}\n"
               f"OB Zone: [{ob_bot2:.4f}–{ob_top2:.4f}]  RSI Divergence: *Bullish*")
        send_msg(msg)


def worker():
    symbols = ["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT","ADAUSDT","XLMUSDT"]
    last_hb = time.time()
    while True:
        now = datetime.utcnow()
        hr = (now.hour+3)%24
        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]: time.sleep(60); continue
        # Heartbeat
        if time.time()-last_hb>HEARTBEAT_INTERVAL:
            send_msg("🤖 *Bot is live and scanning*.")
            last_hb = time.time()
        # scan
        for sym in symbols:
            try: analyze(sym)
            except Exception as e: logging.error(e)
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home(): return "✅ Running"

if __name__=='__main__':
    threading.Thread(target=worker, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',8080)))

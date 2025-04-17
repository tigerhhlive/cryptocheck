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

# Environment variables
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Strategy parameters
EMA_FAST = 9
EMA_SLOW = 15
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
ATR_SL_MULT = 1.0  # SL = ATR * 1
ATR_TP1_MULT = 1.0 # TP1 = ATR * 1
ATR_TP2_MULT = 2.0 # TP2 = ATR * 2
ADX_THRESHOLD = 20
SIGNAL_COOLDOWN = 1800
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
MONITOR_INTERVAL = 120
SLEEP_HOURS = (0, 7)
MIN_DATA_POINTS = max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD, MACD_SLOW + MACD_SIGNAL)

# Tracking
last_signals = {}
daily_signal_count = 0
daily_win_count = 0
daily_loss_count = 0
open_positions = {}

# Logging setup
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


def get_data(timeframe, symbol):
    """
    Fetch historical data and return DataFrame with:
    timestamp, open, high, low, close, volume_coin, volume_quote, volume
    """
    aggregate = 5 if timeframe == '5m' else 15
    limit = 100
    fsym, tsym = symbol[:-4], 'USDT'
    url = 'https://min-api.cryptocompare.com/data/v2/histominute'
    params = { 'fsym': fsym, 'tsym': tsym, 'limit': limit, 'aggregate': aggregate,
               'api_key': CRYPTOCOMPARE_API_KEY }
    res = requests.get(url, params=params, timeout=10)
    raw = res.json().get('Data', {}).get('Data', [])
    df = pd.DataFrame(raw)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['volume_coin']  = df['volumefrom'].astype(float)
    df['volume_quote'] = df['volumeto'].astype(float)
    df['volume'] = df['volume_quote']
    return df[['timestamp','open','high','low','close','volume_coin','volume_quote','volume']]


def check_cooldown(symbol, direction, index):
    key = f"{symbol}_{direction}"
    last_idx = last_signals.get(key)
    if last_idx == index:
        return False
    last_signals[key] = index
    return True


def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count, daily_win_count, daily_loss_count
    df = get_data(timeframe, symbol)
    if df.shape[0] < MIN_DATA_POINTS:
        return None

    # Indicators
    df['EMA_fast'] = ta.ema(df['close'], length=EMA_FAST)
    df['EMA_slow'] = ta.ema(df['close'], length=EMA_SLOW)
    df['RSI']      = ta.rsi(df['close'], length=RSI_PERIOD)
    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['MACD_hist'] = macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
    df['ATR']      = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    idx  = df.index[-1]

    # EMA cross filter
    long_cross  = prev['EMA_fast'] < prev['EMA_slow']  and last['EMA_fast'] > last['EMA_slow']
    short_cross = prev['EMA_fast'] > prev['EMA_slow']  and last['EMA_fast'] < last['EMA_slow']
    if not (long_cross or short_cross):
        logging.info(f"{symbol}: No EMA9/15 cross")
        return None
    direction = 'Long' if long_cross else 'Short'

    # RSI filter
    if direction=='Long' and last['RSI'] < 50:
        logging.info(f"{symbol}: RSI <50 for Long")
        return None
    if direction=='Short' and last['RSI'] > 50:
        logging.info(f"{symbol}: RSI >50 for Short")
        return None

    # MACD Histogram filter
    if direction=='Long' and last['MACD_hist'] < 0:
        logging.info(f"{symbol}: MACDh <0 for Long")
        return None
    if direction=='Short' and last['MACD_hist'] > 0:
        logging.info(f"{symbol}: MACDh >0 for Short")
        return None

    # Price action filter (Marubozu or Engulfing)
    pa = detect_strong_candle(last) or detect_engulfing(df)
    if not pa:
        logging.info(f"{symbol}: No price-action pattern")
        return None

    entry = last['close']
    atr   = last['ATR']
    sl, tp1, tp2 = None, None, None
    if direction=='Long':
        sl  = entry - atr * ATR_SL_MULT
        tp1 = entry + atr * ATR_TP1_MULT
        tp2 = entry + atr * ATR_TP2_MULT
    else:
        sl  = entry + atr * ATR_SL_MULT
        tp1 = entry - atr * ATR_TP1_MULT
        tp2 = entry - atr * ATR_TP2_MULT

    # Cooldown
    if not check_cooldown(symbol, direction, idx):
        logging.info(f"{symbol}: Cooldown active")
        return None

    daily_signal_count += 1
    # Build message
    stars = '🔥'*3
    msg = (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Signal:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Entry:* `{entry:.4f}`  *SL:* `{sl:.4f}`  *TP1:* `{tp1:.4f}`  *TP2:* `{tp2:.4f}`\n"
        f"*RSI:* {last['RSI']:.1f}  *MACDh:* {last['MACD_hist']:.4f}\n"
        f"*Pattern:* {pa}\n"
        f"*Strength:* {stars}"
    )

    # Track open position
    open_positions[symbol] = {'direction':direction, 'sl':sl, 'tp1':tp1, 'tp2':tp2}
    return msg


def detect_strong_candle(row, threshold=0.7):
    body = abs(row['close']-row['open'])
    rng  = row['high']-row['low']
    if rng==0: return None
    return ('bullish_marubozu' if row['close']>row['open'] else 'bearish_marubozu') if body/rng>threshold else None


def detect_engulfing(df):
    if len(df)<3: return None
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    if prev['close']<prev['open'] and curr['close']>curr['open'] and curr['close']>prev['open'] and curr['open']<prev['close']:
        return 'bullish_engulfing'
    if prev['close']>prev['open'] and curr['close']<curr['open'] and curr['open']>prev['close'] and curr['close']<prev['open']:
        return 'bearish_engulfing'
    return None


def monitor_positions():
    global daily_win_count, daily_loss_count
    while True:
        for sym, pos in list(open_positions.items()):
            df = get_data('15m', sym)
            price = df['close'].iloc[-1]
            if pos['direction']=='Long':
                if price>=pos['tp2']: daily_win_count+=1; del open_positions[sym]
                elif price<=pos['sl']: daily_loss_count+=1; del open_positions[sym]
            else:
                if price<=pos['tp2']: daily_win_count+=1; del open_positions[sym]
                elif price>=pos['sl']: daily_loss_count+=1; del open_positions[sym]
        time.sleep(MONITOR_INTERVAL)


def report_daily():
    wins   = daily_win_count
    losses = daily_loss_count
    total  = wins + losses
    winrate= round(wins/total*100,1) if total>0 else 0.0
    send_telegram_message(
        f"📊 Daily Performance Report\n"
        f"Total Signals: {daily_signal_count}\n"
        f"🎯 Wins: {wins}\n"
        f"❌ Losses: {losses}\n"
        f"📈 Winrate: {winrate}%"
    )


def monitor():
    last_hb=0
    symbols=["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT"]
    while True:
        now=datetime.utcnow()
        hour=(now.hour+3)%24;minute=now.minute
        if hour in range(*SLEEP_HOURS): time.sleep(60);continue
        if time.time()-last_hb>HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot live and scanning.")
            last_hb=time.time()
        for sym in symbols:
            msg = analyze_symbol(sym,'15m')
            if msg:
                send_telegram_message(msg)
        if hour==23 and minute>=55:
            report_daily()
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__=='__main__':
    threading.Thread(target=monitor,daemon=True).start()
    threading.Thread(target=monitor_positions,daemon=True).start()
    port=int(os.environ.get('PORT',8080))
    app.run(host='0.0.0.0',port=port)

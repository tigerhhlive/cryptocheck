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
TELEGRAM_BOT_TOKEN     = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID       = os.environ.get('TELEGRAM_CHAT_ID')

# Strategy parameters
EMA_FAST       = 9
EMA_SLOW       = 15
RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
ATR_PERIOD     = 14
ATR_SL_MULT    = 1.0
ATR_TP1_MULT   = 1.0
ATR_TP2_MULT   = 2.0
SIGNAL_COOLDOWN   = 1800
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL     = 600
MONITOR_INTERVAL   = 120
SLEEP_HOURS        = (0, 7)
MIN_DATA_POINTS    = max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD, MACD_SLOW + MACD_SIGNAL)

# Tracking
last_signals      = {}
daily_signal_count = 0
open_positions     = {}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")


def get_data(timeframe, symbol):
    aggregate = 5 if timeframe == '5m' else 15
    limit     = 60
    fsym, tsym = symbol[:-4], 'USDT'
    url = 'https://min-api.cryptocompare.com/data/v2/histominute'
    params = {'fsym': fsym, 'tsym': tsym, 'limit': limit, 'aggregate': aggregate, 'api_key': CRYPTOCOMPARE_API_KEY}
    res = requests.get(url, params=params, timeout=10)
    data = res.json().get('Data', {}).get('Data', [])
    df = pd.DataFrame(data)
    df['timestamp']    = pd.to_datetime(df['time'], unit='s')
    df['volume_coin']  = df['volumefrom'].astype(float)
    df['volume_quote'] = df['volumeto'].astype(float)
    return df[['timestamp','open','high','low','close','volume_coin','volume_quote']]


def check_cooldown(symbol, direction, idx):
    key = f"{symbol}_{direction}"
    if last_signals.get(key) == idx:
        return False
    last_signals[key] = idx
    return True


def detect_strong_candle(row, threshold=0.7):
    body = abs(row['close'] - row['open'])
    rng  = row['high'] - row['low']
    if rng == 0:
        return None
    return ('bullish_marubozu' if row['close'] > row['open'] else 'bearish_marubozu') if (body/rng) > threshold else None


def detect_engulfing(df):
    if len(df) < 3:
        return None
    prev, curr = df.iloc[-3], df.iloc[-2]
    if prev['close'] < prev['open'] and curr['close'] > curr['open'] \
       and curr['close'] > prev['open'] and curr['open'] < prev['close']:
        return 'bullish_engulfing'
    if prev['close'] > prev['open'] and curr['close'] < curr['open'] \
       and curr['open'] > prev['close'] and curr['close'] < prev['open']:
        return 'bearish_engulfing'
    return None


def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count
    df = get_data(timeframe, symbol)
    if df.shape[0] < MIN_DATA_POINTS:
        return None

    # indicators
    df['EMA_fast']  = ta.ema(df['close'], length=EMA_FAST)
    df['EMA_slow']  = ta.ema(df['close'], length=EMA_SLOW)
    df['RSI']       = ta.rsi(df['close'], length=RSI_PERIOD)
    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['MACD_hist'] = macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
    df['ATR']       = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    prev = df.iloc[-2]
    last = df.iloc[-1]
    idx  = df.index[-1]

    long_cross  = (prev['EMA_fast'] < prev['EMA_slow'] and last['EMA_fast'] > last['EMA_slow'])
    short_cross = (prev['EMA_fast'] > prev['EMA_slow'] and last['EMA_fast'] < last['EMA_slow'])
    if not (long_cross or short_cross):
        return None
    direction = 'Long' if long_cross else 'Short'

    if direction=='Long' and last['RSI'] < 50: return None
    if direction=='Short' and last['RSI'] > 50: return None

    if direction=='Long' and last['MACD_hist'] < 0: return None
    if direction=='Short' and last['MACD_hist'] > 0: return None

    pa = detect_strong_candle(last) or detect_engulfing(df)
    if not pa:
        return None

    entry = last['close']
    atr   = last['ATR']
    if direction == 'Long':
        sl  = entry - atr * ATR_SL_MULT
        tp1 = entry + atr * ATR_TP1_MULT
        tp2 = entry + atr * ATR_TP2_MULT
    else:
        sl  = entry + atr * ATR_SL_MULT
        tp1 = entry - atr * ATR_TP1_MULT
        tp2 = entry - atr * ATR_TP2_MULT

    if not check_cooldown(symbol, direction, idx):
        return None

    daily_signal_count += 1

    stars = '🔥🔥🔥'
    msg = (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Signal:* {'🟢 BUY MARKET' if direction=='Long' else '🔴 SELL MARKET'}\n"
        f"*Entry:* `{entry:.6f}`   *SL:* `{sl:.6f}`   *TP1:* `{tp1:.6f}`   *TP2:* `{tp2:.6f}`\n"
        f"*RSI:* {last['RSI']:.1f}   *MACD_h:* {last['MACD_hist']:.4f}\n"
        f"*Pattern:* {pa}\n"
        f"*Strength:* {stars}"
    )

    open_positions[symbol] = {'direction':direction, 'sl':sl, 'tp1':tp1, 'tp2':tp2}
    return msg


def analyze_symbol_mtf(symbol):
    msg5  = analyze_symbol(symbol, '5m')
    msg15 = analyze_symbol(symbol, '15m')
    if msg5 and msg15 and (("BUY" in msg5 and "BUY" in msg15) or ("SELL" in msg5 and "SELL" in msg15)):
        return msg15
    return None


def check_and_alert(symbol):
    logging.info(f"🔍 Checking {symbol} for signal…")
    msg = analyze_symbol_mtf(symbol)
    if msg:
        # لاگ خلاصه‌ی سیگنال
        signal_line = msg.splitlines()[2] if len(msg.splitlines())>=3 else msg
        logging.info(f"✅ Signal for {symbol}: {signal_line}")
        send_telegram_message(msg)
    else:
        logging.info(f"ℹ️ {symbol}: No signal at this time.")


def monitor_positions():
    while True:
        for sym, pos in list(open_positions.items()):
            df = get_data('15m', sym)
            price = df['close'].iloc[-1]
            dir   = pos['direction']
            if dir=='Long' and price>=pos['tp2'] or dir=='Short' and price<=pos['tp2']:
                del open_positions[sym]
            elif dir=='Long' and price<=pos['sl'] or dir=='Short' and price>=pos['sl']:
                del open_positions[sym]
        time.sleep(MONITOR_INTERVAL)


def monitor():
    last_hb = 0
    symbols = ["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
               "RENDERUSDT","TRUMPUSDT","FARTCOINUSDT","XLMUSDT",
               "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSDT","PENDLEUSDT"]
    while True:
        now = datetime.utcnow()
        hr  = (now.hour + 3) % 24; mn = now.minute
        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]:
            time.sleep(60); continue
        if time.time() - last_hb > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 *Bot live and scanning.*")
            last_hb = time.time()
        threads = []
        for sym in symbols:
            t = threading.Thread(target=check_and_alert, args=(sym,))
            t.start(); threads.append(t)
        for t in threads:
            t.join()
        # گزارش روزانه ساعت 23:55
        if hr==23 and mn>=55:
            send_telegram_message(
                f"📊 *Daily Report*\nSignals: {daily_signal_count}"
            )
        time.sleep(CHECK_INTERVAL)


@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."


if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

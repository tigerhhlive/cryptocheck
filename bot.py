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

# Strategy parameters (matching Pine Script)
EMA_FAST       = 9
EMA_SLOW       = 15
RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
ATR_PERIOD     = 14
SL_ATR_MULT    = 1.0
TP1_ATR_MULT   = 1.5
TP2_ATR_MULT   = 3.0
ADX_PERIOD     = 14
ADX_THRESHOLD  = 20
PA_THRESHOLD   = 0.7  # price action body/range
SIGNAL_COOLDOWN = 1800
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL     = 600
MONITOR_INTERVAL   = 120
SLEEP_HOURS        = (0, 7)

# Data requirements
MIN_BARS = max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD, MACD_SLOW + MACD_SIGNAL)

# State
last_signals = {}
daily_signal_count = 0
daily_win_count = 0
daily_loss_count = 0
open_positions = {}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            logging.error(f"Telegram error: {resp.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")
        
def send_csv_to_telegram(symbol):
    file_path = f"{symbol}_data.csv"
    try:
        with open(file_path, 'rb') as file:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
            files = {'document': file}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': f"{symbol} raw data"}
            response = requests.post(url, files=files, data=data)
            if response.status_code != 200:
                logging.error(f"❌ Failed to send CSV for {symbol}: {response.text}")
            else:
                logging.info(f"✅ Sent CSV file for {symbol} to Telegram.")
    except Exception as e:
        logging.error(f"❌ Error sending CSV: {e}")


def get_data(timeframe, symbol):
    agg = 5 if timeframe == '5m' else 15
    limit = 60
    fsym, tsym = symbol[:-4], 'USDT'
    url = 'https://min-api.cryptocompare.com/data/v2/histominute'
    params = dict(fsym=fsym, tsym=tsym, limit=limit, aggregate=agg, api_key=CRYPTOCOMPARE_API_KEY)
    res = requests.get(url, params=params, timeout=10).json()
    data = res.get('Data', {}).get('Data', [])
    df = pd.DataFrame(data)
    df['timestamp']    = pd.to_datetime(df['time'], unit='s')
    df['open']         = df['open'].astype(float)
    df['high']         = df['high'].astype(float)
    df['low']          = df['low'].astype(float)
    df['close']        = df['close'].astype(float)
    df['volume_coin']  = df['volumefrom'].astype(float)
    df['volume_quote'] = df['volumeto'].astype(float)
    return df[['timestamp','open','high','low','close','volume_coin','volume_quote']]


def check_cooldown(symbol, direction, idx):
    key = f"{symbol}_{direction}_{idx}"
    if last_signals.get(key):
        return False
    last_signals[key] = True
    return True


def detect_price_action(df):
    prev = df.iloc[-2]
    body = abs(prev.close - prev.open)
    rng  = prev.high - prev.low
    if rng > 0 and body/rng > PA_THRESHOLD:
        return 'bullish_marubozu' if prev.close>prev.open else 'bearish_marubozu'
    if len(df) >= 3:
        p2, p1 = df.iloc[-3], df.iloc[-2]
        if p2.close < p2.open and p1.close>p1.open and p1.close>p2.open and p1.open<p2.close:
            return 'bullish_engulfing'
        if p2.close > p2.open and p1.close<p1.open and p1.open>p2.close and p1.close<p2.open:
            return 'bearish_engulfing'
    return None


def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count
    df = get_data(timeframe, symbol)
    if df is None or len(df) < MIN_BARS:
        return None

    # ذخیره داده به صورت CSV
    csv_path = f"{symbol}_data.csv"
    df.to_csv(csv_path, index=False)

    # ارسال فایل به تلگرام
    send_csv_to_telegram(symbol)

    # Indicators
    df['ema_fast'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema_slow'] = ta.ema(df['close'], length=EMA_SLOW)
    df['rsi']      = ta.rsi(df['close'], length=RSI_PERIOD)
    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['macd_hist']= macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
    adx = ta.adx(df['high'], df['low'], df['close'], length=ADX_PERIOD)
    df['adx'] = adx[f'ADX_{ADX_PERIOD}']
    df['atr']= ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    prev, last = df.iloc[-2], df.iloc[-1]
    idx = df.index[-1]

    # EMA cross
    long_cross  = prev.ema_fast < prev.ema_slow  and last.ema_fast > last.ema_slow
    short_cross = prev.ema_fast > prev.ema_slow  and last.ema_fast < last.ema_slow
    if not (long_cross or short_cross):
        return None
    direction = 'Long' if long_cross else 'Short'

    # Filters
    if direction=='Long' and last.rsi < 50:  return None
    if direction=='Short' and last.rsi > 50: return None
    if direction=='Long' and last.macd_hist < 0:  return None
    if direction=='Short' and last.macd_hist > 0: return None
    pa = detect_price_action(df)
    if not pa:  return None
    if last.adx < ADX_THRESHOLD: return None

    if not check_cooldown(symbol, direction, idx): return None

    entry = last.close
    atr_val= last.atr
    if direction=='Long':
        sl  = entry - atr_val * SL_ATR_MULT
        tp1 = entry + atr_val * TP1_ATR_MULT
        tp2 = entry + atr_val * TP2_ATR_MULT
    else:
        sl  = entry + atr_val * SL_ATR_MULT
        tp1 = entry - atr_val * TP1_ATR_MULT
        tp2 = entry - atr_val * TP2_ATR_MULT

    daily_signal_count += 1
    open_positions[symbol] = dict(direction=direction, sl=sl, tp1=tp1, tp2=tp2)

    stars = '🔥'*3
    msg = (f"🚨 *AI Signal Alert*\n"
           f"*Symbol:* `{symbol}`\n"
           f"*Signal:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
           f"*Entry:* `{entry:.4f}`  *SL:* `{sl:.4f}`  *TP1:* `{tp1:.4f}`  *TP2:* `{tp2:.4f}`\n"
           f"*RSI:* {last.rsi:.1f}  *MACD_h:* {last.macd_hist:.4f}\n"
           f"*Pattern:* {pa}\n"
           f"*Strength:* {stars}")
    return msg


def analyze_symbol_mtf(symbol):
    m5 = analyze_symbol(symbol,'5m')
    m15= analyze_symbol(symbol,'15m')
    if m5 and m15 and (('BUY' in m5 and 'BUY' in m15) or ('SELL' in m5 and 'SELL' in m15)):
        return m15
    return None


def check_and_alert(symbol):
    logging.info(f"🔍 Checking {symbol} for signal...")
    msg = analyze_symbol_mtf(symbol)
    if msg:
        logging.info(f"{symbol}: ✅ Signal detected and sent.")
        send_telegram_message(msg)
    else:
        logging.info(f"{symbol}: ❌ No signal at this time.")


def monitor_positions():
    global daily_win_count, daily_loss_count
    while True:
        for s,pos in list(open_positions.items()):
            df = get_data('15m', s)
            price = df['close'].iloc[-1]
            dir   = pos['direction']
            if dir=='Long':
                if price>=pos['tp2']: daily_win_count+=1; del open_positions[s]
                elif price<=pos['sl']: daily_loss_count+=1; del open_positions[s]
            else:
                if price<=pos['tp2']: daily_win_count+=1; del open_positions[s]
                elif price>=pos['sl']: daily_loss_count+=1; del open_positions[s]
        time.sleep(MONITOR_INTERVAL)


def report_daily():
    total= daily_win_count+daily_loss_count
    wr = round(daily_win_count/total*100,1) if total else 0.0
    send_telegram_message(
        f"📊 *Daily Report*\n"
        f"Signals: {daily_signal_count}\n"
        f"Wins: {daily_win_count}  Losses: {daily_loss_count}\n"
        f"Winrate: {wr}%")


def monitor():
    last_hb=0
    syms = ["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
            "RENDERUSDT","TRUMPUSDT","FARTCOINUSDT","XLMUSDT",
            "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSDT","PENDLEUSDT"]
    while True:
        now = datetime.utcnow(); hr=(now.hour+3)%24; mn=now.minute
        if SLEEP_HOURS[0]<=hr<SLEEP_HOURS[1]: time.sleep(60); continue
        if time.time()-last_hb>HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot live and scanning.")
            last_hb = time.time()
        threads=[]
        for s in syms:
            t=threading.Thread(target=check_and_alert,args=(s,))
            t.start(); threads.append(t)
        for t in threads: t.join()
        if hr==23 and mn>=55: report_daily()
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

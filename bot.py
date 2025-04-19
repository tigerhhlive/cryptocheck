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

# ────── Configuration ──────
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID     = os.environ.get('TELEGRAM_CHAT_ID')

# Strategy parameters
EMA_FAST        = 9
EMA_SLOW        = 15
EMA_TREND       = 200    # for long-term trend simulation on 15m
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
ATR_PERIOD      = 14
ATR_SL_MULT     = 1.2
TP1_MULTIPLIER  = 1.8
TP2_MULTIPLIER  = 2.8
ADX_PERIOD      = 14
# thresholds for balanced vs strict modes
ADX_THRESHOLD_BALANCED = 20
ADX_THRESHOLD_STRICT   = 25
VOL_MA_PERIOD_BALANCED = 15
VOL_MA_PERIOD_STRICT   = 20
SESSION_START_UTC      = 7    # allow signals from 07:00 UTC
SESSION_END_UTC        = 19   # until 19:00 UTC
SIGNAL_COOLDOWN        = 1800  # seconds
CHECK_INTERVAL         = 600   # seconds between scans
HEARTBEAT_INTERVAL     = 7200
SLEEP_HOURS            = (0, 7)
MIN_BARS               = max(EMA_TREND, ATR_PERIOD, MACD_SLOW + MACD_SIGNAL)

# State
last_signal_time = {}   # key: symbol_mode_direction -> timestamp
last_signal_dir  = {}   # key: symbol -> last direction

# Logging
default_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=default_format)


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")


def get_data(timeframe: str, symbol: str) -> pd.DataFrame | None:
    fsym, tsym = symbol[:-4], 'USDT'
    if timeframe in ('1h', '60m'):
        url = 'https://min-api.cryptocompare.com/data/v2/histohour'
        limit = 48
        params = {'fsym': fsym, 'tsym': tsym, 'limit': limit, 'api_key': CRYPTOCOMPARE_API_KEY}
    else:
        url = 'https://min-api.cryptocompare.com/data/v2/histominute'
        agg = 5 if timeframe == '5m' else 15
        limit = 100
        params = {'fsym': fsym, 'tsym': tsym, 'limit': limit, 'aggregate': agg, 'api_key': CRYPTOCOMPARE_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get('Data', {}).get('Data', [])
        if not isinstance(data, list) or not data:
            logging.warning(f"{symbol} {timeframe}: no data")
            return None
        df = pd.DataFrame(data).dropna()
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        for col in ['open','high','low','close','volumeto']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volumeto']
        return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        logging.error(f"Error fetching {symbol} {timeframe}: {e}")
        return None


def in_session() -> bool:
    hr = datetime.utcnow().hour
    return SESSION_START_UTC <= hr < SESSION_END_UTC


def check_cooldown(symbol: str, mode: str, direction: str) -> bool:
    key = f"{symbol}_{mode}_{direction}"
    now = time.time()
    last = last_signal_time.get(key, 0)
    if now - last < SIGNAL_COOLDOWN:
        return False
    # prevent same-dir repeated
    if last_signal_dir.get((symbol, mode)) == direction:
        return False
    last_signal_time[key] = now
    last_signal_dir[(symbol, mode)] = direction
    return True


def analyze_symbol_mode(symbol: str, mode: str, timeframe: str = '15m') -> str | None:
    """
    mode: 'balanced' or 'strict'
    applies corresponding thresholds and returns signal message or None
    """
    if not in_session(): return None
    df = get_data(timeframe, symbol)
    if df is None or len(df) < MIN_BARS: return None

    # thresholds by mode
    adx_th  = ADX_THRESHOLD_BALANCED if mode=='balanced' else ADX_THRESHOLD_STRICT
    vol_ma  = VOL_MA_PERIOD_BALANCED  if mode=='balanced' else VOL_MA_PERIOD_STRICT

    # indicators
    df['EMA_f'] = ta.ema(df['close'], length=EMA_FAST)
    df['EMA_s'] = ta.ema(df['close'], length=EMA_SLOW)
    df['EMA_t'] = ta.ema(df['close'], length=EMA_TREND)
    df['RSI']   = ta.rsi(df['close'], length=RSI_PERIOD)
    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['MACDh']= macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    df['ATR']  = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    df['vol_ma'] = df['volume'].rolling(vol_ma).mean()
    adx = ta.adx(df['high'], df['low'], df['close'], length=ADX_PERIOD)['ADX_14'].iloc[-1]

    candle  = df.iloc[-2]
    confirm = df.iloc[-1]
    idx     = df.index[-1]

    # long-term trend
    if candle['close'] <= candle['EMA_t']:
        return None
    # volume
    if confirm['volume'] < df['vol_ma'].iloc[-1]:
        return None
    # adx
    if adx < adx_th:
        return None
    # entry conds
    long_x  = candle['EMA_f'] > candle['EMA_s']
    short_x = candle['EMA_f'] < candle['EMA_s']
    long_c  = long_x  and candle['RSI']>=50 and candle['MACDh']>0 and confirm['close']>confirm['open']
    short_c = short_x and candle['RSI']<=50 and candle['MACDh']<0 and confirm['close']<confirm['open']
    direction = 'Long' if long_c else 'Short' if short_c else None
    if not direction: return None
    # cooldown
    if not check_cooldown(symbol, mode, direction):
        return None

    # calc SL/TP
    entry= candle['close']; atr= max(candle['ATR'], entry*0.001)
    if direction=='Long':
        sl = entry-atr*ATR_SL_MULT; tp1=entry+atr*TP1_MULTIPLIER; tp2= tp1+(tp1-entry)*1.2
    else:
        sl = entry+atr*ATR_SL_MULT; tp1=entry-atr*TP1_MULTIPLIER; tp2= tp1-(entry-tp1)*1.2
    rr = abs(tp1-entry)/abs(entry-sl)

    # message
    return (
        f"*Signal ({mode.capitalize()}) Detected*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Type:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Entry:* `{entry:.6f}`  *SL:* `{sl:.6f}`  *TP1:* `{tp1:.6f}`  *TP2:* `{tp2:.6f}`\n"
        f"*RR:* `{rr:.2f}X`  *ADX:* {adx:.1f}  *Vol:* {confirm['volume']:.0f}\n"
    )


def analyze_symbol(symbol: str) -> None:
    # try strict first
    msg = analyze_symbol_mode(symbol, 'strict') or analyze_symbol_mode(symbol, 'balanced')
    return msg


def check_and_alert(symbol: str):
    logging.info(f"🔍 {symbol}…")
    msg = analyze_symbol(symbol)
    if msg:
        send_telegram_message(msg)


def monitor():
    syms=["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT", "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT", "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT", "PENDLEUSDT"]
    last_hb=0
    while True:
        hr=(datetime.utcnow().hour+3)%24
        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]: time.sleep(60); continue
        if time.time()-last_hb>HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot alive and scanning.")
            last_hb=time.time()
        threads=[]
        for s in syms:
            t=threading.Thread(target=check_and_alert,args=(s,)); t.start(); threads.append(t)
        for t in threads: t.join()
        time.sleep(CHECK_INTERVAL)


def monitor_positions():
    pass

@app.route('/')
def home(): return "✅ Crypto Signal Bot is running."

if __name__=='__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port=int(os.environ.get("PORT",8080))
    app.run(host='0.0.0.0',port=port)

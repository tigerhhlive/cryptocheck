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
API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT = os.environ.get('TELEGRAM_CHAT_ID')

# Strategy parameters
EMA_FAST = 9
EMA_SLOW = 15
RSI_LEN = 14
ATR_LEN = 14
OB_LOOKBACK = 10     # Swing lookback for Order Block detection (bars)
SIGNAL_COOLDOWN = 1800
CHECK_INTERVAL = 600
HEARTBEAT_INTERVAL = 7200
SLEEP_HOURS = (0, 7)

# State tracking
last_signal_bar = {}
open_positions = {}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def send_msg(text):
    """Send a Markdown message to Telegram chat."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT, 'text': text, 'parse_mode': 'Markdown'}
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            logging.error(f"Telegram error: {resp.text}")
    except Exception as e:
        logging.error(f"Exception sending Telegram message: {e}")


def get_data(symbol, agg=15):
    """Fetch OHLCV minute data from CryptoCompare."""
    try:
        url = 'https://min-api.cryptocompare.com/data/v2/histominute'
        params = {
            'fsym': symbol[:-4], 'tsym': 'USDT',
            'limit': 100, 'aggregate': agg, 'api_key': API_KEY
        }
        res = requests.get(url, params=params, timeout=10).json()
        data = res.get('Data', {}).get('Data', [])
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def detect_swing_highs(df):
    df['is_swing_high'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
    return df[df['is_swing_high']]


def detect_swing_lows(df):
    df['is_swing_low'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
    return df[df['is_swing_low']]


def detect_rsi_divergence(df):
    # Bearish divergence: price higher highs + RSI lower highs
    sh = detect_swing_highs(df)
    if len(sh) >= 2:
        last = sh['high'].iloc[-2:]
        rsi = df['RSI'].loc[last.index]
        if last.iloc[1] > last.iloc[0] and rsi.iloc[1] < rsi.iloc[0]:
            return 'bear'
    # Bullish divergence: price lower lows + RSI higher lows
    sl = detect_swing_lows(df)
    if len(sl) >= 2:
        lows = sl['low'].iloc[-2:]
        rsi = df['RSI'].loc[lows.index]
        if lows.iloc[1] < lows.iloc[0] and rsi.iloc[1] > rsi.iloc[0]:
            return 'bull'
    return None


def analyze(symbol):
    df = get_data(symbol, agg=15)
    if df.empty or len(df) < max(EMA_SLOW, RSI_LEN, ATR_LEN, OB_LOOKBACK*2):
        return

    # Calculate indicators
    df['EMA9'] = ta.ema(df['close'], length=EMA_FAST)
    df['EMA15'] = ta.ema(df['close'], length=EMA_SLOW)
    df['RSI'] = ta.rsi(df['close'], length=RSI_LEN)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_LEN)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # EMA crossover
    long_cross = (prev['EMA9'] < prev['EMA15'] and last['EMA9'] > last['EMA15'])
    short_cross = (prev['EMA9'] > prev['EMA15'] and last['EMA9'] < last['EMA15'])
    if not (long_cross or short_cross):
        logging.info(f"{symbol}: No EMA9/15 cross")
        return
    direction = 'Long' if long_cross else 'Short'

    # RSI divergence
    div = detect_rsi_divergence(df)
    if direction == 'Long' and div != 'bull':
        logging.info(f"{symbol}: No bullish RSI divergence")
        return
    if direction == 'Short' and div != 'bear':
        logging.info(f"{symbol}: No bearish RSI divergence")
        return

    # Detect Order Block
    if direction == 'Short':
        sh = detect_swing_highs(df)
        if sh.empty: return
        top = sh['high'].iloc[-1]
        window = df.iloc[-(OB_LOOKBACK+1):-1]
        bot = window['low'].min()
        if not (last['close'] <= top and last['close'] >= bot):
            logging.info(f"{symbol}: Price not in bearish OB zone [{bot:.4f}–{top:.4f}]")
            return
        ob_top, ob_bot = top, bot
    else:
        sl = detect_swing_lows(df)
        if sl.empty: return
        bot = sl['low'].iloc[-1]
        window = df.iloc[-(OB_LOOKBACK+1):-1]
        top = window['high'].max()
        if not (last['close'] >= bot and last['close'] <= top):
            logging.info(f"{symbol}: Price not in bullish OB zone [{bot:.4f}–{top:.4f}]")
            return
        ob_top, ob_bot = top, bot

    # Cooldown per bar index
    bar_idx = df.index[-1]
    key = f"{symbol}_{direction}"
    if last_signal_bar.get(key) == bar_idx:
        logging.info(f"{symbol}: Cooldown active for {direction}")
        return
    last_signal_bar[key] = bar_idx

    # Calculate SL/TP
    entry = last['close']
    atr = last['ATR']
    if direction == 'Long':
        sl = ob_bot - 0.5 * (ob_top - ob_bot)
        tp1 = entry + atr * 1
        tp2 = entry + atr * 2
    else:
        sl = ob_top + 0.5 * (ob_top - ob_bot)
        tp1 = entry - atr * 1
        tp2 = entry - atr * 2

    # Send signal message
    msg = (
        f"🚨 *AI SIGNAL Alert*\n"
        f"*{symbol}* - {'🟢 BUY' if direction=='Long' else '🔴 SELL'} @ {entry:.4f}\n"
        f"SL: {sl:.4f} TP1: {tp1:.4f} TP2: {tp2:.4f}\n"
        f"OB Zone: [{ob_bot:.4f}–{ob_top:.4f}]   RSI Divergence: *{div.title()}*"
    )
    logging.info(f"{symbol}: Signal {direction} at {entry:.4f}")
    send_msg(msg)


def worker():
    symbols = ["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT","ADAUSDT","XLMUSDT"]
    logging.info("Worker started, symbols: %s", symbols)
    # Send immediate heartbeat
    send_msg("🤖 *Bot launched and scanning signals...*")
    last_hb = time.time()

    while True:
        now = datetime.utcnow()
        hr = (now.hour + 3) % 24
        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]:
            time.sleep(60)
            continue
        # Heartbeat every HEARTBEAT_INTERVAL
        if time.time() - last_hb > HEARTBEAT_INTERVAL:
            send_msg("🤖 *Bot is live and scanning.*")
            last_hb = time.time()
        # Scan symbols
        for sym in symbols:
            logging.info(f"🔍 Checking {sym} for signal...")
            try:
                analyze(sym)
            except Exception as e:
                logging.error(f"Error in analyze({sym}): {e}")
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == '__main__':
    threading.Thread(target=worker, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

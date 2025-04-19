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
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN      = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID        = os.environ.get('TELEGRAM_CHAT_ID')

# Strategy parameters
EMA_FAST        = 9
EMA_SLOW        = 15
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
ATR_PERIOD      = 14
ATR_MULTIPLIER_SL = 1.2
TP1_MULTIPLIER  = 1.8
TP2_MULTIPLIER  = 2.8
MIN_PERCENT_RISK = 0.03
MIN_ATR         = 0.001
SIGNAL_COOLDOWN = 1800
CHECK_INTERVAL  = 600
HEARTBEAT_INTERVAL = 7200
SLEEP_HOURS     = (0, 7)
MIN_BARS        = 30

# State
last_signals      = {}
daily_signal_count = 0

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")


def get_data(timeframe: str, symbol: str) -> pd.DataFrame | None:
    """Fetch OHLCV minute data from CryptoCompare."""
    aggregate = 5 if timeframe == '5m' else 15
    params = {
        'fsym': symbol[:-4],
        'tsym': 'USDT',
        'limit': 100,
        'aggregate': aggregate,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    try:
        r = requests.get("https://min-api.cryptocompare.com/data/v2/histominute",
                         params=params, timeout=10)
        data = r.json().get("Data", {}).get("Data", [])
        if not isinstance(data, list) or not data:
            logging.warning(f"No data for {symbol}")
            return None
        df = pd.DataFrame(data).dropna()
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume']    = df['volumeto']
        return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {e}")
        return None


def check_cooldown(symbol: str, direction: str) -> bool:
    """Prevent duplicate signals on same candle."""
    key = f"{symbol}_{direction}"
    now = time.time()
    last = last_signals.get(key, 0)
    if now - last < SIGNAL_COOLDOWN:
        return False
    last_signals[key] = now
    return True


def analyze_symbol(symbol: str, timeframe: str = '15m') -> str | None:
    """
    Core signal logic: EMA crossover + RSI + MACD hist + confirmation candle.
    Returns a Markdown message if a signal is detected.
    """
    global daily_signal_count
    df = get_data(timeframe, symbol)
    if df is None or len(df) < MIN_BARS:
        return None

    # Indicators
    df['EMA_FAST']  = ta.ema(df['close'], length=EMA_FAST)
    df['EMA_SLOW']  = ta.ema(df['close'], length=EMA_SLOW)
    df['RSI']       = ta.rsi(df['close'], length=RSI_PERIOD)
    macd = ta.macd(df['close'],
                   fast=MACD_FAST,
                   slow=MACD_SLOW,
                   signal=MACD_SIGNAL)
    df['MACD_Hist'] = macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    df['ATR']       = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    # Use the penultimate candle for entry conditions,
    # and the last candle for confirmation direction.
    candle  = df.iloc[-2]
    confirm = df.iloc[-1]
    entry   = candle['close']
    atr_val = max(candle['ATR'], entry * MIN_PERCENT_RISK, MIN_ATR)

    long_cond  = (candle['EMA_FAST'] > candle['EMA_SLOW']
                  and candle['RSI'] >= 50
                  and candle['MACD_Hist'] > 0)
    short_cond = (candle['EMA_FAST'] < candle['EMA_SLOW']
                  and candle['RSI'] <= 50
                  and candle['MACD_Hist'] < 0)

    direction = None
    if long_cond and confirm['close'] > confirm['open']:
        direction = 'Long'
    elif short_cond and confirm['close'] < confirm['open']:
        direction = 'Short'

    if not direction or not check_cooldown(symbol, direction):
        return None

    # Prepare targets & stop loss
    daily_signal_count += 1
    if direction == 'Long':
        sl  = entry - atr_val * ATR_MULTIPLIER_SL
        tp1 = entry + atr_val * TP1_MULTIPLIER
        tp2 = tp1  + (tp1  - entry) * 1.2
    else:
        sl  = entry + atr_val * ATR_MULTIPLIER_SL
        tp1 = entry - atr_val * TP1_MULTIPLIER
        tp2 = tp1  - (entry - tp1)  * 1.2

    rr_ratio = abs(tp1 - entry) / abs(entry - sl)

    msg = (
        "*Signal Detected*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Type:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Entry:* `{entry:.6f}`\n"
        f"*Stop Loss:* `{sl:.6f}`\n"
        f"*TP1:* `{tp1:.6f}`\n"
        f"*TP2:* `{tp2:.6f}`\n"
        f"*RR:* `{rr_ratio:.2f}X`\n"
    )
    return msg


def analyze_symbol_mtf(symbol: str) -> str | None:
    """Multi-timeframe check: prefer 15m over 5m."""
    m15 = analyze_symbol(symbol, '15m')
    if m15:
        return m15
    return analyze_symbol(symbol, '5m')


def check_and_alert(symbol: str):
    logging.info(f"🔍 Checking {symbol} for signal...")
    msg = analyze_symbol_mtf(symbol)
    if msg:
        send_telegram_message(msg)


def monitor():
    symbols = [
        "BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
        "RENDERUSDT","TRUMPUSDT","FARTCOINUSDT","XLMUSDT",
        "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSDT","PENDLEUSDT"
    ]
    last_hb = 0
    while True:
        now = datetime.utcnow()
        hour_tehran = (now.hour + 3) % 24
        # Sleep during quiet hours
        if SLEEP_HOURS[0] <= hour_tehran < SLEEP_HOURS[1]:
            time.sleep(60)
            continue

        # Heartbeat
        if time.time() - last_hb > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot is alive and checking...")
            last_hb = time.time()

        # Parallel symbol checks
        threads = []
        for sym in symbols:
            t = threading.Thread(target=check_and_alert, args=(sym,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        time.sleep(CHECK_INTERVAL)


def monitor_positions():
    # Placeholder for future PnL tracking
    pass


@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."


if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

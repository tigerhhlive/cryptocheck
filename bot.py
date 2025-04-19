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
EMA_TREND       = 200    # for simulated higher‑TF trend on 15m
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
ATR_PERIOD      = 14
ATR_MULTIPLIER_SL = 1.2
TP1_MULTIPLIER  = 1.8
TP2_MULTIPLIER  = 2.8
ADX_PERIOD      = 14
ADX_THRESHOLD   = 25
VOL_MA_PERIOD   = 20
SESSION_START_UTC = 7    # allow signals from 07:00 UTC
SESSION_END_UTC   = 19   # until 19:00 UTC
SIGNAL_COOLDOWN  = 1800  # seconds
CHECK_INTERVAL   = 600   # seconds between scans
HEARTBEAT_INTERVAL = 7200
SLEEP_HOURS     = (0, 7) # Tehran hours to suspend checks
MIN_BARS        = max(EMA_TREND, VOL_MA_PERIOD, ATR_PERIOD, MACD_SLOW + MACD_SIGNAL)

# State
last_signal_time = {}   # key: symbol_direction -> timestamp
last_signal_dir  = {}   # key: symbol -> last direction
daily_signal_count = 0

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
    """
    Fetch minute or hourly data.
    Use histominute for 5m/15m, histohour for 1h.
    """
    fsym, tsym = symbol[:-4], 'USDT'
    if timeframe in ('1h', '60m'):
        url = 'https://min-api.cryptocompare.com/data/v2/histohour'
        limit = 48
        params = {'fsym': fsym, 'tsym': tsym, 'limit': limit, 'api_key': CRYPTOCOMPARE_API_KEY}
    else:
        url = 'https://min-api.cryptocompare.com/data/v2/histominute'
        agg = 5 if timeframe == '5m' else 15
        limit = 100
        params = {
            'fsym':       fsym,
            'tsym':       tsym,
            'limit':      limit,
            'aggregate':  agg,
            'api_key':    CRYPTOCOMPARE_API_KEY
        }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get('Data', {}).get('Data', [])
        if not isinstance(data, list) or not data:
            logging.warning(f"{symbol} {timeframe}: no data")
            return None
        df = pd.DataFrame(data).dropna()
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['open']      = df['open'].astype(float)
        df['high']      = df['high'].astype(float)
        df['low']       = df['low'].astype(float)
        df['close']     = df['close'].astype(float)
        # use quote volume
        df['volume']    = df['volumeto'].astype(float)
        return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        logging.error(f"Error fetching {symbol} {timeframe}: {e}")
        return None


def in_session() -> bool:
    """Allow signals only during high-liquidity UTC hours."""
    hr = datetime.utcnow().hour
    return SESSION_START_UTC <= hr < SESSION_END_UTC


def check_cooldown(symbol: str, direction: str) -> bool:
    """Prevent duplicate signals: per-candle and per-direction."""
    key = f"{symbol}_{direction}"
    now = time.time()
    last = last_signal_time.get(key, 0)
    if now - last < SIGNAL_COOLDOWN:
        return False
    # also prevent same-dir until opposite dir occurs
    if last_signal_dir.get(symbol) == direction:
        return False
    last_signal_time[key] = now
    last_signal_dir[symbol] = direction
    return True


def analyze_symbol(symbol: str, timeframe: str = '15m') -> str | None:
    global daily_signal_count

    # session filter
    if not in_session():
        logging.info(f"{symbol}: outside session hours")
        return None

    df = get_data(timeframe, symbol)
    if df is None or len(df) < MIN_BARS:
        return None

    # Indicators
    df['EMA_fast']  = ta.ema(df['close'], length=EMA_FAST)
    df['EMA_slow']  = ta.ema(df['close'], length=EMA_SLOW)
    df['EMA_trend'] = ta.ema(df['close'], length=EMA_TREND)
    df['RSI']       = ta.rsi(df['close'], length=RSI_PERIOD)
    macd = ta.macd(df['close'],
                   fast=MACD_FAST,
                   slow=MACD_SLOW,
                   signal=MACD_SIGNAL)
    df['MACD_hist'] = macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    df['ATR']       = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    df['vol_ma']    = df['volume'].rolling(VOL_MA_PERIOD).mean()

    candle  = df.iloc[-2]
    confirm = df.iloc[-1]

    # Trend filter (simulated higher-TF)
    if candle['close'] <= candle['EMA_trend']:
        logging.info(f"{symbol}: against long-term trend")
        return None

    # Volume filter
    if confirm['volume'] < df['vol_ma'].iloc[-1]:
        logging.info(f"{symbol}: low volume")
        return None

    # ADX filter
    adx = ta.adx(df['high'], df['low'], df['close'], length=ADX_PERIOD)['ADX_14'].iloc[-1]
    if adx < ADX_THRESHOLD:
        logging.info(f"{symbol}: low ADX {adx:.1f}")
        return None

    # Entry conditions on candle
    long_cross  = candle['EMA_fast'] > candle['EMA_slow']
    short_cross = candle['EMA_fast'] < candle['EMA_slow']
    long_cond   = long_cross and candle['RSI'] >= 50 and candle['MACD_hist'] > 0
    short_cond  = short_cross and candle['RSI'] <= 50 and candle['MACD_hist'] < 0

    direction = None
    if long_cond and confirm['close'] > confirm['open']:
        direction = 'Long'
    elif short_cond and confirm['close'] < confirm['open']:
        direction = 'Short'
    else:
        logging.info(f"{symbol}: no valid EMA/RSI/MACD/confirm")
        return None

    # Cooldown & dedup
    if not check_cooldown(symbol, direction):
        logging.info(f"{symbol}: cooldown or duplicate")
        return None

    # SL/TP calculation
    entry = candle['close']
    atr   = max(candle['ATR'], entry * MIN_PERCENT_RISK, 0.001)
    if direction == 'Long':
        sl  = entry - atr * ATR_MULTIPLIER_SL
        tp1 = entry + atr * TP1_MULTIPLIER
        tp2 = tp1  + (tp1  - entry) * 1.2
    else:
        sl  = entry + atr * ATR_MULTIPLIER_SL
        tp1 = entry - atr * TP1_MULTIPLIER
        tp2 = tp1  - (entry - tp1)  * 1.2

    rr = abs(tp1 - entry) / abs(entry - sl)
    daily_signal_count += 1

    # Build message
    msg = (
        f"*Signal Detected*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Type:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Entry:* `{entry:.6f}`\n"
        f"*SL:* `{sl:.6f}`\n"
        f"*TP1:* `{tp1:.6f}`\n"
        f"*TP2:* `{tp2:.6f}`\n"
        f"*RR:* `{rr:.2f}X`\n"
        f"*ADX:* {adx:.1f}\n"
        f"*Vol:* {confirm['volume']:.0f} (MA{VOL_MA_PERIOD})\n"
    )
    return msg


def analyze_symbol_mtf(symbol: str) -> str | None:
    # prefer 15m, fallback to 5m
    m15 = analyze_symbol(symbol, '15m')
    return m15 or analyze_symbol(symbol, '5m')


def check_and_alert(symbol: str):
    logging.info(f"🔍 Checking {symbol}…")
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
        # suspend during Tehran sleep hours
        hr_tehran = (datetime.utcnow().hour + 3) % 24
        if SLEEP_HOURS[0] <= hr_tehran < SLEEP_HOURS[1]:
            time.sleep(60)
            continue

        # heartbeat
        if time.time() - last_hb > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot is alive and scanning.")
            last_hb = time.time()

        # scan symbols in threads
        threads = []
        for sym in symbols:
            t = threading.Thread(target=check_and_alert, args=(sym,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        time.sleep(CHECK_INTERVAL)


def monitor_positions():
    # placeholder for future PnL tracking
    pass


@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."


if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

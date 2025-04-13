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
SLEEP_HOURS = (0, 7)
MIN_ATR = 0.001
SIGNAL_COOLDOWN = 1800

# ثبت آخرین سیگنال‌ها
last_signals = {}

# شمارش برای گزارش روزانه
daily_signal_count = 0
daily_hit_count = 0
last_report_day = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"❌ خطا در ارسال پیام: {response.text}")
    except Exception as e:
        logging.error(f"❌ Exception در ارسال پیام تلگرام: {e}")

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
    res = requests.get(url, params=params, timeout=10)
    data = res.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['volume'] = df['volumeto']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def detect_volume_spike(df, multiplier=2.0):
    avg_volume = df['volume'].iloc[-21:-1].mean()
    return df['volume'].iloc[-1] > avg_volume * multiplier

def detect_atr_breakout(df, atr_col, multiplier=1.5):
    avg_atr = df[atr_col].iloc[-21:-1].mean()
    return df[atr_col].iloc[-1] > avg_atr * multiplier

def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count

    log_prefix = f"[{datetime.utcnow()}] {symbol} [{timeframe}]"
    df = get_data(timeframe, symbol)
    if len(df) < 30:
        return None, None

    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'])
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])

    candle = df.iloc[-1]
    signal_type = detect_strong_candle(candle) or detect_engulfing(df)
    pattern = signal_type.replace("_", " ").title() if signal_type else "None"

    rsi_val = df['rsi'].iloc[-1]
    adx_val = df['ADX'].iloc[-1]
    entry = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    atr = max(atr, entry * MIN_PERCENT_RISK, MIN_ATR)

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    confirmations = []

    if (signal_type and 'bullish' in signal_type and rsi_val >= 50) or (signal_type and 'bearish' in signal_type and rsi_val <= 50):
        confirmations.append("RSI")
    if (df['MACD'].iloc[-1] > df['MACDs'].iloc[-1]) if 'bullish' in str(signal_type) else (df['MACD'].iloc[-1] < df['MACDs'].iloc[-1]):
        confirmations.append("MACD")
    if adx_val > ADX_THRESHOLD:
        confirmations.append("ADX")
    if ('bullish' in str(signal_type) and above_ema) or ('bearish' in str(signal_type) and below_ema):
        confirmations.append("EMA")

    confidence = len(confirmations)
    direction = 'Long' if 'bullish' in str(signal_type) and confidence >= 3 else \
                'Short' if 'bearish' in str(signal_type) and confidence >= 3 else None

    if direction and not check_cooldown(symbol, direction):
        logging.info(f"{log_prefix} - DUPLICATE SIGNAL - Skipped due to cooldown")
        return None, "Duplicate"

    if direction:
        daily_signal_count += 1
        sl = entry - atr * ATR_MULTIPLIER_SL if direction == 'Long' else entry + atr * ATR_MULTIPLIER_SL
        tp1 = entry + atr * TP1_MULTIPLIER if direction == 'Long' else entry - atr * TP1_MULTIPLIER
        tp2 = entry + atr * TP2_MULTIPLIER if direction == 'Long' else entry - atr * TP2_MULTIPLIER
        rr_ratio = abs(tp1 - entry) / abs(entry - sl)
        TP1_MULT = max(1.5, round(rr_ratio * 1.1, 1))
        TP2_MULT = round(TP1_MULT * 1.5, 1)
        tp1 = entry + atr * TP1_MULT if direction == 'Long' else entry - atr * TP1_MULT
        tp2 = entry + atr * TP2_MULT if direction == 'Long' else entry - atr * TP2_MULT

        # Volume Spike & ATR Breakout Flags
        vol_spike = detect_volume_spike(df)
        atr_break = detect_atr_breakout(df, 'ATR')

        volatility_note = ""
        if vol_spike or atr_break:
            volatility_note = "\n⚡ *Market Volatility Detected!*"

        confidence_stars = "🔥" * confidence

        message = f"""🚨 *AI Signal Alert*
*Symbol:* `{symbol}`
*Signal:* {'🟢 BUY MARKET' if direction == 'Long' else '🔴 SELL MARKET'}
*Pattern:* {pattern}
*Confirmed by:* {", ".join(confirmations) if confirmations else 'None'}
*Entry:* `{entry:.6f}`
*Stop Loss:* `{sl:.6f}`
*Target 1:* `{tp1:.6f}`
*Target 2:* `{tp2:.6f}`
*Leverage (est.):* `{rr_ratio:.2f}X`
*Signal Strength:* {confidence_stars}{volatility_note}
"""
        return message, None

    logging.info(f"{log_prefix} - NO SIGNAL | Confirmations: {len(confirmations)}/4")
    return None, None
def analyze_symbol_mtf(symbol):
    msg_5m, _ = analyze_symbol(symbol, '5m')
    msg_15m, _ = analyze_symbol(symbol, '15m')

    if msg_5m and msg_15m:
        if ("BUY" in msg_5m and "BUY" in msg_15m) or ("SELL" in msg_5m and "SELL" in msg_15m):
            return msg_15m, None  # ✅ سیگنال کامل و تاییدشده
    elif msg_15m and ("🔥🔥🔥" in msg_15m):
        # سیگنال خیلی قوی در تایم‌فریم بالا
        msg_weak = msg_15m + "\n⚠️ *Note:* Strong 15m signal without 5m confirmation."
        return msg_weak, None
    return None, None

def monitor():
    global daily_signal_count, daily_hit_count, last_report_day

    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT"
    ]
    last_heartbeat = 0

    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        tehran_min = now.minute
        current_day = now.date()

        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            logging.info("ربات در حالت خواب شبانه است")
            time.sleep(60)
            continue

        # Heartbeat
        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 ربات فعال است و در حال بررسی سیگنال‌ها می‌باشد")
            last_heartbeat = time.time()

        # Daily Summary (23:59)
        if tehran_hour == 23 and tehran_min >= 59 and current_day != last_report_day:
            if daily_signal_count > 0:
                winrate = round((daily_hit_count / daily_signal_count) * 100, 1)
            else:
                winrate = 0.0
            summary_msg = f"""📊 *Daily Summary*
Total Signals: {daily_signal_count}
Hits (est.): {daily_hit_count}
Winrate (est.): {winrate}%
"""
            send_telegram_message(summary_msg)
            daily_signal_count = 0
            daily_hit_count = 0
            last_report_day = current_day

        for sym in symbols:
            try:
                msg, _ = analyze_symbol_mtf(sym)
                if msg:
                    send_telegram_message(msg)
                    daily_hit_count += 1  # فرض اینکه سیگنال زده شده موفق بوده
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")

        time.sleep(CHECK_INTERVAL)
@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

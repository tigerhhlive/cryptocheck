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
MONITOR_INTERVAL = 120
SLEEP_HOURS = (0, 7)
MIN_ATR = 0.001
SIGNAL_COOLDOWN = 1800

last_signals = {}
daily_signal_count = 0
daily_hit_count = 0
last_report_day = None
open_positions = {}
tp1_count = 0
tp2_count = 0
sl_count = 0

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

def monitor_positions():
    global tp1_count, tp2_count, sl_count, last_report_day, daily_signal_count, daily_hit_count
    while True:
        for symbol, pos in list(open_positions.items()):
            try:
                df = get_data('15m', symbol)
                current_price = df['close'].iloc[-1]
                direction = pos['direction']

                if direction == 'Long':
                    if current_price >= pos['tp2']:
                        send_telegram_message(f"✅ *{symbol} TP2 Hit* - Full Target Reached. Position Closed.")
                        tp2_count += 1
                        del open_positions[symbol]
                    elif current_price >= pos['tp1']:
                        send_telegram_message(f"🎯 *{symbol} TP1 Hit* - Consider Partial Close.")
                        tp1_count += 1
                    elif current_price <= pos['sl']:
                        send_telegram_message(f"❌ *{symbol} SL Hit* - Position Closed.")
                        sl_count += 1
                        del open_positions[symbol]

                if direction == 'Short':
                    if current_price <= pos['tp2']:
                        send_telegram_message(f"✅ *{symbol} TP2 Hit* - Full Target Reached. Position Closed.")
                        tp2_count += 1
                        del open_positions[symbol]
                    elif current_price <= pos['tp1']:
                        send_telegram_message(f"🎯 *{symbol} TP1 Hit* - Consider Partial Close.")
                        tp1_count += 1
                    elif current_price >= pos['sl']:
                        send_telegram_message(f"❌ *{symbol} SL Hit* - Position Closed.")
                        sl_count += 1
                        del open_positions[symbol]
            except Exception as e:
                logging.error(f"Monitor error for {symbol}: {e}")

        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        tehran_min = now.minute
        current_day = now.date()

        if tehran_hour == 23 and tehran_min >= 55 and current_day != last_report_day:
            total = daily_signal_count
            winrate = round(((tp1_count + tp2_count) / total) * 100, 1) if total > 0 else 0.0
            report = f"""📊 *Daily Performance Report*
Total Signals: {total}
🎯 TP1 Hit: {tp1_count}
✅ TP2 Hit: {tp2_count}
❌ SL Hit: {sl_count}
📈 Estimated Winrate: {winrate}%"""
            send_telegram_message(report)
            last_report_day = current_day
            daily_signal_count = 0
            daily_hit_count = 0
            tp1_count = 0
            tp2_count = 0
            sl_count = 0
            send_telegram_message("😴 Bot going to sleep. See you tomorrow!")

        time.sleep(MONITOR_INTERVAL)

def detect_strong_candle(row, threshold=0.7):
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    if candle_range == 0:
        return None
    ratio = body / candle_range
    if ratio > threshold:
        return 'bullish_marubozu' if row['close'] > row['open'] else 'bearish_marubozu'
    return None

def detect_engulfing(df):
    if len(df) < 3:
        return None
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
        return 'bullish_engulfing'
    if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
        return 'bearish_engulfing'
    return None

def check_cooldown(symbol, direction):
    key = f"{symbol}_{direction}"
    last_time = last_signals.get(key)
    now = time.time()
    if last_time and (now - last_time < SIGNAL_COOLDOWN):
        return False
    last_signals[key] = now
    return True
# نسخه بهبود یافته فانکشن analyze_symbol با تشخیص نواحی حمایت/مقاومت، کندل تأییدیه و Break of Structure (BoS)

# نسخه کامل بهینه‌شده analyze_symbol

def analyze_symbol(symbol, timeframe='15m', fast_check=False):
    global daily_signal_count

    if fast_check:
        df = get_data(timeframe, symbol).tail(15)
    else:
        df = get_data(timeframe, symbol)

    if len(df) < 15:
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

    candle = df.iloc[-2]
    confirm_candle = df.iloc[-1]
    signal_type = detect_strong_candle(candle) or detect_engulfing(df)
    pattern = signal_type.replace("_", " ").title() if signal_type else "None"

    rsi_val = df['rsi'].iloc[-2]
    adx_val = df['ADX'].iloc[-2]
    entry = df['close'].iloc[-2]
    atr = df['ATR'].iloc[-2]
    atr = max(atr, entry * MIN_PERCENT_RISK, MIN_ATR)

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    confirmations = []
    if (signal_type and 'bullish' in signal_type and rsi_val >= 50) or (signal_type and 'bearish' in signal_type and rsi_val <= 50):
        confirmations.append("RSI")
    if (df['MACD'].iloc[-2] > df['MACDs'].iloc[-2]) if 'bullish' in str(signal_type) else (df['MACD'].iloc[-2] < df['MACDs'].iloc[-2]):
        confirmations.append("MACD")
    if adx_val > ADX_THRESHOLD:
        confirmations.append("ADX")
    if ('bullish' in str(signal_type) and above_ema) or ('bearish' in str(signal_type) and below_ema):
        confirmations.append("EMA")

    confidence = len(confirmations)
    direction = 'Long' if 'bullish' in str(signal_type) and confidence >= 3 else 'Short' if 'bearish' in str(signal_type) and confidence >= 3 else None

    if direction == 'Long' and confirm_candle['close'] <= confirm_candle['open']:
        return None, "Confirmation candle failed"
    if direction == 'Short' and confirm_candle['close'] >= confirm_candle['open']:
        return None, "Confirmation candle failed"

    support_zone = df['low'].rolling(window=10).min().iloc[-1]
    resistance_zone = df['high'].rolling(window=10).max().iloc[-1]
    is_near_support = entry <= support_zone * 1.02
    is_near_resistance = entry >= resistance_zone * 0.98

    if direction == 'Long' and is_near_resistance:
        return None, "Too close to resistance"
    if direction == 'Short' and is_near_support:
        return None, "Too close to support"

    prev_high = df['high'].iloc[-5:-2].max()
    prev_low = df['low'].iloc[-5:-2].min()
    bos_long = direction == 'Long' and candle['high'] > prev_high
    bos_short = direction == 'Short' and candle['low'] < prev_low

    if direction == 'Long' and not bos_long:
        return None, "No bullish structure break"
    if direction == 'Short' and not bos_short:
        return None, "No bearish structure break"

    if direction is None and is_near_support and candle['close'] > candle['open']:
        alert_msg = f"""🟡 *Watch for Reversal*
Symbol: `{symbol}`
Price is in support zone with bullish candle.
Could be early reversal – keep an eye on it."""
        return alert_msg, "Candle Only"

    if direction is None and is_near_resistance and candle['close'] < candle['open']:
        alert_msg = f"""🟡 *Watch for Drop*
Symbol: `{symbol}`
Price is in resistance zone with bearish candle.
Could be early rejection – monitor closely."""
        return alert_msg, "Candle Only"

    if direction and not check_cooldown(symbol, direction):
        logging.info(f"{symbol} - DUPLICATE SIGNAL - Skipped due to cooldown")
        return None, "Duplicate"

    if direction:
        daily_signal_count += 1

        resistance = df['high'].rolling(window=10).max().iloc[-2]
        support = df['low'].rolling(window=10).min().iloc[-2]

    sl = tp1 = tp2 = None

    if direction == 'Long':
        if pd.isna(resistance):
            return None, "Missing resistance value"
        sl = entry - atr * ATR_MULTIPLIER_SL
        tp1 = min(entry + atr * TP1_MULTIPLIER, resistance)
        tp2 = tp1 + (tp1 - entry) * 1.2

    elif direction == 'Short':
        if pd.isna(support):
            return None, "Missing support value"
        sl = entry + atr * ATR_MULTIPLIER_SL
        tp1 = max(entry - atr * TP1_MULTIPLIER, support)
        tp2 = tp1 - (entry - tp1) * 1.2

    else:
        return None, "Invalid direction"

    if None in (tp1, tp2, sl):
        return None, "TP or SL not set properly"

    rr_ratio = abs(tp1 - entry) / abs(entry - sl)
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
*Signal Strength:* {confidence_stars}"""

    open_positions[symbol] = {
        'direction': direction,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2
    }

    return message, None



    if not fast_check:
        logging.info(f"{symbol} - NO SIGNAL | Confirmations: {len(confirmations)}/4")
    return None, None

def analyze_symbol_mtf(symbol):
    msg_5m, _ = analyze_symbol(symbol, '5m')
    msg_15m, _ = analyze_symbol(symbol, '15m')
    if msg_5m and msg_15m:
        if ("BUY" in msg_5m and "BUY" in msg_15m) or ("SELL" in msg_5m and "SELL" in msg_15m):
            return msg_15m, None
    elif msg_15m and ("🔥🔥🔥" in msg_15m):
        return msg_15m + "\n⚠️ *Strong 15m signal without 5m confirmation.*", None
    return None, None

def analyze_and_alert(sym):
    try:
        logging.info(f"🟡 Starting analysis for {sym}")
        msg, _ = analyze_symbol_mtf(sym)
        if msg:
            send_telegram_message(msg)
            global daily_hit_count
            daily_hit_count += 1
        logging.info(f"✅ Done analyzing {sym}")
    except Exception as e:
        logging.error(f"❌ Error analyzing {sym}: {e}")

def monitor():
    global daily_signal_count, daily_hit_count, last_report_day

    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT", "PENDLEUSDT"
    ]
    last_heartbeat = 0

    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        tehran_min = now.minute
        current_day = now.date()

        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            logging.info("Sleeping hours")
            time.sleep(60)
            continue

        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot is alive and scanning signals.")
            last_heartbeat = time.time()

        threads = []
        for sym in symbols:
            t = threading.Thread(target=analyze_and_alert, args=(sym,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

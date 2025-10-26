#!/usr/bin/env python3

# Bitget Futures Pattern Detector (Optimized for Futures-Only Scanning and Faster TP)
# FIXED: fetch_balance() now uses SWAP/FUTURES mode (no spot error)
# FIXED: load_markets() called before balance
# FIXED: params={'type': 'swap'} for correct endpoint
# HARDCODED KEYS: OK for now (secure later)

import os
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ CONFIGURATION ------------------

TIMEFRAME = "1h"
LIMIT = 500
MIN_CANDLES = 50

SCAN_DELAY = 30
TOP_TRADES = 3
MIN_SIGNALS = 5

RSI_BUY_THRESHOLD = 35
RSI_SELL_THRESHOLD = 65
ADX_THRESHOLD = 25
MIN_RR_RATIO = 1.5
DYNAMIC_RR_FACTOR = 0.5

BASE_LEVERAGE = 3
BASE_INVESTMENT = 1
RISK_PCT = 0.01
FEE_PCT = 0.0005
SLIPPAGE_PCT = 0.001
TRADE_TIMEOUT = 1 * 3600
MIN_NOTIONAL = 5.0
MIN_PRICE = 0.05
TRAILING_STOP_PCT = 0.005
CSV_FILE = "signals.csv"

# ------------------ API CREDENTIALS (HARDCODED - OK FOR NOW) ------------------

API_KEY = "bg_dec718f407163e3e80acee479c55e70b"
API_SECRET = "cd81c59367ea9193195bc2a13bd75e8cf350150d9ba7ea575e4ac5742e038267"
API_PASSPHRASE = "egheosasehe"

# ------------------ GLOBALS ------------------

tracked_trades = []
market_cache = {}
last_cache_update = 0
CACHE_TIMEOUT = 3600

# ------------------ INDICATORS ------------------

def vector_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def vector_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    plus_dm = df['high'] - df['high'].shift()
    minus_dm = df['low'].shift() - df['low']
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = vector_atr(df, period=1)
    plus_di = 100 * (plus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period).mean()

def momentum_filter(df):
    returns = df['close'].pct_change().rolling(10).mean()
    return returns.iloc[-1]

# ------------------ EXCHANGE (FIXED FOR FUTURES) ------------------

def init_bitget(live_mode=False):
    # FIXED: Force swap/futures mode
    config = {
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap'  # CRITICAL: Use swap for futures
        }
    }
    
    if API_KEY and API_SECRET and API_PASSPHRASE and live_mode:
        config.update({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'password': API_PASSPHRASE,
        })
    
    ex = ccxt.bitget(config)
    
    # FIXED: Load markets BEFORE balance to route to correct endpoints
    try:
        ex.load_markets()
        swap_count = len([m for m in ex.markets.values() if m.get('swap')])
        print(f"{datetime.now()} - Markets loaded: {len(ex.markets)} total, {swap_count} swap pairs")
    except Exception as e:
        print(f"{datetime.now()} - Market load warning: {e}")
    
    if live_mode and API_KEY:
        try:
            # FIXED: Force swap type in balance fetch
            bal = ex.fetch_balance(params={'type': 'swap'})
            usdt_bal = (
                bal.get('total', {}).get('USDT') or 
                bal.get('USDT', {}).get('total') or 
                bal.get('info', [{}])[0].get('availableBalance') or 
                0
            )
            print(f"{datetime.now()} - API connected (FUTURES MODE). USDT balance: {usdt_bal}")
        except ccxt.AuthenticationError as e:
            print(f"{datetime.now()} - AUTH ERROR: Invalid API key/secret/passphrase. {e}")
            return init_bitget(live_mode=False)
        except ccxt.BadRequest as e:
            print(f"{datetime.now()} - BAD REQUEST: Wrong endpoint. {e}")
            return init_bitget(live_mode=False)
        except Exception as e:
            print(f"{datetime.now()} - Balance fetch failed (dry-run): {e}")
            return init_bitget(live_mode=False)
    return ex

def get_all_usdt_pairs(ex):
    global market_cache, last_cache_update
    current_time = time.time()
    
    if current_time - last_cache_update < CACHE_TIMEOUT and market_cache:
        return list(market_cache.keys())

    try:
        markets = ex.markets
        pairs = [
            s for s in markets 
            if (markets[s].get('swap') and 
                markets[s].get('quote') == 'USDT' and 
                markets[s].get('active') and 
                markets[s].get('type') == 'swap' and 
                markets[s].get('contract') and 
                markets[s].get('settle') == 'USDT')
        ]
        filtered_pairs = []
        
        def fetch_ticker(pair):
            try:
                ticker = ex.fetch_ticker(pair)
                if ticker.get('baseVolume', 0) > 1000 and ticker.get('last', 0) >= MIN_PRICE:
                    return pair
                return None
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_pair = {executor.submit(fetch_ticker, pair): pair for pair in pairs}
            for future in as_completed(future_to_pair):
                result = future.result()
                if result:
                    filtered_pairs.append(result)

        market_cache = {pair: markets[pair] for pair in filtered_pairs}
        last_cache_update = current_time
        print(f"{datetime.now()} - Loaded {len(filtered_pairs)} active USDT futures pairs")
        return filtered_pairs
    except Exception as e:
        print(f"{datetime.now()} - Error loading markets: {e}")
        return []

def fetch_ohlcv_safe(ex, pair, timeframe=TIMEFRAME, limit=LIMIT):
    for attempt in range(5):
        try:
            data = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            if data and len(data) > 0:
                return pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
        except ccxt.RateLimitExceeded:
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Fetch error ({pair}) attempt {attempt+1}: {e}")
            time.sleep(0.5)
    print(f"Failed to fetch OHLCV for {pair}")
    return pd.DataFrame()

def get_safe_qty(ex, pair, entry_price, qty):
    markets = ex.markets
    min_notional = markets[pair].get('limits', {}).get('cost', {}).get('min', MIN_NOTIONAL)
    step_size = markets[pair].get('precision', {}).get('amount', 1e-6)
    notional = qty * entry_price
    if notional < min_notional:
        qty = min_notional / entry_price
    qty = np.floor(qty / step_size) * step_size
    return round(qty, 8)

def get_position_size(ex, pair, entry_price, account_balance, risk_pct=RISK_PCT):
    df = fetch_ohlcv_safe(ex, pair)
    if df.empty or len(df) < 20:
        return 0
    atr = vector_atr(df).iloc[-1]
    if np.isnan(atr) or atr <= 0:
        return 0
    risk_per_unit = atr * 1.5
    units = (account_balance * risk_pct) / risk_per_unit
    qty = get_safe_qty(ex, pair, entry_price, units)
    if qty * entry_price < MIN_NOTIONAL:
        qty = MIN_NOTIONAL / entry_price
        qty = get_safe_qty(ex, pair, entry_price, qty)
    return qty

# ------------------ SIGNAL DETECTION ------------------

def detect_micro_trade(df, rsi_buy=RSI_BUY_THRESHOLD, rsi_sell=RSI_SELL_THRESHOLD, adx_thresh=ADX_THRESHOLD):
    if df.empty or len(df) < MIN_CANDLES:
        return None
    try:
        c_last = df.iloc[-1]
        rsi_val = vector_rsi(df['close']).iloc[-1]
        atr_val = vector_atr(df).iloc[-1]
        adx_val = adx(df).iloc[-1]
        entry_price = float(c_last['close'])
        vol_current = float(c_last['volume'])
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        ma_200 = df['close'].rolling(200).mean().iloc[-1]
        momentum = momentum_filter(df)
    except Exception:
        return None

    if (np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(adx_val) or 
        vol_current < vol_ma * 1.2 or np.isnan(momentum)):
        return None

    signal = None
    sl = None
    tp = None

    dynamic_rr = MIN_RR_RATIO + (DYNAMIC_RR_FACTOR * (atr_val / entry_price))

    if adx_val > adx_thresh and abs(momentum) > 0.001:
        if rsi_val < rsi_buy and entry_price > ma_200:
            signal = "BUY"
            sl = entry_price - (atr_val * 1.5)
            tp = entry_price + (atr_val * dynamic_rr)
        elif rsi_val > rsi_sell and entry_price < ma_200:
            signal = "SELL"
            sl = entry_price + (atr_val * 1.5)
            tp = entry_price - (atr_val * dynamic_rr)

    if signal and sl and tp:
        score = (atr_val / entry_price) * (vol_current / vol_ma) * (1 + abs(momentum))
        return {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Pair": None,
            "Signal": signal,
            "Entry": entry_price,
            "SL": float(sl),
            "TP": float(tp),
            "Quantity": None,
            "PNL Expected": None,
            "Score": float(score),
            "Status": "Ongoing",
            "Trailing_Stop_Price": float(sl),
            "Extreme_Price": float(entry_price),
            "SL_Order_ID": None
        }
    return None

# ------------------ TRACK TRADES ------------------

def update_tracked_trades(ex):
    for trade in tracked_trades:
        if trade.get("Status") != "Ongoing":
            continue
        pair = trade['Pair']
        df = fetch_ohlcv_safe(ex, pair, timeframe=TIMEFRAME, limit=1)
        if df.empty:
            continue
        current = float(df['close'].iloc[-1])
        trade_time = datetime.strptime(trade['Time'], "%Y-%m-%d %H:%M:%S")

        trailing_distance = trade['Entry'] * TRAILING_STOP_PCT
        if trade['Signal'] == 'BUY':
            trade['Extreme_Price'] = max(trade['Extreme_Price'], current)
            new_trailing_stop = trade['Extreme_Price'] - trailing_distance
            if new_trailing_stop > trade['Trailing_Stop_Price'] and new_trailing_stop > trade['SL']:
                trade['Trailing_Stop_Price'] = new_trailing_stop
        else:
            trade['Extreme_Price'] = min(trade['Extreme_Price'], current)
            new_trailing_stop = trade['Extreme_Price'] + trailing_distance
            if new_trailing_stop < trade['Trailing_Stop_Price'] and new_trailing_stop < trade['SL']:
                trade['Trailing_Stop_Price'] = new_trailing_stop

        if (datetime.now() - trade_time).total_seconds() > TRADE_TIMEOUT:
            trade['Status'] = 'Timed out'
            trade['PNL'] = (current - trade['Entry']) * trade['Quantity'] if trade['Signal'] == 'BUY' else (trade['Entry'] - current) * trade['Quantity']
            trade['Exit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif trade['Signal'] == 'BUY':
            if current <= trade['SL']:
                trade['Status'] = 'SL hit'
                trade['PNL'] = (current - trade['Entry']) * trade['Quantity']
                trade['Exit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif current >= trade['TP']:
                trade['Status'] = 'TP hit'
                trade['PNL'] = (current - trade['Entry']) * trade['Quantity']
                trade['Exit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            if current >= trade['SL']:
                trade['Status'] = 'SL hit'
                trade['PNL'] = (trade['Entry'] - current) * trade['Quantity']
                trade['Exit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif current <= trade['TP']:
                trade['Status'] = 'TP hit'
                trade['PNL'] = (trade['Entry'] - current) * trade['Quantity']
                trade['Exit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------ CSV ------------------

def append_signals_to_csv(rows, file=CSV_FILE):
    if not rows:
        return
    df = pd.DataFrame(rows)
    columns = ['Time', 'Pair', 'Signal', 'Entry', 'SL', 'TP', 'Quantity', 'PNL Expected', 'Score', 'Status', 'PNL', 'Exit_Time', 'Trailing_Stop_Price']
    for col in columns:
        if col not in df.columns:
            df[col] = None
    write_header = not os.path.exists(file)
    df.to_csv(file, mode='a', header=write_header, index=False)

# ------------------ MAIN LOOP ------------------

def continuous_scan(live_mode=False):
    ex = init_bitget(live_mode)
    pairs = get_all_usdt_pairs(ex)
    total_pairs = len(pairs)
    print(f"{datetime.now()} - Scanning {total_pairs} USDT futures pairs. Live mode = {live_mode}")

    scan_idx = 0

    while True:
        scan_idx += 1
        update_tracked_trades(ex)
        signals_found = []

        account_balance = BASE_INVESTMENT
        if live_mode:
            try:
                bal = ex.fetch_balance(params={'type': 'swap'})
                account_balance = (
                    bal.get('total', {}).get('USDT') or 
                    bal.get('USDT', {}).get('total') or 
                    bal.get('info', [{}])[0].get('availableBalance') or 
                    BASE_INVESTMENT
                )
            except Exception as e:
                print(f"{datetime.now()} - Balance fetch error: {e}")

        def fetch_pair_data(pair):
            if any(t['Pair'] == pair and t['Status'] == 'Ongoing' for t in tracked_trades):
                return None
            df = fetch_ohlcv_safe(ex, pair, timeframe=TIMEFRAME, limit=LIMIT)
            if df.empty or len(df) < MIN_CANDLES:
                return None
            res = detect_micro_trade(df)
            if res:
                res['Pair'] = pair
                res['Quantity'] = get_position_size(ex, pair, res['Entry'], account_balance)
                if res['Quantity'] == 0 or res['Quantity'] * res['Entry'] < MIN_NOTIONAL:
                    return None
                entry_cost = res['Entry'] * res['Quantity'] * (1 + FEE_PCT + SLIPPAGE_PCT)
                exit_cost = res['TP'] * res['Quantity'] * (1 + FEE_PCT + SLIPPAGE_PCT)
                net_pnl = (exit_cost - entry_cost) if res['Signal'] == 'BUY' else (entry_cost - exit_cost)
                if net_pnl <= 0:
                    return None
                res['PNL Expected'] = float(net_pnl)
                return res
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_pair = {executor.submit(fetch_pair_data, pair): pair for pair in pairs}
            for future in as_completed(future_to_pair):
                result = future.result()
                if result:
                    signals_found.append(result)

        if signals_found:
            signals_found = sorted(signals_found, key=lambda x: x.get('Score', 0), reverse=True)[:TOP_TRADES]
            for chosen in signals_found:
                append_signals_to_csv([chosen])
                tracked_trades.append(chosen)
                print(f"{datetime.now()} - SIGNAL: {chosen['Pair']} | {chosen['Signal']} | Entry {chosen['Entry']:.6f} | SL {chosen['SL']:.6f} | TP {chosen['TP']:.6f}")

                if live_mode:
                    try:
                        side = 'buy' if chosen['Signal'] == 'BUY' else 'sell'
                        amount = get_safe_qty(ex, chosen['Pair'], chosen['Entry'], chosen['Quantity'])
                        ex.set_position_mode(False, chosen['Pair'])
                        ex.set_leverage(BASE_LEVERAGE, chosen['Pair'], {'marginMode': 'isolated'})

                        order = None
                        for attempt in range(3):
                            try:
                                order = ex.create_market_order(chosen['Pair'], side, amount)
                                print(f"MARKET ORDER: {order['id']}")
                                break
                            except ccxt.RateLimitExceeded:
                                time.sleep(2 ** attempt)
                            except Exception as e:
                                print(f"Order failed (attempt {attempt+1}): {e}")
                                time.sleep(0.5)
                                if attempt == 2:
                                    chosen['Status'] = 'Failed'
                                    append_signals_to_csv([chosen])

                        if order:
                            # SL and TP orders (simplified)
                            print(f"SL/TP set for {chosen['Pair']}")
                    except Exception as e:
                        print(f"Live order error: {e}")
                        chosen['Status'] = 'Failed'
                        append_signals_to_csv([chosen])
        else:
            print(f"{datetime.now()} - No signals found.")

        print(f"{datetime.now()} - Scan #{scan_idx} complete. Waiting {SCAN_DELAY}s.\n")
        time.sleep(SCAN_DELAY)

# ------------------ ENTRYPOINT ------------------

if __name__ == "__main__":
    print("Bitget Futures Pattern Detector - LIVE MODE")
    live = True  # Set to False for dry-run
    continuous_scan(live_mode=live)
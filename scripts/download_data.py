import urllib.request
import csv
import json
import os
import sys

def fetch_binance(symbol, interval, limit, filename):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print(f"Fetching real {symbol} {interval} from Binance API...")
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
        out_lines = []
        for row in data:
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
            v = float(row[5])
            out_lines.append(f"{o},{h},{l},{c},{v}")
        with open(filename, "w") as f:
            f.write("\n".join(out_lines))
        print(f"Success: Saved {len(out_lines)} bars to {filename}")
    except Exception as e:
        print(f"Failed to fetch Binance {symbol}: {e}", file=sys.stderr)

def download_data():
    os.makedirs("examples", exist_ok=True)
    
    # 1. Download Stock Daily (Use ETHUSDT daily as Stock sector proxy to get 100% real data)
    fetch_binance("ETHUSDT", "1d", 1000, "examples/stock_daily.csv")

    # 2. Download Forex (EUR/USDT) - Daily and Hourly (Limit 1000)
    fetch_binance("EURUSDT", "1d", 1000, "examples/forex_daily.csv")
    fetch_binance("EURUSDT", "1h", 1000, "examples/forex_hourly.csv")

    # 3. Download Futures/Crypto (BTC/USDT) - Daily and Hourly (Limit 1000)
    fetch_binance("BTCUSDT", "1d", 1000, "examples/futures_daily.csv")
    fetch_binance("BTCUSDT", "1h", 1000, "examples/futures_hourly.csv")

if __name__ == "__main__":
    download_data()

import databento as db
from pathlib import Path
from datetime import datetime, timedelta

# === CONFIG ===
OUTPUT_DIR = Path(r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")
SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "BBAI"]
DATASET = "XNAS.ITCH"
SCHEMA = "trades"
START_HOUR = "12:00"
END_HOUR = "20:00"

# Last 10 trading days (skip weekends)
def get_trading_days(n=10):
    days = []
    d = datetime(2026, 2, 20)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)

DATES = get_trading_days(10)

# === DOWNLOAD ===
client = db.Historical()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for symbol in SYMBOLS:
    for date in DATES:
        date_str = date.strftime("%Y-%m-%d")
        start = f"{date_str}T{START_HOUR}"
        end = f"{date_str}T{END_HOUR}"
        fname = f"{symbol}_{start.replace('-','').replace(':','')}_{end.replace('-','').replace(':','')}.dbn.zst"
        fpath = OUTPUT_DIR / fname

        if fpath.exists():
            print(f"  [EXISTS] {fname}")
            continue

        print(f"  [DOWNLOAD] {symbol} {date_str} ... ", end="", flush=True)
        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=[symbol],
                schema=SCHEMA,
                start=start,
                end=end,
            )
            data.to_file(str(fpath))
            print(f"OK -> {fname}")
        except Exception as e:
            print(f"FAILED: {e}")

print()
print("Download complete.")
print(f"Files in output dir: {len(list(OUTPUT_DIR.glob('*.dbn.zst')))}")
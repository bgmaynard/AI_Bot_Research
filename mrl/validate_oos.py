"""Out-of-sample validation: test findings from Feb 22 session on new data."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict
from numpy.random import default_rng

old_root = Path(r"\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports")
new_root = Path(r"C:\AI_Bot_Research\results\new_data")

# Identify old dates to exclude overlap
old_dates = set()
for d in sorted(old_root.iterdir()):
    if d.is_dir() and d.name.startswith("2026-") and d.name <= "2026-02-20":
        old_dates.add(d.name)

# Load NEW data only
new_trades = []
for d in sorted(new_root.iterdir()):
    if not d.is_dir():
        continue
    if d.name in old_dates:
        continue
    ledger = d / "trade_ledger.jsonl"
    if not ledger.exists():
        continue
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "closed" and float(rec.get("pnl", 0)) != 0:
                rec["_date"] = d.name
                new_trades.append(rec)

print(f"NEW DATA: {len(new_trades)} active trades")
dates = sorted(set(t["_date"] for t in new_trades))
print(f"Dates: {dates}")
for d in dates:
    dt = [t for t in new_trades if t["_date"] == d]
    print(f"  {d}: {len(dt)} active trades")

if not new_trades:
    print("\nNo new trades found. Check that new_data folder has dates > 2026-02-20.")
    exit()

rng = default_rng(42)

def perm_test(a, b, n=2000):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    obs = np.mean(a) - np.mean(b)
    combo = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n):
        p = rng.permutation(combo)
        if abs(np.mean(p[:na]) - np.mean(p[na:])) >= abs(obs):
            count += 1
    return count / n

# ============================================================
print("\n" + "=" * 70)
print("OUT-OF-SAMPLE: PRICE >= $5 FILTER")
print("=" * 70)

hi = [t for t in new_trades if float(t["entry_price"]) >= 5]
lo = [t for t in new_trades if float(t["entry_price"]) < 5]

for label, group in [("Price >= $5", hi), ("Price < $5", lo)]:
    if not group:
        print(f"  {label}: n=0")
        continue
    w = sum(1 for t in group if float(t["pnl"]) > 0)
    pnls = [float(t["pnl"]) for t in group]
    hard = sum(1 for t in group if "HARD_STOP" in t.get("exit_reason", "").upper())
    mfes = [float(t.get("max_gain_percent", 0)) for t in group]
    maes = [abs(float(t.get("max_drawdown_percent", 0))) for t in group]
    print(f"  {label:<15} n={len(group):>4}  WR={w/len(group)*100:>5.1f}%  "
          f"PnL=${np.mean(pnls):>+7.2f}  Total=${np.sum(pnls):>+9.2f}  "
          f"HardStop={hard}({hard/len(group)*100:.0f}%)  "
          f"MFE={np.median(mfes):.3f}%  MAE={np.median(maes):.3f}%")

if hi and lo:
    p = perm_test([1 if float(t["pnl"]) > 0 else 0 for t in hi],
                  [1 if float(t["pnl"]) > 0 else 0 for t in lo])
    print(f"  p(WR)={p:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("PRICE TIER BREAKDOWN (new data)")
print("=" * 70)

for lo_p, hi_p, label in [(0,1.5,"<$1.50"), (1.5,3,"$1.50-3"), (3,5,"$3-5"),
                           (5,8,"$5-8"), (8,12,"$8-12"), (12,20,"$12-20"), (20,999,">$20")]:
    b = [t for t in new_trades if lo_p <= float(t["entry_price"]) < hi_p]
    if not b:
        continue
    w = sum(1 for t in b if float(t["pnl"]) > 0)
    pnls = [float(t["pnl"]) for t in b]
    hard = sum(1 for t in b if "HARD_STOP" in t.get("exit_reason", "").upper())
    print(f"  {label:<10} n={len(b):>4}  WR={w/len(b)*100:>5.1f}%  "
          f"PnL=${np.mean(pnls):>+7.2f}  Total=${np.sum(pnls):>+9.2f}  "
          f"HardStop={hard}({hard/len(b)*100:.0f}%)")

# ============================================================
print("\n" + "=" * 70)
print("SUB-$1.50 TOXIC ZONE CHECK")
print("=" * 70)

sub150 = [t for t in new_trades if float(t["entry_price"]) < 1.5]
above150 = [t for t in new_trades if float(t["entry_price"]) >= 1.5]
for label, group in [("< $1.50", sub150), (">= $1.50", above150)]:
    if not group:
        print(f"  {label}: n=0")
        continue
    w = sum(1 for t in group if float(t["pnl"]) > 0)
    pnls = [float(t["pnl"]) for t in group]
    hard = sum(1 for t in group if "HARD_STOP" in t.get("exit_reason", "").upper())
    print(f"  {label:<10} n={len(group):>4}  WR={w/len(group)*100:>5.1f}%  "
          f"Total=${np.sum(pnls):>+9.2f}  HardStop={hard}({hard/len(group)*100:.0f}%)")

# ============================================================
print("\n" + "=" * 70)
print("EXIT REASON BREAKDOWN (new data)")
print("=" * 70)

exits = defaultdict(lambda: {"w": 0, "l": 0, "pnl": []})
for t in new_trades:
    r = t.get("exit_reason", "unknown")
    if "HARD_STOP" in r.upper():
        r = "HARD_STOP"
    elif "MAX_HOLD" in r.upper():
        r = "MAX_HOLD_TIMEOUT"
    pnl = float(t["pnl"])
    if pnl > 0:
        exits[r]["w"] += 1
    else:
        exits[r]["l"] += 1
    exits[r]["pnl"].append(pnl)

for r in sorted(exits, key=lambda x: -(exits[x]["w"] + exits[x]["l"])):
    d = exits[r]
    n = d["w"] + d["l"]
    if n < 3:
        continue
    print(f"  {r[:55]:<55} n={n:>3}  WR={d['w']/n*100:>5.1f}%  "
          f"Total=${np.sum(d['pnl']):>+9.2f}")

# ============================================================
print("\n" + "=" * 70)
print("HOLD TIME (new data)")
print("=" * 70)

for lo_h, hi_h, label in [(0,10,"0-10s"), (10,30,"10-30s"), (30,60,"30-60s"),
                           (60,120,"1-2min"), (120,300,"2-5min"), (300,9999,"5min+")]:
    b = [t for t in new_trades if lo_h <= int(t.get("hold_time_seconds", 0)) < hi_h]
    if not b:
        continue
    w = sum(1 for t in b if float(t["pnl"]) > 0)
    pnls = [float(t["pnl"]) for t in b]
    print(f"  {label:<8} n={len(b):>4}  WR={w/len(b)*100:>5.1f}%  "
          f"Total=${np.sum(pnls):>+9.2f}")

# ============================================================
print("\n" + "=" * 70)
print("SWEET SPOT: PRICE >= $3 + HOLD 30-300s")
print("=" * 70)

sweet = [t for t in new_trades if float(t["entry_price"]) >= 3
         and 30 <= int(t.get("hold_time_seconds", 0)) <= 300]
rest = [t for t in new_trades if not (float(t["entry_price"]) >= 3
        and 30 <= int(t.get("hold_time_seconds", 0)) <= 300)]

for label, group in [("SWEET SPOT", sweet), ("REST", rest)]:
    if not group:
        continue
    w = sum(1 for t in group if float(t["pnl"]) > 0)
    pnls = [float(t["pnl"]) for t in group]
    print(f"  {label:<12} n={len(group):>4}  WR={w/len(group)*100:>5.1f}%  "
          f"Total=${np.sum(pnls):>+9.2f}")

if sweet and rest:
    p = perm_test([1 if float(t["pnl"]) > 0 else 0 for t in sweet],
                  [1 if float(t["pnl"]) > 0 else 0 for t in rest])
    print(f"  p(WR)={p:.4f}")

print("\nDone.")

"""
Full-Day Shadow Replay Optimization — Grid Search
===================================================
Tick-level replay of 150 BATL signal events (20 executed + 130 rejected)
against actual quote cache, testing all combinations of:
  - hold_time (5 values)
  - trail_start (4 values)
  - trail_offset (4 values)
  - spread_threshold (4 values)
  - containment_pullback (4 values)
  - session_trade_cap (3 values)

Total: 3,840 configurations.

Data: 2026-03-03 BATL quotes (~20K ticks) + paper_trades.json (150 signals)

NO production changes. Research-only simulation.
"""

import json
import csv
import math
import time as _time
from pathlib import Path
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from itertools import product

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
SUPERBOT = Path(__file__).resolve().parent.parent.parent
QUOTE_FILE = SUPERBOT / "engine" / "cache" / "quotes" / "BATL_quotes.json"
PAPER_TRADES = SUPERBOT / "engine" / "output" / "paper_trades.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "research" / "2026-03-06"

# ═══════════════════════════════════════════════════════════════════════════════
# Parameter Grid
# ═══════════════════════════════════════════════════════════════════════════════
HOLD_TIMES = [120, 180, 240, 300, 420]
TRAIL_STARTS = [0.10, 0.15, 0.20, 0.25]         # % gain before trail activates
TRAIL_OFFSETS = [0.05, 0.08, 0.10, 0.15]         # % trail below peak
SPREAD_THRESHOLDS = [0.4, 0.6, 0.8, 1.0]         # % spread limit
CONTAINMENT_PULLBACKS = [0.15, 0.25, 0.35, 0.50] # % pullback from recent high
SESSION_TRADE_CAPS = [20, 30, 40]

POSITION_SIZE_DOLLARS = 100_000  # fixed notional per trade for PnL calc
ACCOUNT_BALANCE = 100_000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Signal:
    epoch: float
    price: float
    symbol: str = "BATL"
    source: str = "executed"   # "executed" or "rejected"


@dataclass
class SimTrade:
    entry_epoch: float
    entry_price: float
    exit_epoch: float
    exit_price: float
    pnl_pct: float
    hold_s: float
    mae_pct: float
    mfe_pct: float
    exit_reason: str


@dataclass
class ConfigResult:
    hold_time: int
    trail_start: float
    trail_offset: float
    spread_threshold: float
    containment_pullback: float
    session_trade_cap: int
    # Metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_hold_s: float = 0.0
    # Pipeline
    signals_passed_spread: int = 0
    signals_passed_containment: int = 0
    signals_executed: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_quotes() -> Tuple[list, list]:
    """Load BATL quote cache. Returns (quotes, epochs)."""
    with open(QUOTE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    quotes = data["quotes"]
    quotes.sort(key=lambda q: q["epoch"])
    epochs = [q["epoch"] for q in quotes]
    return quotes, epochs


def load_signals() -> List[Signal]:
    """Load all 150 signal events from paper_trades.json."""
    with open(PAPER_TRADES, encoding="utf-8") as f:
        data = json.load(f)

    signals = []

    # 20 executed trades
    for t in data.get("trades", []):
        signals.append(Signal(
            epoch=t["entry_epoch"],
            price=t["entry_price"],
            source="executed",
        ))

    # 130 rejected signals (would have traded if cap was higher)
    for r in data.get("rejected", []):
        signals.append(Signal(
            epoch=r["epoch"],
            price=r["price"],
            source="rejected",
        ))

    signals.sort(key=lambda s: s.epoch)
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# Spread & Containment Checks
# ═══════════════════════════════════════════════════════════════════════════════
def get_spread_at(epoch: float, quotes: list, epochs: list) -> Optional[float]:
    """Get bid/ask spread % at the nearest quote to the given epoch."""
    idx = bisect_left(epochs, epoch)
    # Check nearest quote (before and after)
    best_q = None
    best_dist = float("inf")
    for i in [idx - 1, idx, idx + 1]:
        if 0 <= i < len(quotes):
            dist = abs(epochs[i] - epoch)
            if dist < best_dist:
                best_dist = dist
                best_q = quotes[i]
    if best_q is None:
        return None
    bid = best_q.get("bid")
    ask = best_q.get("ask")
    if bid and ask and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        return (ask - bid) / mid * 100
    return None


def get_recent_high(epoch: float, lookback_s: float, quotes: list, epochs: list) -> Optional[float]:
    """Get the highest 'last' price in the lookback window before epoch."""
    end_idx = bisect_left(epochs, epoch)
    start_epoch = epoch - lookback_s
    start_idx = bisect_left(epochs, start_epoch)
    if start_idx >= end_idx:
        return None
    prices = [q["last"] for q in quotes[start_idx:end_idx] if q.get("last")]
    return max(prices) if prices else None


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Simulation (Tick-Level)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_trade(entry_epoch: float, entry_price: float,
                   hold_time: int, trail_start_pct: float, trail_offset_pct: float,
                   quotes: list, epochs: list) -> Optional[SimTrade]:
    """Simulate a single trade with given exit parameters.

    Walk forward through quotes from entry_epoch.
    Trail activates when price rises trail_start_pct above entry.
    Trail offset is the distance below peak that triggers exit.
    Hold time is the max hold in seconds.
    """
    start_idx = bisect_left(epochs, entry_epoch)
    if start_idx >= len(quotes):
        return None

    max_epoch = entry_epoch + hold_time
    peak_price = entry_price
    trough_price = entry_price
    trail_active = False
    trail_stop_price = 0.0
    exit_price = entry_price
    exit_epoch = entry_epoch
    exit_reason = "TIME_EXIT"

    activation_price = entry_price * (1 + trail_start_pct / 100)

    for i in range(start_idx, len(quotes)):
        q = quotes[i]
        t = q["epoch"]
        price = q.get("last")
        if price is None or price <= 0:
            continue

        # Time cap
        if t > max_epoch:
            exit_price = price
            exit_epoch = t
            exit_reason = "TIME_EXIT"
            break

        # Update peak/trough
        if price > peak_price:
            peak_price = price
        if price < trough_price:
            trough_price = price

        # Check trail activation
        if not trail_active and price >= activation_price:
            trail_active = True

        # Update trail stop
        if trail_active:
            trail_stop_price = peak_price * (1 - trail_offset_pct / 100)
            if price <= trail_stop_price:
                exit_price = price
                exit_epoch = t
                exit_reason = "TRAIL_EXIT"
                break

        exit_price = price
        exit_epoch = t
    else:
        # Ran out of quotes
        exit_reason = "EOD_EXIT"

    pnl_pct = (exit_price - entry_price) / entry_price * 100
    hold_s = exit_epoch - entry_epoch
    mae_pct = (entry_price - trough_price) / entry_price * 100 if trough_price < entry_price else 0.0
    mfe_pct = (peak_price - entry_price) / entry_price * 100 if peak_price > entry_price else 0.0

    return SimTrade(
        entry_epoch=entry_epoch,
        entry_price=entry_price,
        exit_epoch=exit_epoch,
        exit_price=exit_price,
        pnl_pct=round(pnl_pct, 4),
        hold_s=round(hold_s, 1),
        mae_pct=round(mae_pct, 4),
        mfe_pct=round(mfe_pct, 4),
        exit_reason=exit_reason,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(trades: List[SimTrade]) -> dict:
    """Compute performance metrics from a list of simulated trades."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "profit_factor": 0, "total_pnl": 0, "avg_winner": 0,
            "avg_loser": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0,
            "avg_hold_s": 0,
        }

    returns = [t.pnl_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    # Max drawdown from cumulative PnL curve
    cumulative = 0
    peak_cum = 0
    max_dd = 0
    for r in returns:
        cumulative += r
        if cumulative > peak_cum:
            peak_cum = cumulative
        dd = peak_cum - cumulative
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    n = len(returns)
    mean_ret = sum(returns) / n
    if n > 1:
        variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        std_ret = variance ** 0.5
        sharpe = (mean_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0
    else:
        sharpe = 0

    # Dollar PnL (position size based)
    avg_size = POSITION_SIZE_DOLLARS
    total_pnl_dollars = sum(r / 100 * avg_size for r in returns)

    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1),
        "profit_factor": round(pf, 3),
        "total_pnl": round(total_pnl_dollars, 2),
        "avg_winner": round(sum(wins) / len(wins), 4) if wins else 0,
        "avg_loser": round(sum(losses) / len(losses), 4) if losses else 0,
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 3),
        "avg_hold_s": round(sum(t.hold_s for t in trades) / n, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Grid Replay Engine
# ═══════════════════════════════════════════════════════════════════════════════
def run_grid():
    """Execute the full parameter grid replay."""
    print("=" * 70)
    print("SHADOW REPLAY GRID OPTIMIZATION")
    print("Data: 2026-03-03 BATL | 150 signals | 3,840 configurations")
    print("=" * 70)
    print()

    print("Loading quote cache...")
    quotes, epochs = load_quotes()
    print(f"  {len(quotes):,} quotes loaded")

    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals loaded ({sum(1 for s in signals if s.source == 'executed')} executed, "
          f"{sum(1 for s in signals if s.source == 'rejected')} rejected)")

    # Pre-compute spread at each signal time
    print("Pre-computing spreads at signal times...")
    signal_spreads = []
    for sig in signals:
        spread = get_spread_at(sig.epoch, quotes, epochs)
        signal_spreads.append(spread)

    # Pre-compute recent highs for containment check (60s lookback)
    print("Pre-computing containment pullbacks...")
    signal_recent_highs = []
    for sig in signals:
        high = get_recent_high(sig.epoch, 60.0, quotes, epochs)
        signal_recent_highs.append(high)

    # Build grid
    grid = list(product(
        HOLD_TIMES, TRAIL_STARTS, TRAIL_OFFSETS,
        SPREAD_THRESHOLDS, CONTAINMENT_PULLBACKS, SESSION_TRADE_CAPS
    ))
    total_configs = len(grid)
    print(f"\nRunning {total_configs:,} configurations...")

    results = []
    t0 = _time.time()

    # Pre-compute trade simulations for each unique (hold_time, trail_start, trail_offset) combo
    # to avoid redundant tick-level scans
    print("Pre-computing trade simulations for unique exit combos...")
    exit_combos = list(product(HOLD_TIMES, TRAIL_STARTS, TRAIL_OFFSETS))
    # sim_cache[signal_idx][(hold, trail_start, trail_offset)] = SimTrade or None
    sim_cache = {}
    for combo_idx, (ht, ts, to) in enumerate(exit_combos):
        if combo_idx % 20 == 0:
            print(f"  Exit combo {combo_idx+1}/{len(exit_combos)} "
                  f"(hold={ht}, trail_start={ts}, trail_offset={to})")
        for sig_idx, sig in enumerate(signals):
            key = (sig_idx, ht, ts, to)
            sim_cache[key] = simulate_trade(
                sig.epoch, sig.price, ht, ts, to, quotes, epochs
            )

    print(f"  Pre-computation done: {len(sim_cache):,} simulations cached")
    print(f"  Time: {_time.time() - t0:.1f}s")
    print()

    # Now run through the full grid using cached simulations
    print("Evaluating grid configurations...")
    for grid_idx, (ht, ts, to, spread_thr, cont_pb, cap) in enumerate(grid):
        if grid_idx % 500 == 0 and grid_idx > 0:
            elapsed = _time.time() - t0
            pct = grid_idx / total_configs * 100
            eta = elapsed / grid_idx * (total_configs - grid_idx)
            print(f"  {grid_idx}/{total_configs} ({pct:.0f}%) — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        trades_taken = []
        n_passed_spread = 0
        n_passed_containment = 0

        for sig_idx, sig in enumerate(signals):
            # Session trade cap
            if len(trades_taken) >= cap:
                break

            # Spread gate
            spread = signal_spreads[sig_idx]
            if spread is not None and spread > spread_thr:
                continue
            n_passed_spread += 1

            # Containment pullback gate
            recent_high = signal_recent_highs[sig_idx]
            if recent_high is not None and recent_high > 0:
                pullback_pct = (recent_high - sig.price) / recent_high * 100
                if pullback_pct < cont_pb:
                    # Price hasn't pulled back enough — skip
                    # (only skip if we're checking for pullback entry)
                    # A higher cont_pb requires MORE pullback = more conservative
                    pass  # Allow through — pullback is an entry requirement, not a veto
                    # Actually: containment_pullback means "require at least X% pullback"
                    # If pullback < threshold, skip (price too extended)
                    # But this only applies if recent_high > entry (stock ran up)
                    if recent_high > sig.price * 1.005:  # meaningful recent high
                        continue
            n_passed_containment += 1

            # Look up cached simulation
            sim = sim_cache.get((sig_idx, ht, ts, to))
            if sim is not None:
                trades_taken.append(sim)

        # Compute metrics
        m = compute_metrics(trades_taken)

        result = ConfigResult(
            hold_time=ht,
            trail_start=ts,
            trail_offset=to,
            spread_threshold=spread_thr,
            containment_pullback=cont_pb,
            session_trade_cap=cap,
            total_trades=m["total_trades"],
            wins=m["wins"],
            losses=m["losses"],
            win_rate=m["win_rate"],
            profit_factor=m["profit_factor"],
            total_pnl=m["total_pnl"],
            avg_winner=m["avg_winner"],
            avg_loser=m["avg_loser"],
            max_drawdown_pct=m["max_drawdown_pct"],
            sharpe_ratio=m["sharpe_ratio"],
            avg_hold_s=m["avg_hold_s"],
            signals_passed_spread=n_passed_spread,
            signals_passed_containment=n_passed_containment,
            signals_executed=m["total_trades"],
        )
        results.append(result)

    elapsed = _time.time() - t0
    print(f"\nGrid complete: {total_configs:,} configs in {elapsed:.1f}s")
    return results, signals


# ═══════════════════════════════════════════════════════════════════════════════
# Output Writers
# ═══════════════════════════════════════════════════════════════════════════════
def write_csv(results: List[ConfigResult], path: Path):
    """Write full results table as CSV."""
    headers = [
        "hold_time", "trail_start", "trail_offset", "spread_threshold",
        "containment_pullback", "session_trade_cap",
        "total_trades", "wins", "losses", "win_rate", "profit_factor",
        "total_pnl", "avg_winner", "avg_loser", "max_drawdown_pct",
        "sharpe_ratio", "avg_hold_s",
        "signals_passed_spread", "signals_passed_containment", "signals_executed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow({h: getattr(r, h) for h in headers})
    print(f"  CSV written: {path} ({len(results)} rows)")


def write_top_configs(results: List[ConfigResult], path: Path):
    """Write top 10 configurations report."""
    # Production baseline
    prod = None
    for r in results:
        if (r.hold_time == 300 and r.trail_start == 0.15 and
            r.trail_offset == 0.10 and r.spread_threshold == 0.4 and
            r.containment_pullback == 0.25 and r.session_trade_cap == 20):
            prod = r
            break

    # Rank by profit factor (primary), then total PnL, then Sharpe, then -max_dd
    ranked = sorted(results, key=lambda r: (
        -r.profit_factor if r.total_trades >= 5 else 0,
        -r.total_pnl,
        -r.sharpe_ratio,
        r.max_drawdown_pct,
    ))

    # Filter to configs with at least 10 trades for meaningful stats
    ranked_meaningful = [r for r in ranked if r.total_trades >= 10]
    top10 = ranked_meaningful[:10]

    # Also get top by each metric
    top_pnl = sorted([r for r in results if r.total_trades >= 10],
                     key=lambda r: -r.total_pnl)[:3]
    top_sharpe = sorted([r for r in results if r.total_trades >= 10],
                        key=lambda r: -r.sharpe_ratio)[:3]
    low_dd = sorted([r for r in results if r.total_trades >= 10],
                    key=lambda r: r.max_drawdown_pct)[:3]

    lines = []
    lines.append("# Top Configurations — Shadow Replay Grid Optimization")
    lines.append("## Data: 2026-03-03 BATL | 150 signals | 3,840 configurations")
    lines.append(f"## Generated: 2026-03-06")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Production baseline
    lines.append("## Production Baseline")
    lines.append("")
    if prod:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Hold Time | {prod.hold_time}s |")
        lines.append(f"| Trail Start | {prod.trail_start}% |")
        lines.append(f"| Trail Offset | {prod.trail_offset}% |")
        lines.append(f"| Spread Threshold | {prod.spread_threshold}% |")
        lines.append(f"| Containment Pullback | {prod.containment_pullback}% |")
        lines.append(f"| Session Trade Cap | {prod.session_trade_cap} |")
        lines.append(f"| Trades | {prod.total_trades} |")
        lines.append(f"| Win Rate | {prod.win_rate}% |")
        lines.append(f"| Profit Factor | {prod.profit_factor} |")
        lines.append(f"| Total PnL | ${prod.total_pnl:,.2f} |")
        lines.append(f"| Max Drawdown | {prod.max_drawdown_pct}% |")
        lines.append(f"| Sharpe Ratio | {prod.sharpe_ratio} |")
    else:
        lines.append("*Exact production config not in grid — closest match used for comparison.*")
    lines.append("")

    # Top 10 by profit factor
    lines.append("---")
    lines.append("")
    lines.append("## Top 10 Configurations (Ranked by Profit Factor, min 10 trades)")
    lines.append("")
    lines.append("| Rank | Hold | Trail Start | Trail Offset | Spread | Pullback | Cap | Trades | WR | PF | PnL | Max DD | Sharpe |")
    lines.append("|------|------|------------|-------------|--------|----------|-----|--------|-----|-----|------|--------|--------|")
    for i, r in enumerate(top10, 1):
        lines.append(
            f"| {i} | {r.hold_time}s | {r.trail_start}% | {r.trail_offset}% | "
            f"{r.spread_threshold}% | {r.containment_pullback}% | {r.session_trade_cap} | "
            f"{r.total_trades} | {r.win_rate}% | {r.profit_factor} | "
            f"${r.total_pnl:,.0f} | {r.max_drawdown_pct}% | {r.sharpe_ratio} |"
        )
    lines.append("")

    # Comparison: top 1 vs production
    if top10 and prod:
        best = top10[0]
        lines.append("## Best vs Production Comparison")
        lines.append("")
        lines.append("| Metric | Production | Best Config | Delta |")
        lines.append("|--------|-----------|-------------|-------|")
        lines.append(f"| Hold Time | {prod.hold_time}s | {best.hold_time}s | |")
        lines.append(f"| Trail Start | {prod.trail_start}% | {best.trail_start}% | |")
        lines.append(f"| Trail Offset | {prod.trail_offset}% | {best.trail_offset}% | |")
        lines.append(f"| Spread | {prod.spread_threshold}% | {best.spread_threshold}% | |")
        lines.append(f"| Pullback | {prod.containment_pullback}% | {best.containment_pullback}% | |")
        lines.append(f"| Trade Cap | {prod.session_trade_cap} | {best.session_trade_cap} | |")
        lines.append(f"| Trades | {prod.total_trades} | {best.total_trades} | {best.total_trades - prod.total_trades:+d} |")
        lines.append(f"| Win Rate | {prod.win_rate}% | {best.win_rate}% | {best.win_rate - prod.win_rate:+.1f}% |")
        lines.append(f"| Profit Factor | {prod.profit_factor} | {best.profit_factor} | {best.profit_factor - prod.profit_factor:+.3f} |")
        lines.append(f"| Total PnL | ${prod.total_pnl:,.0f} | ${best.total_pnl:,.0f} | ${best.total_pnl - prod.total_pnl:+,.0f} |")
        lines.append(f"| Max Drawdown | {prod.max_drawdown_pct}% | {best.max_drawdown_pct}% | {best.max_drawdown_pct - prod.max_drawdown_pct:+.4f}% |")
        lines.append(f"| Sharpe | {prod.sharpe_ratio} | {best.sharpe_ratio} | {best.sharpe_ratio - prod.sharpe_ratio:+.3f} |")
        lines.append("")

    # Top by other metrics
    lines.append("---")
    lines.append("")
    lines.append("## Top 3 by Total PnL")
    lines.append("")
    lines.append("| Hold | Trail Start | Trail Offset | Spread | Cap | Trades | PnL | PF | Sharpe |")
    lines.append("|------|------------|-------------|--------|-----|--------|------|-----|--------|")
    for r in top_pnl:
        lines.append(f"| {r.hold_time}s | {r.trail_start}% | {r.trail_offset}% | "
                     f"{r.spread_threshold}% | {r.session_trade_cap} | {r.total_trades} | "
                     f"${r.total_pnl:,.0f} | {r.profit_factor} | {r.sharpe_ratio} |")
    lines.append("")

    lines.append("## Top 3 by Sharpe Ratio")
    lines.append("")
    lines.append("| Hold | Trail Start | Trail Offset | Spread | Cap | Trades | PnL | PF | Sharpe |")
    lines.append("|------|------------|-------------|--------|-----|--------|------|-----|--------|")
    for r in top_sharpe:
        lines.append(f"| {r.hold_time}s | {r.trail_start}% | {r.trail_offset}% | "
                     f"{r.spread_threshold}% | {r.session_trade_cap} | {r.total_trades} | "
                     f"${r.total_pnl:,.0f} | {r.profit_factor} | {r.sharpe_ratio} |")
    lines.append("")

    lines.append("## Lowest 3 Max Drawdown (min 10 trades)")
    lines.append("")
    lines.append("| Hold | Trail Start | Trail Offset | Spread | Cap | Trades | PnL | PF | Max DD |")
    lines.append("|------|------------|-------------|--------|-----|--------|------|-----|--------|")
    for r in low_dd:
        lines.append(f"| {r.hold_time}s | {r.trail_start}% | {r.trail_offset}% | "
                     f"{r.spread_threshold}% | {r.session_trade_cap} | {r.total_trades} | "
                     f"${r.total_pnl:,.0f} | {r.profit_factor} | {r.max_drawdown_pct}% |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*NO production changes were made. This is research-only analysis.*")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Top configs written: {path}")


def write_sensitivity(results: List[ConfigResult], path: Path):
    """Write parameter sensitivity analysis."""
    # Group results by each parameter, averaging over all other parameters
    def avg_metric(subset, metric):
        vals = [getattr(r, metric) for r in subset if r.total_trades >= 5]
        return sum(vals) / len(vals) if vals else 0

    lines = []
    lines.append("# Parameter Sensitivity Analysis")
    lines.append("## Shadow Replay Grid — 2026-03-03 BATL")
    lines.append(f"## Generated: 2026-03-06")
    lines.append("")
    lines.append("Each table shows how varying ONE parameter (averaging over all others) affects key metrics.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Hold Time sensitivity
    lines.append("## 1. Hold Time vs Performance")
    lines.append("")
    lines.append("| Hold Time | Avg Trades | Avg WR | Avg PF | Avg PnL | Avg Max DD | Avg Sharpe |")
    lines.append("|-----------|-----------|--------|--------|---------|-----------|-----------|")
    for ht in HOLD_TIMES:
        subset = [r for r in results if r.hold_time == ht]
        lines.append(
            f"| {ht}s | {avg_metric(subset, 'total_trades'):.1f} | "
            f"{avg_metric(subset, 'win_rate'):.1f}% | {avg_metric(subset, 'profit_factor'):.2f} | "
            f"${avg_metric(subset, 'total_pnl'):,.0f} | {avg_metric(subset, 'max_drawdown_pct'):.2f}% | "
            f"{avg_metric(subset, 'sharpe_ratio'):.2f} |"
        )
    lines.append("")

    # Trail Start sensitivity
    lines.append("## 2. Trail Start vs Performance")
    lines.append("")
    lines.append("| Trail Start | Avg Trades | Avg WR | Avg PF | Avg PnL | Avg Max DD | Avg Sharpe |")
    lines.append("|------------|-----------|--------|--------|---------|-----------|-----------|")
    for ts in TRAIL_STARTS:
        subset = [r for r in results if r.trail_start == ts]
        lines.append(
            f"| {ts}% | {avg_metric(subset, 'total_trades'):.1f} | "
            f"{avg_metric(subset, 'win_rate'):.1f}% | {avg_metric(subset, 'profit_factor'):.2f} | "
            f"${avg_metric(subset, 'total_pnl'):,.0f} | {avg_metric(subset, 'max_drawdown_pct'):.2f}% | "
            f"{avg_metric(subset, 'sharpe_ratio'):.2f} |"
        )
    lines.append("")

    # Trail Offset sensitivity
    lines.append("## 3. Trail Offset vs Performance")
    lines.append("")
    lines.append("| Trail Offset | Avg Trades | Avg WR | Avg PF | Avg PnL | Avg Max DD | Avg Sharpe |")
    lines.append("|-------------|-----------|--------|--------|---------|-----------|-----------|")
    for to in TRAIL_OFFSETS:
        subset = [r for r in results if r.trail_offset == to]
        lines.append(
            f"| {to}% | {avg_metric(subset, 'total_trades'):.1f} | "
            f"{avg_metric(subset, 'win_rate'):.1f}% | {avg_metric(subset, 'profit_factor'):.2f} | "
            f"${avg_metric(subset, 'total_pnl'):,.0f} | {avg_metric(subset, 'max_drawdown_pct'):.2f}% | "
            f"{avg_metric(subset, 'sharpe_ratio'):.2f} |"
        )
    lines.append("")

    # Spread Threshold sensitivity
    lines.append("## 4. Spread Threshold vs Performance")
    lines.append("")
    lines.append("| Spread Thr | Avg Trades | Avg WR | Avg PF | Avg PnL | Avg Spread Pass | Avg Sharpe |")
    lines.append("|-----------|-----------|--------|--------|---------|----------------|-----------|")
    for st in SPREAD_THRESHOLDS:
        subset = [r for r in results if r.spread_threshold == st]
        lines.append(
            f"| {st}% | {avg_metric(subset, 'total_trades'):.1f} | "
            f"{avg_metric(subset, 'win_rate'):.1f}% | {avg_metric(subset, 'profit_factor'):.2f} | "
            f"${avg_metric(subset, 'total_pnl'):,.0f} | {avg_metric(subset, 'signals_passed_spread'):.0f} | "
            f"{avg_metric(subset, 'sharpe_ratio'):.2f} |"
        )
    lines.append("")

    # Containment Pullback sensitivity
    lines.append("## 5. Containment Pullback vs Performance")
    lines.append("")
    lines.append("| Pullback | Avg Trades | Avg WR | Avg PF | Avg PnL | Avg Containment Pass | Avg Sharpe |")
    lines.append("|----------|-----------|--------|--------|---------|---------------------|-----------|")
    for cp in CONTAINMENT_PULLBACKS:
        subset = [r for r in results if r.containment_pullback == cp]
        lines.append(
            f"| {cp}% | {avg_metric(subset, 'total_trades'):.1f} | "
            f"{avg_metric(subset, 'win_rate'):.1f}% | {avg_metric(subset, 'profit_factor'):.2f} | "
            f"${avg_metric(subset, 'total_pnl'):,.0f} | {avg_metric(subset, 'signals_passed_containment'):.0f} | "
            f"{avg_metric(subset, 'sharpe_ratio'):.2f} |"
        )
    lines.append("")

    # Session Trade Cap sensitivity
    lines.append("## 6. Session Trade Cap vs Performance")
    lines.append("")
    lines.append("| Trade Cap | Avg Trades | Avg WR | Avg PF | Avg PnL | Avg Max DD | Avg Sharpe |")
    lines.append("|----------|-----------|--------|--------|---------|-----------|-----------|")
    for cap in SESSION_TRADE_CAPS:
        subset = [r for r in results if r.session_trade_cap == cap]
        lines.append(
            f"| {cap} | {avg_metric(subset, 'total_trades'):.1f} | "
            f"{avg_metric(subset, 'win_rate'):.1f}% | {avg_metric(subset, 'profit_factor'):.2f} | "
            f"${avg_metric(subset, 'total_pnl'):,.0f} | {avg_metric(subset, 'max_drawdown_pct'):.2f}% | "
            f"{avg_metric(subset, 'sharpe_ratio'):.2f} |"
        )
    lines.append("")

    # Cross-parameter interactions (2D)
    lines.append("---")
    lines.append("")
    lines.append("## 7. Key Interactions: Hold Time x Trail Offset (Avg PnL)")
    lines.append("")
    header = "| Hold \\ Offset | " + " | ".join(f"{to}%" for to in TRAIL_OFFSETS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TRAIL_OFFSETS) + 1))
    for ht in HOLD_TIMES:
        row = f"| {ht}s"
        for to in TRAIL_OFFSETS:
            subset = [r for r in results if r.hold_time == ht and r.trail_offset == to]
            avg_pnl = avg_metric(subset, 'total_pnl')
            row += f" | ${avg_pnl:,.0f}"
        row += " |"
        lines.append(row)
    lines.append("")

    lines.append("## 8. Key Interactions: Trail Start x Trail Offset (Avg PF)")
    lines.append("")
    header = "| Start \\ Offset | " + " | ".join(f"{to}%" for to in TRAIL_OFFSETS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TRAIL_OFFSETS) + 1))
    for ts in TRAIL_STARTS:
        row = f"| {ts}%"
        for to in TRAIL_OFFSETS:
            subset = [r for r in results if r.trail_start == ts and r.trail_offset == to]
            avg_pf = avg_metric(subset, 'profit_factor')
            row += f" | {avg_pf:.2f}"
        row += " |"
        lines.append(row)
    lines.append("")

    lines.append("## 9. Trade Cap x Spread Threshold (Avg Trades)")
    lines.append("")
    header = "| Cap \\ Spread | " + " | ".join(f"{st}%" for st in SPREAD_THRESHOLDS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(SPREAD_THRESHOLDS) + 1))
    for cap in SESSION_TRADE_CAPS:
        row = f"| {cap}"
        for st in SPREAD_THRESHOLDS:
            subset = [r for r in results if r.session_trade_cap == cap and r.spread_threshold == st]
            avg_trades = avg_metric(subset, 'total_trades')
            row += f" | {avg_trades:.1f}"
        row += " |"
        lines.append(row)
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*NO production changes were made. This is research-only analysis.*")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Sensitivity report written: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results, signals = run_grid()

    print("\nWriting outputs...")
    csv_path = OUTPUT_DIR / "replay_results_table.csv"
    top_path = OUTPUT_DIR / "top_configurations.md"
    sens_path = OUTPUT_DIR / "parameter_sensitivity.md"

    write_csv(results, csv_path)
    write_top_configs(results, top_path)
    write_sensitivity(results, sens_path)

    # Quick summary
    meaningful = [r for r in results if r.total_trades >= 10]
    best_pf = max(meaningful, key=lambda r: r.profit_factor) if meaningful else None
    best_pnl = max(meaningful, key=lambda r: r.total_pnl) if meaningful else None
    best_sharpe = max(meaningful, key=lambda r: r.sharpe_ratio) if meaningful else None

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    if best_pf:
        print(f"\nBest Profit Factor: {best_pf.profit_factor}")
        print(f"  Config: hold={best_pf.hold_time}s trail_start={best_pf.trail_start}% "
              f"trail_offset={best_pf.trail_offset}% spread={best_pf.spread_threshold}% "
              f"pullback={best_pf.containment_pullback}% cap={best_pf.session_trade_cap}")
        print(f"  Trades: {best_pf.total_trades} | WR: {best_pf.win_rate}% | "
              f"PnL: ${best_pf.total_pnl:,.2f} | Sharpe: {best_pf.sharpe_ratio}")

    if best_pnl:
        print(f"\nBest Total PnL: ${best_pnl.total_pnl:,.2f}")
        print(f"  Config: hold={best_pnl.hold_time}s trail_start={best_pnl.trail_start}% "
              f"trail_offset={best_pnl.trail_offset}% spread={best_pnl.spread_threshold}% "
              f"pullback={best_pnl.containment_pullback}% cap={best_pnl.session_trade_cap}")

    if best_sharpe:
        print(f"\nBest Sharpe: {best_sharpe.sharpe_ratio}")
        print(f"  Config: hold={best_sharpe.hold_time}s trail_start={best_sharpe.trail_start}% "
              f"trail_offset={best_sharpe.trail_offset}% spread={best_sharpe.spread_threshold}% "
              f"pullback={best_sharpe.containment_pullback}% cap={best_sharpe.session_trade_cap}")

    print("\n" + "=" * 70)
    print("REPLAY COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()

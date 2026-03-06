"""
Daily Tracker - Discovery-to-EOD Performance Tracking
=====================================================
Tracks each discovered symbol from discovery through end of day,
measuring peak, trough, and final performance.

Outcomes: winner, active, faded, loser, noise.

NO production changes. Research-only tracking.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datetime import datetime, timezone
from pathlib import Path

SUPERBOT = Path(__file__).resolve().parent.parent
QUOTE_DIR = SUPERBOT / "engine" / "cache" / "quotes"


@dataclass
class TrackedSymbol:
    symbol: str
    discovery_price: float
    discovery_timestamp: str
    discovery_source: str       # "scanner", "webull", "tradingview", "manual"
    discovery_tier: str         # "A", "B", "C"
    current_price: float = 0.0
    peak_price: float = 0.0
    trough_price: float = 0.0
    change_pct: float = 0.0
    peak_change_pct: float = 0.0
    drawdown_from_peak_pct: float = 0.0
    outcome: str = "active"     # winner, active, faded, loser, noise

    def to_dict(self) -> dict:
        return asdict(self)


class DailyTracker:
    def __init__(self):
        self.tracked: Dict[str, TrackedSymbol] = {}

    def register(self, symbol: str, price: float, timestamp: str,
                 source: str = "scanner", tier: str = "C") -> Optional[TrackedSymbol]:
        """Register a symbol at discovery time."""
        if symbol in self.tracked:
            return self.tracked[symbol]
        if price <= 0:
            return None

        ts = TrackedSymbol(
            symbol=symbol,
            discovery_price=price,
            discovery_timestamp=timestamp,
            discovery_source=source,
            discovery_tier=tier,
            current_price=price,
            peak_price=price,
            trough_price=price,
        )
        self.tracked[symbol] = ts
        return ts

    def update_price(self, symbol: str, price: float):
        """Update a tracked symbol with a new price."""
        ts = self.tracked.get(symbol)
        if not ts or price <= 0:
            return

        ts.current_price = price
        if price > ts.peak_price:
            ts.peak_price = price
        if price < ts.trough_price:
            ts.trough_price = price

        disc = ts.discovery_price
        ts.change_pct = round((price - disc) / disc * 100, 2)
        ts.peak_change_pct = round((ts.peak_price - disc) / disc * 100, 2)

        if ts.peak_price > disc:
            gain = ts.peak_price - disc
            retreat = ts.peak_price - price
            ts.drawdown_from_peak_pct = round(retreat / gain * 100, 2) if gain > 0 else 0.0
        else:
            ts.drawdown_from_peak_pct = 0.0

        ts.outcome = self._classify_outcome(ts)

    def _classify_outcome(self, ts: TrackedSymbol) -> str:
        """Classify current outcome based on price action."""
        if ts.peak_change_pct >= 10 and ts.change_pct >= 5:
            return "winner"
        if ts.change_pct <= -5:
            return "loser"
        if ts.peak_change_pct >= 5 and ts.drawdown_from_peak_pct > 50:
            return "faded"
        if abs(ts.change_pct) < 2 and ts.peak_change_pct < 3:
            return "noise"
        if ts.change_pct > 0 and ts.drawdown_from_peak_pct < 50:
            return "active"
        return "active"

    def refresh_from_cache(self):
        """Read quote cache files and update all tracked symbols."""
        if not QUOTE_DIR.exists():
            return

        for qf in QUOTE_DIR.glob("*_quotes.json"):
            try:
                with open(qf) as f:
                    data = json.load(f)
                sym = data.get("symbol", "")
                if sym not in self.tracked:
                    continue
                quotes = data.get("quotes", [])
                if not quotes:
                    continue
                quotes.sort(key=lambda q: q["epoch"])
                last_quote = quotes[-1]
                price = last_quote.get("last")
                if price and price > 0:
                    self.update_price(sym, price)
            except (json.JSONDecodeError, KeyError):
                continue

    def get_state(self) -> dict:
        """Full tracker snapshot."""
        symbols = []
        for sym in sorted(self.tracked.keys()):
            symbols.append(self.tracked[sym].to_dict())

        outcome_counts = {}
        for ts in self.tracked.values():
            outcome_counts[ts.outcome] = outcome_counts.get(ts.outcome, 0) + 1

        return {
            "total_tracked": len(self.tracked),
            "outcome_summary": outcome_counts,
            "symbols": symbols,
        }

    def generate_eod_report(self) -> dict:
        """End-of-day report: tier validation + source grouping."""
        self.refresh_from_cache()

        # Group by tier
        by_tier = {"A": [], "B": [], "C": []}
        for ts in self.tracked.values():
            tier_key = ts.discovery_tier if ts.discovery_tier in by_tier else "C"
            by_tier[tier_key].append(ts.to_dict())

        tier_stats = {}
        for tier, entries in by_tier.items():
            if not entries:
                tier_stats[tier] = {"count": 0}
                continue
            changes = [e["change_pct"] for e in entries]
            peaks = [e["peak_change_pct"] for e in entries]
            winners = sum(1 for e in entries if e["outcome"] == "winner")
            losers = sum(1 for e in entries if e["outcome"] == "loser")
            tier_stats[tier] = {
                "count": len(entries),
                "avg_change_pct": round(sum(changes) / len(changes), 2),
                "avg_peak_pct": round(sum(peaks) / len(peaks), 2),
                "winners": winners,
                "losers": losers,
                "win_rate": round(winners / len(entries) * 100, 1) if entries else 0,
            }

        # Group by source
        by_source = {}
        for ts in self.tracked.values():
            src = ts.discovery_source
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(ts.to_dict())

        source_stats = {}
        for src, entries in by_source.items():
            changes = [e["change_pct"] for e in entries]
            source_stats[src] = {
                "count": len(entries),
                "avg_change_pct": round(sum(changes) / len(changes), 2) if changes else 0,
                "winners": sum(1 for e in entries if e["outcome"] == "winner"),
            }

        return {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_tracked": len(self.tracked),
            "tier_performance": tier_stats,
            "by_tier": by_tier,
            "source_performance": source_stats,
            "by_source": by_source,
        }

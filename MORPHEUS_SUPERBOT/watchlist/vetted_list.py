"""
Vetted List - Curated Top 5-10 Watchlist
=========================================
Maintains a curated list of the best actionable stocks:
  - A-class auto-qualifies
  - B-class can be manually promoted (POST)
  - C-class always rejected
  - Auto-removes faded, demoted, or low-score entries

NO production changes. Research-only curation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timezone


@dataclass
class VettedEntry:
    symbol: str
    score: float
    tier: str
    entry_method: str               # "auto" or "manual"
    added_timestamp: str
    added_price: Optional[float] = None
    current_price: Optional[float] = None
    change_since_add_pct: float = 0.0
    late_entry_penalty: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VettedListConfig:
    max_size: int = 10
    late_entry_minutes: int = 60
    late_entry_penalty_pct: int = 10
    auto_remove_fade_pct: float = 5.0
    auto_remove_min_score: float = 0.30

    def to_dict(self) -> dict:
        return asdict(self)


class VettedListManager:
    def __init__(self, classifier, tracker, config=None):
        self.classifier = classifier        # StockClassifierManager
        self.tracker = tracker              # DailyTracker
        self.config = config or VettedListConfig()
        self.vetted: Dict[str, VettedEntry] = {}
        self.removal_log: List[dict] = []
        self.rejection_log: List[dict] = []

    def auto_qualify(self, classification) -> Optional[VettedEntry]:
        """A-class auto-adds to vetted list.

        Late entry penalty raises effective threshold.
        If list is full, must outscore weakest to bump it.
        """
        if classification.tier != "A":
            return None

        symbol = classification.symbol
        if symbol in self.vetted:
            self.vetted[symbol].score = classification.final_score
            return self.vetted[symbol]

        # Check late entry penalty
        late_penalty = False
        if classification.first_seen:
            try:
                from dateutil import parser as dp
                disc_dt = dp.parse(classification.first_seen)
                now = datetime.now(timezone.utc)
                elapsed_min = (now - disc_dt).total_seconds() / 60
                if elapsed_min > self.config.late_entry_minutes:
                    late_penalty = True
            except Exception:
                pass

        # Check capacity
        if len(self.vetted) >= self.config.max_size:
            weakest = min(self.vetted.values(), key=lambda v: v.score)
            if classification.final_score <= weakest.score:
                self.rejection_log.append({
                    "symbol": symbol,
                    "reason": f"full list, score {classification.final_score} <= weakest {weakest.symbol} ({weakest.score})",
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                return None
            # Bump the weakest
            self.removal_log.append({
                "symbol": weakest.symbol,
                "reason": f"bumped by {symbol} (score {classification.final_score} > {weakest.score})",
                "at": datetime.now(timezone.utc).isoformat(),
            })
            del self.vetted[weakest.symbol]

        # Get price from tracker
        tracked = self.tracker.tracked.get(symbol)
        price = tracked.current_price if tracked else classification.entry_price

        entry = VettedEntry(
            symbol=symbol,
            score=classification.final_score,
            tier="A",
            entry_method="auto",
            added_timestamp=datetime.now(timezone.utc).isoformat(),
            added_price=price,
            current_price=price,
            late_entry_penalty=late_penalty,
        )
        self.vetted[symbol] = entry
        return entry

    def manual_add(self, symbol: str, source: str = "manual") -> dict:
        """B-class promotion via manual POST. C-class rejected."""
        cls = self.classifier.get_classification(symbol)
        if cls is None:
            return {"success": False, "reason": f"{symbol} not classified"}

        if cls.tier == "C":
            self.rejection_log.append({
                "symbol": symbol,
                "reason": f"C-class ({cls.final_score}) cannot be manually promoted",
                "at": datetime.now(timezone.utc).isoformat(),
            })
            return {
                "success": False,
                "reason": f"{symbol} is C-class (score {cls.final_score}), rejected",
            }

        if symbol in self.vetted:
            return {
                "success": True,
                "reason": f"{symbol} already on vetted list",
                "entry": self.vetted[symbol].to_dict(),
            }

        # Check capacity
        if len(self.vetted) >= self.config.max_size:
            weakest = min(self.vetted.values(), key=lambda v: v.score)
            if cls.final_score <= weakest.score:
                return {
                    "success": False,
                    "reason": (f"list full, {symbol} ({cls.final_score}) "
                               f"doesn't outscore weakest ({weakest.symbol}: {weakest.score})"),
                }
            self.removal_log.append({
                "symbol": weakest.symbol,
                "reason": f"bumped by manual add of {symbol}",
                "at": datetime.now(timezone.utc).isoformat(),
            })
            del self.vetted[weakest.symbol]

        tracked = self.tracker.tracked.get(symbol)
        price = tracked.current_price if tracked else cls.entry_price

        entry = VettedEntry(
            symbol=symbol,
            score=cls.final_score,
            tier=cls.tier,
            entry_method="manual",
            added_timestamp=datetime.now(timezone.utc).isoformat(),
            added_price=price,
            current_price=price,
            late_entry_penalty=False,
        )
        self.vetted[symbol] = entry
        return {
            "success": True,
            "reason": f"{symbol} ({cls.tier}-class, score {cls.final_score}) added via {source}",
            "entry": entry.to_dict(),
        }

    def refresh(self):
        """Remove stocks that faded, demoted to C, or score below min."""
        to_remove = []

        for sym, entry in self.vetted.items():
            # Update current price from tracker
            tracked = self.tracker.tracked.get(sym)
            if tracked:
                entry.current_price = tracked.current_price
                if entry.added_price and entry.added_price > 0:
                    entry.change_since_add_pct = round(
                        (tracked.current_price - entry.added_price) / entry.added_price * 100, 2)

            # Check fade removal
            if entry.change_since_add_pct <= -self.config.auto_remove_fade_pct:
                to_remove.append((sym, f"faded {entry.change_since_add_pct:.1f}% since add"))
                continue

            # Check if demoted to C
            cls = self.classifier.get_classification(sym)
            if cls and cls.tier == "C":
                to_remove.append((sym, f"demoted to C-class (score {cls.final_score})"))
                continue

            # Check min score (normalized to 0-1 scale)
            if cls:
                entry.score = cls.final_score
                normalized = cls.final_score / 100.0
                if normalized < self.config.auto_remove_min_score:
                    to_remove.append((sym, f"score {cls.final_score} below threshold ({self.config.auto_remove_min_score * 100})"))

        for sym, reason in to_remove:
            self.removal_log.append({
                "symbol": sym,
                "reason": reason,
                "at": datetime.now(timezone.utc).isoformat(),
            })
            del self.vetted[sym]

    def get_vetted_list(self) -> dict:
        """Return vetted list sorted by score, with logs."""
        self.refresh()
        entries = sorted(self.vetted.values(), key=lambda v: -v.score)
        return {
            "count": len(entries),
            "max_size": self.config.max_size,
            "entries": [e.to_dict() for e in entries],
            "removal_log": self.removal_log[-20:],
            "rejection_log": self.rejection_log[-20:],
        }

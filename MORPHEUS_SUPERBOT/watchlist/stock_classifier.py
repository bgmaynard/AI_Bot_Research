"""
Stock Classifier - A/B/C Tier Classification Engine
====================================================
Scores scanner-discovered stocks on a 0-100 scale and classifies into tiers:
  A-Class (Trade it): >=60 with catalyst, >=75 without
  B-Class (Watch it): >=40 but below A threshold
  C-Class (Skip it):  <40

Scoring components: catalyst, relative_volume, gap%, spread,
                    confidence, momentum, scanner_volume.

NO production changes. Research-only classification.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# Paths relative to SUPERBOT root
SUPERBOT = Path(__file__).resolve().parent.parent
CACHE_DIR = SUPERBOT / "engine" / "cache"
REPORTS_DIR = CACHE_DIR / "morpheus_reports"


def _today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def parse_gap_pct(tag: str) -> Optional[float]:
    """Extract gap percentage from tag.

    Handles: gap:54pct, gap:54, gap:5.5pct, gap:5.5
    Returns None for malformed tags.
    """
    m = re.match(r"gap:(\d+(?:\.\d+)?)\s*(?:pct)?$", tag)
    if m:
        return float(m.group(1))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoringWeights:
    catalyst: int = 30
    relative_volume: int = 20
    gap_pct: int = 15
    spread: int = 10
    confidence: int = 10
    momentum: int = 10
    scanner_volume: int = 5


@dataclass
class ClassificationThresholds:
    a_class_with_catalyst: int = 60
    a_class_without_catalyst: int = 75
    b_class_min: int = 40
    overextension_30_penalty: int = 10
    overextension_50_penalty: int = 25
    midday_no_catalyst_boost: int = 10
    midday_hour_utc: int = 14    # 10:30 ET = 14:30 UTC
    midday_min_utc: int = 30


@dataclass
class ScoreComponents:
    catalyst: float = 0.0
    relative_volume: float = 0.0
    gap_pct: float = 0.0
    spread: float = 0.0
    confidence: float = 0.0
    momentum: float = 0.0
    scanner_volume: float = 0.0
    penalties: float = 0.0
    penalty_reasons: List[str] = field(default_factory=list)


@dataclass
class StockClassification:
    symbol: str
    raw_score: float
    final_score: float
    tier: str                          # "A", "B", "C"
    components: ScoreComponents
    has_catalyst: bool
    strategies: List[str]
    tags: List[str]
    entry_price: Optional[float]
    first_seen: Optional[str]
    classified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: List[dict] = field(default_factory=list)
    asset_type: str = "unknown"
    sector: str = "unknown"
    cap_bucket: str = "unknown"
    sector_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier Manager
# ═══════════════════════════════════════════════════════════════════════════════

class StockClassifierManager:
    def __init__(self, weights=None, thresholds=None, sector_classifier=None):
        self.weights = weights or ScoringWeights()
        self.thresholds = thresholds or ClassificationThresholds()
        self.classifications: Dict[str, StockClassification] = {}
        self._gating_data: Dict[str, dict] = {}
        self._sector_classifier = sector_classifier

    # -------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------

    def load_gating_blocks(self, date_str: Optional[str] = None):
        """Load gating_blocks.jsonl to extract spread and extension data."""
        date_str = date_str or _today_str()
        path = REPORTS_DIR / date_str / "gating_blocks.jsonl"
        if not path.exists():
            return

        self._gating_data = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                if not sym:
                    continue

                # Extract spread from IGNITION_GATE failures
                if rec.get("stage") == "IGNITION_GATE":
                    reason = rec.get("reason", "")
                    spread_match = re.search(r"HIGH_SPREAD:\s*([\d.]+)", reason)
                    if spread_match:
                        spread_val = float(spread_match.group(1))
                        existing = self._gating_data.get(sym, {}).get("spread_pct", 999)
                        if spread_val < existing:
                            self._gating_data.setdefault(sym, {})["spread_pct"] = spread_val

                # Extract change_pct from EXTENSION_GATE
                if rec.get("stage") == "EXTENSION_GATE":
                    details = rec.get("details", {})
                    change_pct = details.get("change_pct")
                    if change_pct is not None:
                        self._gating_data.setdefault(sym, {})["change_pct"] = abs(float(change_pct))
                        self._gating_data[sym]["has_catalyst"] = details.get("has_catalyst", False)

    def load_signals(self, date_str: Optional[str] = None):
        """Load signal_ledger.jsonl and classify all discovered symbols."""
        date_str = date_str or _today_str()
        path = REPORTS_DIR / date_str / "signal_ledger.jsonl"
        if not path.exists():
            return

        self.load_gating_blocks(date_str)

        # Aggregate per-symbol data (mirrors max_ai_scanner_audit.py pattern)
        symbols = defaultdict(lambda: {
            "first_ts": None, "signal_count": 0,
            "strategies": [], "tags": [],
            "scanner_scores": [], "gap_pcts": [],
            "entry_prices": [], "has_catalyst": False,
            "confidence_scores": [], "momentum_scores": [],
            "score_rationales": [],
        })

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                if not sym:
                    continue
                s = symbols[sym]
                decision = rec.get("decision", "")

                if decision == "":
                    # Initial signal
                    s["signal_count"] += 1
                    ts = rec.get("timestamp", "")
                    if s["first_ts"] is None or ts < s["first_ts"]:
                        s["first_ts"] = ts
                    strat = rec.get("strategy", "")
                    if strat and strat not in s["strategies"]:
                        s["strategies"].append(strat)
                    for t in rec.get("tags", []):
                        if t not in s["tags"]:
                            s["tags"].append(t)
                        if t.startswith("scanner:"):
                            try:
                                s["scanner_scores"].append(int(t.split(":")[1]))
                            except ValueError:
                                pass
                        if t.startswith("gap:"):
                            gap = parse_gap_pct(t)
                            if gap is not None:
                                s["gap_pcts"].append(gap)
                        if t in ("catalyst", "news", "earnings", "fda", "merger"):
                            s["has_catalyst"] = True
                    ep = rec.get("entry_price")
                    if ep:
                        s["entry_prices"].append(float(ep))

                elif decision == "SIGNAL_SCORED":
                    conf = rec.get("confidence")
                    if conf is not None:
                        s["confidence_scores"].append(float(conf))
                    rationale = rec.get("score_rationale", "")
                    if rationale:
                        s["score_rationales"].append(rationale)

                elif decision == "IGNITION_PASS":
                    ms = rec.get("momentum_snapshot", {})
                    if ms and ms.get("momentum_score") is not None:
                        s["momentum_scores"].append(float(ms["momentum_score"]))

        # Classify each symbol
        for sym, data in symbols.items():
            self.classify_symbol(sym, data)

    # -------------------------------------------------------------------
    # Scoring engine
    # -------------------------------------------------------------------

    def classify_symbol(self, symbol: str, data: dict) -> StockClassification:
        """Score and classify a single symbol on the 0-100 scale."""
        w = self.weights
        thr = self.thresholds
        comp = ScoreComponents()

        tags = data.get("tags", [])
        strategies = data.get("strategies", [])
        gating = self._gating_data.get(symbol, {})

        # --- Catalyst (max 30) ---
        has_catalyst = data.get("has_catalyst", False)
        if gating.get("has_catalyst"):
            has_catalyst = True

        catalyst_score = 0
        if has_catalyst:
            tag_lower = set(t.lower() for t in tags)
            if "fda" in tag_lower or "merger" in tag_lower:
                catalyst_score = w.catalyst          # 30
            elif "earnings" in tag_lower:
                catalyst_score = 20
            elif tag_lower & {"news", "catalyst"}:
                catalyst_score = 10
            else:
                catalyst_score = 10                 # generic catalyst tag
        comp.catalyst = catalyst_score

        # --- Relative Volume (max 20) ---
        rvol = None
        for rationale in data.get("score_rationales", []):
            rv_match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(?:rel|relative|avg)?\s*vol", rationale)
            if rv_match:
                rvol = float(rv_match.group(1))
                break
        if rvol is None:
            # Fallback: momentum_score as proxy
            ms = data.get("momentum_scores", [])
            if ms:
                avg_mom = sum(ms) / len(ms)
                if avg_mom >= 80:
                    rvol = 5.0
                elif avg_mom >= 60:
                    rvol = 2.0
                else:
                    rvol = 1.0

        if rvol is not None:
            if rvol >= 5:
                comp.relative_volume = w.relative_volume    # 20
            elif rvol >= 2:
                comp.relative_volume = 15
            elif rvol >= 1.5:
                comp.relative_volume = 10
            else:
                comp.relative_volume = 0

        # --- Gap % (max 15) ---
        gap_pcts = data.get("gap_pcts", [])
        if gap_pcts:
            max_gap = max(gap_pcts)
            if max_gap >= 20:
                comp.gap_pct = w.gap_pct                    # 15
            elif max_gap >= 10:
                comp.gap_pct = 12
            elif max_gap >= 5:
                comp.gap_pct = 10
            elif max_gap >= 2:
                comp.gap_pct = 5
            else:
                comp.gap_pct = 0

        # --- Spread (max 10) ---
        spread_pct = gating.get("spread_pct")
        if spread_pct is not None:
            if spread_pct <= 0.05:
                comp.spread = w.spread                      # 10
            elif spread_pct <= 0.10:
                comp.spread = 8
            elif spread_pct <= 0.50:
                comp.spread = 5
            else:
                comp.spread = 0

        # --- Confidence (max 10) ---
        conf_scores = data.get("confidence_scores", [])
        if conf_scores:
            avg_conf = sum(conf_scores) / len(conf_scores)
            comp.confidence = round(avg_conf * w.confidence, 1)     # Linear 0-1 -> 0-10

        # --- Momentum (max 10) ---
        mom_scores = data.get("momentum_scores", [])
        if mom_scores:
            avg_mom = sum(mom_scores) / len(mom_scores)
            comp.momentum = round(min(avg_mom / 100.0, 1.0) * w.momentum, 1)

        # --- Scanner/Volume (max 5) ---
        scanner_scores = data.get("scanner_scores", [])
        if scanner_scores:
            max_scanner = max(scanner_scores)
            if max_scanner >= 90:
                comp.scanner_volume = w.scanner_volume      # 5
            elif max_scanner >= 70:
                comp.scanner_volume = 3
            elif max_scanner >= 50:
                comp.scanner_volume = 1
            else:
                comp.scanner_volume = 0

        # --- Raw score ---
        raw_score = (comp.catalyst + comp.relative_volume + comp.gap_pct +
                     comp.spread + comp.confidence + comp.momentum +
                     comp.scanner_volume)

        # --- Penalties ---
        change_pct = gating.get("change_pct")
        if change_pct is None and gap_pcts:
            change_pct = max(gap_pcts)

        penalty = 0
        if change_pct is not None:
            if change_pct > 50:
                penalty += thr.overextension_50_penalty
                comp.penalty_reasons.append(f"overextended {change_pct:.1f}% (don't chase)")
            elif change_pct > 30:
                penalty += thr.overextension_30_penalty
                comp.penalty_reasons.append(f"overextended {change_pct:.1f}%")

        # Midday penalty — after 10:30 ET (14:30 UTC) without catalyst
        first_ts = data.get("first_ts", "")
        is_midday = False
        if first_ts and not has_catalyst:
            try:
                from dateutil import parser as dp
                dt = dp.parse(first_ts)
                utc_minutes = dt.hour * 60 + dt.minute
                midday_threshold = thr.midday_hour_utc * 60 + thr.midday_min_utc
                if utc_minutes >= midday_threshold:
                    is_midday = True
                    comp.penalty_reasons.append("midday without catalyst: threshold +10")
            except Exception:
                pass

        comp.penalties = penalty
        final_score = max(0, raw_score - penalty)

        # --- Tier assignment ---
        midday_boost = thr.midday_no_catalyst_boost if is_midday else 0
        a_threshold = (thr.a_class_with_catalyst if has_catalyst
                       else thr.a_class_without_catalyst) + midday_boost

        if final_score >= a_threshold:
            tier = "A"
        elif final_score >= thr.b_class_min:
            tier = "B"
        else:
            tier = "C"

        entry_prices = data.get("entry_prices", [])
        classification = StockClassification(
            symbol=symbol,
            raw_score=round(raw_score, 1),
            final_score=round(final_score, 1),
            tier=tier,
            components=comp,
            has_catalyst=has_catalyst,
            strategies=strategies,
            tags=tags,
            entry_price=entry_prices[0] if entry_prices else None,
            first_seen=data.get("first_ts"),
        )

        # Track promotion/demotion history
        prev = self.classifications.get(symbol)
        if prev and prev.tier != tier:
            classification.history.append({
                "event": "reclassified",
                "from": prev.tier,
                "to": tier,
                "at": datetime.now(timezone.utc).isoformat(),
                "prev_score": prev.final_score,
                "new_score": final_score,
            })
        elif prev:
            classification.history = prev.history

        # Enrich with sector classification if available
        if self._sector_classifier:
            signal_data = {
                "entry_price": entry_prices[0] if entry_prices else None,
                "spread_pct": gating.get("spread_pct"),
                "gap_pcts": gap_pcts,
                "tags": tags,
            }
            sec_info = self._sector_classifier.classify(symbol, signal_data)
            classification.asset_type = sec_info.asset_type
            classification.sector = sec_info.sector
            classification.cap_bucket = sec_info.cap_bucket
            classification.sector_confidence = sec_info.confidence

        self.classifications[symbol] = classification
        return classification

    # -------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------

    def get_classification(self, symbol: str) -> Optional[StockClassification]:
        return self.classifications.get(symbol)

    def get_tier(self, symbol: str) -> Optional[str]:
        cls = self.classifications.get(symbol)
        return cls.tier if cls else None

    def get_all_classified(self) -> dict:
        """Return all classifications grouped by tier, sector, and asset type."""
        result = {"A": [], "B": [], "C": []}
        by_sector = defaultdict(list)
        by_asset_type = defaultdict(list)
        for sym, cls in sorted(self.classifications.items(),
                                key=lambda x: -x[1].final_score):
            result[cls.tier].append(cls.to_dict())
            by_sector[cls.sector].append(sym)
            by_asset_type[cls.asset_type].append(sym)
        return {
            "total": len(self.classifications),
            "counts": {t: len(v) for t, v in result.items()},
            "tiers": result,
            "by_sector": dict(by_sector),
            "by_asset_type": dict(by_asset_type),
        }

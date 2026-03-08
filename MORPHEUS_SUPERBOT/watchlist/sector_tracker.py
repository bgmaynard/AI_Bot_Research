"""
Sector Performance Tracker — Real-Time Heat Scoring & Parameter Profiles
=========================================================================
Tracks sector "heat" (0-100) and provides per-asset-type trading parameter
profiles with heat-adjusted weights.

Heat zones:
  Hot (>=70):    Trade aggressively — lower thresholds, wider caps
  Normal (40-69): Default parameters
  Cold (20-39):  Pull back — raise thresholds, tighter caps
  Frozen (<20):  Suggest skipping entirely

NO production changes. Research-only tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SectorHeatScore:
    sector: str
    heat_score: float = 50.0
    heat_zone: str = "normal"           # hot, normal, cold, frozen
    intraday_performance: float = 50.0  # 0-100 component
    recent_trade_outcomes: float = 50.0
    historical_trend: float = 50.0
    momentum_signals: float = 50.0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectorWeight:
    sector: str
    weight: float = 1.0                # 0-2
    reason: str = "default"
    spread_multiplier: float = 1.0
    cap_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectorParameterProfile:
    """Per-asset-type trading parameter profile."""
    asset_type: str
    hold_s: int = 300
    trail_start_pct: float = 0.15
    trail_offset_pct: float = 0.10
    spread_gate_pct: float = 0.4
    hard_stop_pct: float = 2.0
    size_multiplier: float = 1.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectorFilterThresholds:
    """Per-asset-type filter thresholds for regime validation."""
    vol_threshold: float = 0.3
    spread_threshold: float = 0.6
    ofi_threshold: float = -0.2
    suppress_regimes: Set[str] = field(default_factory=lambda: {"LOW_VOLATILITY"})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["suppress_regimes"] = list(self.suppress_regimes)
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Default profiles by asset type
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PROFILES = {
    "inverse_etf": SectorParameterProfile(
        asset_type="inverse_etf", hold_s=180,
        trail_start_pct=0.20, trail_offset_pct=0.08,
        spread_gate_pct=1.0, hard_stop_pct=1.5,
        size_multiplier=0.7,
        notes="Wider spreads OK, shorter holds, smaller size",
    ),
    "leveraged_etf": SectorParameterProfile(
        asset_type="leveraged_etf", hold_s=180,
        trail_start_pct=0.20, trail_offset_pct=0.08,
        spread_gate_pct=1.0, hard_stop_pct=1.5,
        size_multiplier=0.7,
        notes="Wider spreads OK, shorter holds, smaller size",
    ),
    "volatility_etf": SectorParameterProfile(
        asset_type="volatility_etf", hold_s=120,
        trail_start_pct=0.30, trail_offset_pct=0.05,
        spread_gate_pct=1.5, hard_stop_pct=2.0,
        size_multiplier=0.5,
        notes="Very short holds, trade only vol spikes",
    ),
    "micro_cap": SectorParameterProfile(
        asset_type="micro_cap", hold_s=420,
        trail_start_pct=0.25, trail_offset_pct=0.05,
        spread_gate_pct=0.8, hard_stop_pct=2.0,
        size_multiplier=0.8,
        notes="Wider spreads, momentum-following",
    ),
    "mid_large_cap": SectorParameterProfile(
        asset_type="mid_large_cap", hold_s=300,
        trail_start_pct=0.15, trail_offset_pct=0.10,
        spread_gate_pct=0.4, hard_stop_pct=2.0,
        size_multiplier=1.0,
        notes="Tighter spreads, conservative",
    ),
    "sector_etf": SectorParameterProfile(
        asset_type="sector_etf", hold_s=240,
        trail_start_pct=0.15, trail_offset_pct=0.08,
        spread_gate_pct=0.5, hard_stop_pct=1.5,
        size_multiplier=0.9,
        notes="Plain sector ETFs — tighter than leveraged",
    ),
    "small_cap": SectorParameterProfile(
        asset_type="small_cap", hold_s=360,
        trail_start_pct=0.20, trail_offset_pct=0.08,
        spread_gate_pct=0.6, hard_stop_pct=2.0,
        size_multiplier=0.8,
        notes="Small-cap equities — wider trails, smaller size",
    ),
    "default": SectorParameterProfile(
        asset_type="default", hold_s=300,
        trail_start_pct=0.15, trail_offset_pct=0.10,
        spread_gate_pct=0.6, hard_stop_pct=2.0,
        size_multiplier=1.0,
        notes="Default parameters",
    ),
}

_DEFAULT_FILTER_THRESHOLDS = {
    "inverse_etf": SectorFilterThresholds(
        vol_threshold=0.5, spread_threshold=1.0,
        ofi_threshold=-0.3, suppress_regimes=set(),
    ),
    "leveraged_etf": SectorFilterThresholds(
        vol_threshold=0.5, spread_threshold=1.0,
        ofi_threshold=-0.3, suppress_regimes=set(),
    ),
    "volatility_etf": SectorFilterThresholds(
        vol_threshold=0.5, spread_threshold=1.5,
        ofi_threshold=-0.3, suppress_regimes={"LOW_VOLATILITY"},
    ),
    "micro_cap": SectorFilterThresholds(
        vol_threshold=0.3, spread_threshold=0.8,
        ofi_threshold=-0.2, suppress_regimes={"LOW_VOLATILITY"},
    ),
    "mid_large_cap": SectorFilterThresholds(
        vol_threshold=0.2, spread_threshold=0.4,
        ofi_threshold=-0.15, suppress_regimes={"LOW_VOLATILITY"},
    ),
    "sector_etf": SectorFilterThresholds(
        vol_threshold=0.2, spread_threshold=0.5,
        ofi_threshold=-0.2, suppress_regimes={"LOW_VOLATILITY"},
    ),
    "small_cap": SectorFilterThresholds(
        vol_threshold=0.25, spread_threshold=0.6,
        ofi_threshold=-0.2, suppress_regimes={"LOW_VOLATILITY"},
    ),
    "default": SectorFilterThresholds(
        vol_threshold=0.3, spread_threshold=0.6,
        ofi_threshold=-0.2, suppress_regimes={"LOW_VOLATILITY"},
    ),
}


def _resolve_profile_key(sector_classification):
    """Map a SectorClassification to a profile key."""
    if sector_classification is None:
        return "default"
    at = sector_classification.asset_type
    if at in ("inverse_etf", "leveraged_etf", "volatility_etf"):
        return at
    if at == "sector_etf":
        return "sector_etf"
    cap = sector_classification.cap_bucket
    if cap == "micro":
        return "micro_cap"
    if cap == "small":
        return "small_cap"
    if cap in ("mid", "large"):
        return "mid_large_cap"
    return "default"


def _heat_zone(score: float) -> str:
    if score >= 70:
        return "hot"
    if score >= 40:
        return "normal"
    if score >= 20:
        return "cold"
    return "frozen"


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# ═══════════════════════════════════════════════════════════════════════════════
# Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class SectorPerformanceTracker:
    """Track sector heat scores and provide parameter profiles."""

    def __init__(self, sector_classifier=None, daily_tracker=None):
        self._sector_classifier = sector_classifier
        self._daily_tracker = daily_tracker
        self._heat_scores: Dict[str, SectorHeatScore] = {}
        self._overrides: Dict[str, SectorWeight] = {}
        self._trade_outcomes: Dict[str, list] = {}     # sector -> list of pnl
        self._scorecard_data: Optional[dict] = None

    def set_scorecard_data(self, data: dict):
        """Inject scorecard data for historical trend scoring."""
        self._scorecard_data = data

    def record_trade_outcome(self, sector: str, pnl: float):
        """Record a trade outcome for a sector."""
        self._trade_outcomes.setdefault(sector, []).append(pnl)

    # ------------------------------------------------------------------
    # Heat scoring
    # ------------------------------------------------------------------

    def compute_heat_score(self, sector: str) -> SectorHeatScore:
        """Compute heat score for one sector.

        Components (weighted):
          30% - Intraday performance (from tracker)
          30% - Recent trade outcomes (win rate of completed trades)
          20% - Historical trend (multi-day PF from scorecard)
          20% - Momentum signals (avg momentum of sector symbols)
        """
        intraday = 50.0
        trade_outcome = 50.0
        historical = 50.0
        momentum = 50.0

        # Component 1: Intraday performance from daily tracker
        if self._daily_tracker and self._sector_classifier:
            sector_syms = self._sector_classifier.get_group(sector=sector)
            changes = []
            for sc in sector_syms:
                ts = self._daily_tracker.tracked.get(sc.symbol)
                if ts:
                    changes.append(ts.change_pct)
            if changes:
                avg_change = sum(changes) / len(changes)
                # Map: >5% = 100, 0% = 50, <-5% = 0
                intraday = _clamp(50.0 + avg_change * 10.0)

        # Component 2: Recent trade outcomes
        outcomes = self._trade_outcomes.get(sector, [])
        if outcomes:
            wins = sum(1 for p in outcomes if p > 0)
            wr = wins / len(outcomes) * 100
            # Map: >50% = 100, 33% = 50, <20% = 0
            if wr >= 50:
                trade_outcome = 100.0
            elif wr >= 33:
                trade_outcome = 50.0 + (wr - 33) / 17 * 50
            elif wr >= 20:
                trade_outcome = (wr - 20) / 13 * 50
            else:
                trade_outcome = 0.0

        # Component 3: Historical trend from scorecard
        if self._scorecard_data:
            days = self._scorecard_data.get("days", [])
            if days:
                # Use overall filtered PF as proxy for now
                latest = days[-1]
                filt_pf = latest.get("filtered", {}).get("pf")
                if isinstance(filt_pf, (int, float)):
                    # Map: PF>1.5=100, PF=1.0=50, PF<0.7=0
                    if filt_pf >= 1.5:
                        historical = 100.0
                    elif filt_pf >= 1.0:
                        historical = 50.0 + (filt_pf - 1.0) / 0.5 * 50
                    elif filt_pf >= 0.7:
                        historical = (filt_pf - 0.7) / 0.3 * 50
                    else:
                        historical = 0.0

        # Component 4: Momentum signals
        if self._sector_classifier and self._daily_tracker:
            sector_syms = self._sector_classifier.get_group(sector=sector)
            mom_scores = []
            for sc in sector_syms:
                ts = self._daily_tracker.tracked.get(sc.symbol)
                if ts and ts.peak_change_pct > 0:
                    # Use peak_change as momentum proxy (0-100)
                    mom_scores.append(min(ts.peak_change_pct * 5, 100))
            if mom_scores:
                momentum = sum(mom_scores) / len(mom_scores)

        # Weighted average
        heat = (intraday * 0.30 + trade_outcome * 0.30 +
                historical * 0.20 + momentum * 0.20)
        heat = round(_clamp(heat), 1)

        score = SectorHeatScore(
            sector=sector,
            heat_score=heat,
            heat_zone=_heat_zone(heat),
            intraday_performance=round(intraday, 1),
            recent_trade_outcomes=round(trade_outcome, 1),
            historical_trend=round(historical, 1),
            momentum_signals=round(momentum, 1),
        )
        self._heat_scores[sector] = score
        return score

    def compute_all_heat_scores(self) -> Dict[str, SectorHeatScore]:
        """Compute heat scores for all known sectors."""
        if not self._sector_classifier:
            return {}
        sectors = set()
        for cls in self._sector_classifier._classifications.values():
            sectors.add(cls.sector)
        for sector in sorted(sectors):
            self.compute_heat_score(sector)
        return dict(self._heat_scores)

    # ------------------------------------------------------------------
    # Weights and profiles
    # ------------------------------------------------------------------

    def get_sector_weight(self, sector: str) -> SectorWeight:
        """Get weight for a sector based on heat or override."""
        # Manual override takes precedence
        if sector in self._overrides:
            return self._overrides[sector]

        heat = self._heat_scores.get(sector)
        if not heat:
            return SectorWeight(sector=sector, weight=1.0, reason="no heat data")

        if heat.heat_score >= 70:
            return SectorWeight(
                sector=sector, weight=1.5,
                reason="hot sector (heat=%.0f)" % heat.heat_score,
                spread_multiplier=1.2, cap_multiplier=1.5,
            )
        elif heat.heat_score >= 40:
            return SectorWeight(
                sector=sector, weight=1.0,
                reason="normal (heat=%.0f)" % heat.heat_score,
            )
        elif heat.heat_score >= 20:
            return SectorWeight(
                sector=sector, weight=0.5,
                reason="cold sector (heat=%.0f)" % heat.heat_score,
                spread_multiplier=0.8, cap_multiplier=0.6,
            )
        else:
            return SectorWeight(
                sector=sector, weight=0.0,
                reason="frozen sector (heat=%.0f) — skip" % heat.heat_score,
                spread_multiplier=0.5, cap_multiplier=0.3,
            )

    def get_parameter_profile(self, symbol: str) -> SectorParameterProfile:
        """Get parameter profile for a symbol based on its asset type."""
        cls = None
        if self._sector_classifier:
            cls = self._sector_classifier.get_classification(symbol)
        key = _resolve_profile_key(cls)
        return _DEFAULT_PROFILES.get(key, _DEFAULT_PROFILES["default"])

    def get_adjusted_filter_thresholds(self, symbol: str) -> dict:
        """Get per-symbol filter thresholds for regime validation.

        Returns:
            dict with keys: vol_threshold, spread_threshold, ofi_threshold,
            suppress_regimes (as set).
        """
        cls = None
        if self._sector_classifier:
            cls = self._sector_classifier.get_classification(symbol)
        key = _resolve_profile_key(cls)
        ft = _DEFAULT_FILTER_THRESHOLDS.get(key, _DEFAULT_FILTER_THRESHOLDS["default"])
        return {
            "vol_threshold": ft.vol_threshold,
            "spread_threshold": ft.spread_threshold,
            "ofi_threshold": ft.ofi_threshold,
            "suppress_regimes": set(ft.suppress_regimes),
        }

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def set_override(self, sector: str, weight: float, reason: str = "manual"):
        """Manual sector weight override."""
        self._overrides[sector] = SectorWeight(
            sector=sector, weight=weight, reason=reason,
        )

    def clear_override(self, sector: str):
        """Remove manual override for a sector."""
        self._overrides.pop(sector, None)

    # ------------------------------------------------------------------
    # State / reporting
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Full tracker snapshot for API."""
        return {
            "heat_scores": {
                s: h.to_dict() for s, h in sorted(self._heat_scores.items())
            },
            "weights": {
                s: self.get_sector_weight(s).to_dict()
                for s in sorted(self._heat_scores.keys())
            },
            "overrides": {
                s: w.to_dict() for s, w in sorted(self._overrides.items())
            },
            "profiles": {
                k: v.to_dict() for k, v in sorted(_DEFAULT_PROFILES.items())
            },
            "filter_thresholds": {
                k: v.to_dict() for k, v in sorted(_DEFAULT_FILTER_THRESHOLDS.items())
            },
        }

    def generate_sector_report(self) -> str:
        """Generate per-sector performance summary as markdown text."""
        lines = []
        lines.append("## SECTOR HEAT MAP")
        lines.append("")
        lines.append("| Sector | Heat | Zone | Intraday | Trades | Historical | Momentum |")
        lines.append("|--------|------|------|----------|--------|------------|----------|")

        for sector in sorted(self._heat_scores.keys()):
            h = self._heat_scores[sector]
            lines.append("| %s | %.0f | %s | %.0f | %.0f | %.0f | %.0f |" % (
                sector, h.heat_score, h.heat_zone,
                h.intraday_performance, h.recent_trade_outcomes,
                h.historical_trend, h.momentum_signals))

        lines.append("")
        lines.append("## SECTOR WEIGHTS")
        lines.append("")
        lines.append("| Sector | Weight | Spread Mult | Cap Mult | Reason |")
        lines.append("|--------|--------|-------------|----------|--------|")

        for sector in sorted(self._heat_scores.keys()):
            w = self.get_sector_weight(sector)
            lines.append("| %s | %.1f | %.1f | %.1f | %s |" % (
                sector, w.weight, w.spread_multiplier, w.cap_multiplier, w.reason))

        if self._overrides:
            lines.append("")
            lines.append("### Manual Overrides Active")
            for s, w in sorted(self._overrides.items()):
                lines.append("- **%s**: weight=%.1f (%s)" % (s, w.weight, w.reason))

        return "\n".join(lines)

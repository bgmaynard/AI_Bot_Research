"""
Morpheus Lab — Promotion Gate
=================================
Formal criteria for promoting a strategy from research to shadow/production.

This is the gatekeeper of reality.
No strategy reaches a live bot without passing these thresholds.

Process:
  Research PC (edge discovery)
    → Promotion Gate (this module)
      → Shadow mode (paper trading on live data)
        → Production (real capital)

Thresholds are conservative by design.
Better to miss a marginal edge than to deploy an overfit one.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.edge_metrics import EdgeMetrics

logger = logging.getLogger(__name__)


@dataclass
class PromotionThresholds:
    """Configurable promotion criteria."""
    min_profit_factor: float = 1.15
    max_drawdown_pct_of_pnl: float = 30.0   # DD < 30% of total PnL
    min_trades: int = 200
    min_oos_profit_factor: float = 1.05      # out-of-sample
    min_stability_ratio: float = 0.8         # OOS_PF / IS_PF
    min_win_rate: float = 20.0               # minimum % (low for momentum)
    min_reward_risk: float = 1.0             # R:R minimum
    max_avg_loser_multiple: float = 3.0      # avg loser < N × avg winner
    # Regime robustness
    min_best_regime_pf: float = 1.25         # best regime must be strong
    min_worst_regime_pf: float = 0.85        # worst regime can't be destructive


@dataclass
class PromotionDecision:
    """Result of a promotion gate check."""
    qualifies: bool = False
    score: float = 0.0  # 0-100 composite score
    rejections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "QUALIFIES FOR SHADOW" if self.qualifies else "REJECTED"
        lines = [
            f"Promotion Decision: {status}",
            f"Score: {self.score:.0f}/100",
        ]
        if self.rejections:
            lines.append("Rejections:")
            for r in self.rejections:
                lines.append(f"  ✗ {r}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def qualifies_for_shadow(
    metrics: EdgeMetrics,
    oos_metrics: Optional[EdgeMetrics] = None,
    stability_ratio: Optional[float] = None,
    thresholds: Optional[PromotionThresholds] = None,
    regime_breakdown: Optional[Dict] = None,
) -> PromotionDecision:
    """
    Check if a strategy qualifies for shadow trading.

    Args:
        metrics: In-sample (or full-period) EdgeMetrics
        oos_metrics: Out-of-sample EdgeMetrics (from walk-forward)
        stability_ratio: Pre-computed OOS_PF / IS_PF
        thresholds: Custom thresholds (defaults if None)
        regime_breakdown: Dict from compute_regime_breakdown() for robustness check

    Returns:
        PromotionDecision with pass/fail and rejection reasons.
    """
    t = thresholds or PromotionThresholds()
    d = PromotionDecision()
    d.details = {"thresholds": t.__dict__.copy()}

    # ── HARD GATES (any failure = rejection) ─────────────

    # 1. Minimum trades
    if metrics.total_trades < t.min_trades:
        d.rejections.append(
            f"Insufficient trades: {metrics.total_trades} < {t.min_trades} required"
        )

    # 2. Profit factor
    pf = metrics.profit_factor
    if pf == float('inf'):
        pf_check = True  # No losses = technically infinite PF
        d.warnings.append("Infinite profit factor (no losses — likely insufficient data)")
    elif pf < t.min_profit_factor:
        d.rejections.append(
            f"Profit factor too low: {pf:.2f} < {t.min_profit_factor} required"
        )

    # 3. Max drawdown relative to PnL
    if metrics.total_pnl > 0 and metrics.max_drawdown > 0:
        dd_pct_of_pnl = (metrics.max_drawdown / metrics.total_pnl) * 100
        d.details["dd_pct_of_pnl"] = round(dd_pct_of_pnl, 1)
        if dd_pct_of_pnl > t.max_drawdown_pct_of_pnl:
            d.rejections.append(
                f"Drawdown too high relative to PnL: {dd_pct_of_pnl:.0f}% > {t.max_drawdown_pct_of_pnl}%"
            )
    elif metrics.total_pnl <= 0:
        d.rejections.append(
            f"Net losing strategy: total PnL = ${metrics.total_pnl:.2f}"
        )

    # 4. Win rate floor
    if metrics.win_rate < t.min_win_rate:
        d.rejections.append(
            f"Win rate too low: {metrics.win_rate:.1f}% < {t.min_win_rate}%"
        )

    # 5. R:R minimum
    if metrics.reward_risk < t.min_reward_risk:
        d.rejections.append(
            f"R:R too low: {metrics.reward_risk:.2f} < {t.min_reward_risk} required"
        )

    # 6. Out-of-sample profit factor
    if oos_metrics is not None:
        oos_pf = oos_metrics.profit_factor
        if oos_pf != float('inf') and oos_pf < t.min_oos_profit_factor:
            d.rejections.append(
                f"OOS profit factor too low: {oos_pf:.2f} < {t.min_oos_profit_factor} required"
            )
        elif oos_pf == float('inf'):
            d.warnings.append("Infinite OOS profit factor (possible data insufficiency)")

        # Trade count in OOS
        if oos_metrics.total_trades < 20:
            d.warnings.append(
                f"Very few OOS trades ({oos_metrics.total_trades}) — results may not be reliable"
            )

    # 7. Stability ratio
    if stability_ratio is not None:
        d.details["stability_ratio"] = round(stability_ratio, 3)
        if stability_ratio < t.min_stability_ratio:
            d.rejections.append(
                f"Stability ratio too low: {stability_ratio:.3f} < {t.min_stability_ratio} required"
            )

    # ── WARNINGS (non-blocking) ──────────────────────────

    if metrics.avg_hold_seconds < 1.0:
        d.warnings.append(
            f"Very short avg hold time ({metrics.avg_hold_seconds:.1f}s) — possible scalping noise"
        )

    if metrics.pnl_stddev > abs(metrics.avg_pnl) * 10:
        d.warnings.append(
            f"High PnL variance (stddev ${metrics.pnl_stddev:.2f} >> avg ${metrics.avg_pnl:.2f})"
        )

    if metrics.avg_loser != 0 and abs(metrics.avg_loser) > abs(metrics.avg_winner) * t.max_avg_loser_multiple:
        d.warnings.append(
            f"Avg loser (${metrics.avg_loser:.2f}) much larger than avg winner (${metrics.avg_winner:.2f})"
        )

    # ── REGIME ROBUSTNESS (if breakdown provided) ────────

    if regime_breakdown:
        # Only check regimes with >= 20 trades (statistically meaningful)
        sig_regimes = {
            r: m for r, m in regime_breakdown.items()
            if m.get("trades", 0) >= 20
        }

        if sig_regimes:
            pf_values = []
            for r, m in sig_regimes.items():
                pf_val = m.get("profit_factor", 0)
                if isinstance(pf_val, str):  # "inf"
                    pf_val = 99.0
                pf_values.append((r, pf_val))

            best_regime, best_pf = max(pf_values, key=lambda x: x[1])
            worst_regime, worst_pf = min(pf_values, key=lambda x: x[1])

            d.details["best_regime"] = best_regime
            d.details["best_regime_pf"] = round(best_pf, 2)
            d.details["worst_regime"] = worst_regime
            d.details["worst_regime_pf"] = round(worst_pf, 2)
            d.details["regime_spread"] = round(best_pf - worst_pf, 2)

            if best_pf < t.min_best_regime_pf:
                d.rejections.append(
                    f"Best regime PF too low: {best_regime}={best_pf:.2f} < {t.min_best_regime_pf} required"
                )

            if worst_pf < t.min_worst_regime_pf:
                d.rejections.append(
                    f"Worst regime PF too low: {worst_regime}={worst_pf:.2f} < {t.min_worst_regime_pf} required"
                )

            if best_pf - worst_pf > 0.5:
                d.warnings.append(
                    f"Wide regime spread: {best_regime}={best_pf:.2f} vs {worst_regime}={worst_pf:.2f} (Δ={best_pf - worst_pf:.2f})"
                )

    # ── COMPOSITE SCORE ──────────────────────────────────

    d.score = _compute_score(metrics, oos_metrics, stability_ratio, t)

    # ── FINAL DECISION ───────────────────────────────────

    d.qualifies = len(d.rejections) == 0

    return d


def _compute_score(
    m: EdgeMetrics,
    oos: Optional[EdgeMetrics],
    stability: Optional[float],
    t: PromotionThresholds,
) -> float:
    """
    Composite promotion score (0-100).

    Weighted components:
      30% - Profit factor
      20% - R:R ratio
      15% - Win rate
      15% - Stability (if available)
      10% - Drawdown control
      10% - Trade count confidence
    """
    score = 0.0

    # Profit factor: 0-30 points
    pf = min(m.profit_factor, 5.0) if m.profit_factor != float('inf') else 5.0
    score += (pf / 5.0) * 30

    # R:R: 0-20 points
    rr = min(m.reward_risk, 5.0)
    score += (rr / 5.0) * 20

    # Win rate: 0-15 points (scaled 20-60% → 0-15)
    wr = max(0, min(m.win_rate - 20, 40)) / 40
    score += wr * 15

    # Stability: 0-15 points
    if stability is not None:
        stab = min(max(stability, 0), 1.5) / 1.5
        score += stab * 15
    else:
        score += 7.5  # neutral if no OOS data

    # Drawdown control: 0-10 points
    if m.total_pnl > 0 and m.max_drawdown > 0:
        dd_ratio = m.max_drawdown / m.total_pnl
        dd_score = max(0, 1 - dd_ratio) * 10
        score += dd_score
    elif m.total_pnl > 0:
        score += 10  # no drawdown = perfect

    # Trade count confidence: 0-10 points
    tc = min(m.total_trades, 500) / 500
    score += tc * 10

    return round(score, 1)

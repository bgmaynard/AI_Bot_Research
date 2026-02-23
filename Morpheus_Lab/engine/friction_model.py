"""
Morpheus Lab — Realistic Friction Model
==========================================
Post-trade friction layer for stress testing strategy robustness.

Applies realistic execution costs to backtest trades WITHOUT
modifying signal logic. This is a pure measurement tool.

Friction components:
  1. Slippage: Entry fills worse by N ticks, exits fill worse by N ticks
  2. Latency: Synthetic delay approximated as additional slippage
  3. Spread: Fixed half-spread penalty deducted per side (entry + exit)
  4. Commission: Flat per-trade fee

Architecture:
  Strategy.on_batch() → raw trades → FrictionModel.apply() → adjusted trades

The raw trade list is never modified. A new list of adjusted BatchTrade
copies is returned with modified entry/exit prices. Since BatchTrade.pnl
is a @property computed from prices, PnL recalculates automatically.

Usage:
    friction = FrictionModel(slippage_ticks=1, spread_cost=0.01, commission=0.50)
    adjusted_trades = friction.apply(raw_trades)
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

from strategies.batch_strategy import BatchTrade


# Default tick size for sub-$20 equities (NASDAQ penny increment)
DEFAULT_TICK_SIZE = 0.01


@dataclass
class FrictionConfig:
    """Configuration for all friction components."""

    slippage_ticks: int = 1        # Ticks of adverse fill on entry AND exit
    latency_ticks: int = 0         # Additional ticks of delay (modeled as slippage)
    spread_cost: float = 0.0       # Fixed half-spread per side (total = 2x per trade)
    commission: float = 0.0        # Flat fee per trade (deducted from PnL)
    tick_size: float = DEFAULT_TICK_SIZE

    @property
    def total_slippage_ticks(self) -> int:
        """Combined slippage + latency in ticks."""
        return self.slippage_ticks + self.latency_ticks

    @property
    def entry_penalty(self) -> float:
        """Total adverse adjustment on entry price."""
        return (self.total_slippage_ticks * self.tick_size) + self.spread_cost

    @property
    def exit_penalty(self) -> float:
        """Total adverse adjustment on exit price."""
        return (self.slippage_ticks * self.tick_size) + self.spread_cost

    def total_cost_per_trade(self, share_size: int = 100) -> float:
        """Total friction cost per trade in dollars."""
        price_penalty = (self.entry_penalty + self.exit_penalty) * share_size
        return price_penalty + self.commission

    def describe(self) -> str:
        """Human-readable description of friction settings."""
        parts = []
        if self.slippage_ticks > 0:
            parts.append(f"slippage={self.slippage_ticks} tick(s)")
        if self.latency_ticks > 0:
            parts.append(f"latency={self.latency_ticks} tick(s)")
        if self.spread_cost > 0:
            parts.append(f"spread=${self.spread_cost:.4f}/side")
        if self.commission > 0:
            parts.append(f"commission=${self.commission:.2f}/trade")
        return ", ".join(parts) if parts else "none (frictionless)"


class FrictionModel:
    """
    Apply realistic execution friction to backtest trades.

    Does NOT modify signal logic. Creates adjusted copies of trades
    with worse fills to stress test edge robustness.

    For LONG trades:
      - Entry price moves UP (worse fill)
      - Exit price moves DOWN (worse fill)

    For SHORT trades:
      - Entry price moves DOWN (worse fill)
      - Exit price moves UP (worse fill)
    """

    def __init__(self, config: FrictionConfig):
        self.config = config

    @classmethod
    def from_args(
        cls,
        slippage_ticks: int = 1,
        latency_ticks: int = 0,
        spread_cost: float = 0.0,
        commission: float = 0.0,
        tick_size: float = DEFAULT_TICK_SIZE,
    ) -> "FrictionModel":
        """Convenience constructor from individual parameters."""
        config = FrictionConfig(
            slippage_ticks=slippage_ticks,
            latency_ticks=latency_ticks,
            spread_cost=spread_cost,
            commission=commission,
            tick_size=tick_size,
        )
        return cls(config)

    def apply(self, trades: List[BatchTrade]) -> List[BatchTrade]:
        """
        Apply friction to a list of trades.

        Returns a NEW list of adjusted BatchTrade objects.
        The original trades are not modified.

        Commission is handled separately since BatchTrade.pnl is a
        @property based on prices. We adjust prices for slippage/spread,
        then track commission as metadata for the comparison report.
        """
        if not trades:
            return []

        adjusted = []
        cfg = self.config

        entry_penalty = cfg.entry_penalty
        exit_penalty = cfg.exit_penalty

        for t in trades:
            # Create a copy with adjusted prices
            new_t = BatchTrade(
                symbol=t.symbol,
                entry_ts=t.entry_ts,
                exit_ts=t.exit_ts,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                size=t.size,
                direction=t.direction,
                exit_reason=t.exit_reason,
                entry_regime=t.entry_regime,
                entry_type=t.entry_type,
            )

            # Apply directional friction
            if t.direction == 1:  # LONG
                new_t.entry_price = t.entry_price + entry_penalty
                new_t.exit_price = t.exit_price - exit_penalty
            else:  # SHORT
                new_t.entry_price = t.entry_price - entry_penalty
                new_t.exit_price = t.exit_price + exit_penalty

            adjusted.append(new_t)

        return adjusted

    def total_commission(self, trades: List[BatchTrade]) -> float:
        """Total commission across all trades."""
        return len(trades) * self.config.commission


def compute_friction_comparison(
    raw_trades: List[BatchTrade],
    adjusted_trades: List[BatchTrade],
    friction_config: FrictionConfig,
) -> Dict:
    """
    Compute side-by-side comparison of raw vs friction-adjusted metrics.

    Returns a dict with both metric sets and deltas.
    """
    from engine.edge_metrics import compute_edge_metrics

    raw_m = compute_edge_metrics(raw_trades, strategy_name="baseline")
    adj_m = compute_edge_metrics(adjusted_trades, strategy_name="friction")

    # Commission adjustment (not captured in price-based PnL)
    total_commission = len(adjusted_trades) * friction_config.commission
    adj_total_pnl = adj_m.total_pnl - total_commission

    # Recompute PF with commission
    gross_win_adj = sum(t.pnl for t in adjusted_trades if t.pnl > 0)
    gross_loss_adj = abs(sum(t.pnl for t in adjusted_trades if t.pnl < 0))
    # Distribute commission proportionally: deduct from gross wins
    gross_win_after_comm = max(0, gross_win_adj - total_commission)
    pf_adj = gross_win_after_comm / gross_loss_adj if gross_loss_adj > 0 else float('inf')

    # Winners/losers may shift due to friction
    adj_winners = sum(1 for t in adjusted_trades if t.pnl > 0)
    adj_losers = sum(1 for t in adjusted_trades if t.pnl < 0)
    adj_decided = adj_winners + adj_losers
    adj_win_rate = (adj_winners / adj_decided * 100) if adj_decided > 0 else 0.0

    comparison = {
        "friction": {
            "slippage_ticks": friction_config.slippage_ticks,
            "latency_ticks": friction_config.latency_ticks,
            "spread_cost": friction_config.spread_cost,
            "commission": friction_config.commission,
            "tick_size": friction_config.tick_size,
            "total_slippage_ticks": friction_config.total_slippage_ticks,
            "cost_per_trade_100sh": round(friction_config.total_cost_per_trade(100), 2),
        },
        "baseline": {
            "trades": raw_m.total_trades,
            "winners": raw_m.winners,
            "losers": raw_m.losers,
            "win_rate": round(raw_m.win_rate, 1),
            "total_pnl": round(raw_m.total_pnl, 2),
            "avg_pnl": round(raw_m.avg_pnl, 2),
            "avg_winner": round(raw_m.avg_winner, 2),
            "avg_loser": round(raw_m.avg_loser, 2),
            "rr": round(raw_m.reward_risk, 2),
            "profit_factor": round(raw_m.profit_factor, 2),
            "sharpe": round(raw_m.sharpe, 2),
            "max_drawdown": round(raw_m.max_drawdown, 2),
        },
        "friction_adjusted": {
            "trades": adj_m.total_trades,
            "winners": adj_winners,
            "losers": adj_losers,
            "win_rate": round(adj_win_rate, 1),
            "total_pnl": round(adj_total_pnl, 2),
            "avg_pnl": round(adj_total_pnl / len(adjusted_trades), 2) if adjusted_trades else 0,
            "avg_winner": round(adj_m.avg_winner, 2),
            "avg_loser": round(adj_m.avg_loser, 2),
            "rr": round(adj_m.reward_risk, 2),
            "profit_factor": round(pf_adj, 2),
            "sharpe": round(adj_m.sharpe, 2),
            "max_drawdown": round(adj_m.max_drawdown, 2),
            "total_commission": round(total_commission, 2),
        },
        "delta": {
            "win_rate": round(adj_win_rate - raw_m.win_rate, 1),
            "total_pnl": round(adj_total_pnl - raw_m.total_pnl, 2),
            "profit_factor": round(pf_adj - raw_m.profit_factor, 2),
            "rr": round(adj_m.reward_risk - raw_m.reward_risk, 2),
            "sharpe": round(adj_m.sharpe - raw_m.sharpe, 2),
        },
        "verdict": "",
    }

    # Verdict
    if pf_adj >= 1.4:
        comparison["verdict"] = "ROBUST — Edge survives realistic friction (PF >= 1.4)"
    elif pf_adj >= 1.2:
        comparison["verdict"] = "MARGINAL — Edge degraded but positive (1.2 <= PF < 1.4)"
    elif pf_adj >= 1.0:
        comparison["verdict"] = "FRAGILE — Edge barely positive (1.0 <= PF < 1.2)"
    else:
        comparison["verdict"] = "COLLAPSED — No edge under friction (PF < 1.0)"

    return comparison


def print_friction_comparison(comparison: Dict) -> None:
    """Pretty-print the friction comparison report."""

    fc = comparison["friction"]
    base = comparison["baseline"]
    adj = comparison["friction_adjusted"]
    delta = comparison["delta"]

    print(f"\n{'='*68}")
    print(f"  FRICTION STRESS TEST — flush_reclaim_v1")
    print(f"{'='*68}")
    print(f"  Slippage:    {fc['slippage_ticks']} tick(s) entry + exit")
    print(f"  Latency:     {fc['latency_ticks']} tick(s) additional delay")
    print(f"  Spread:      ${fc['spread_cost']:.4f} per side")
    print(f"  Commission:  ${fc['commission']:.2f} per trade")
    print(f"  Tick size:   ${fc['tick_size']:.4f}")
    print(f"  Total cost:  ${fc['cost_per_trade_100sh']:.2f} per trade (100 shares)")
    print(f"{'='*68}")

    print(f"\n  {'Metric':<20} {'Baseline':>12} {'Friction':>12} {'Delta':>10}")
    print(f"  {'─'*56}")

    rows = [
        ("Trades",        f"{base['trades']}",         f"{adj['trades']}",         ""),
        ("Winners",       f"{base['winners']}",        f"{adj['winners']}",        ""),
        ("Losers",        f"{base['losers']}",         f"{adj['losers']}",         ""),
        ("Win Rate",      f"{base['win_rate']:.1f}%",  f"{adj['win_rate']:.1f}%",  f"{delta['win_rate']:+.1f}%"),
        ("Total PnL",     f"${base['total_pnl']:,.2f}", f"${adj['total_pnl']:,.2f}", f"${delta['total_pnl']:+,.2f}"),
        ("Avg PnL",       f"${base['avg_pnl']:.2f}",  f"${adj['avg_pnl']:.2f}",   ""),
        ("Avg Winner",    f"${base['avg_winner']:.2f}", f"${adj['avg_winner']:.2f}", ""),
        ("Avg Loser",     f"${base['avg_loser']:.2f}", f"${adj['avg_loser']:.2f}", ""),
        ("R:R",           f"{base['rr']:.2f}",         f"{adj['rr']:.2f}",         f"{delta['rr']:+.2f}"),
        ("Profit Factor", f"{base['profit_factor']:.2f}", f"{adj['profit_factor']:.2f}", f"{delta['profit_factor']:+.2f}"),
        ("Sharpe",        f"{base['sharpe']:.2f}",     f"{adj['sharpe']:.2f}",     f"{delta['sharpe']:+.2f}"),
        ("Max Drawdown",  f"${base['max_drawdown']:.2f}", f"${adj['max_drawdown']:.2f}", ""),
    ]

    if adj.get("total_commission", 0) > 0:
        rows.append(("Commission",  "", f"${adj['total_commission']:.2f}", ""))

    for label, b, a, d in rows:
        print(f"  {label:<20} {b:>12} {a:>12} {d:>10}")

    print(f"  {'─'*56}")
    print(f"\n  VERDICT: {comparison['verdict']}")
    print(f"{'='*68}\n")

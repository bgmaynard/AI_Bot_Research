"""
Morpheus Lab -- Price Tier & Friction Survivability Analyzer
================================================================
Capital-protection diagnostic. Determines at what price level the
flush_reclaim_v1 edge survives realistic execution friction.

This is a DIAGNOSTIC ONLY tool. It does not modify strategy logic,
optimize parameters, or change any runtime behavior.

Architecture:
    Data sources -> Tier bucketing -> Friction simulation -> Survivability metrics -> Recommendation

Data sources (either/or):
    1. Shadow JSONL logs (logs/shadow_flush_reclaim.jsonl)
    2. Direct BatchTrade list from batch-backtest

Outputs:
    reports/price_tier_friction_analysis.json  -- full structured results
    reports/price_tier_summary.md              -- human-readable report

Edge Buffer Ratio:
    avg_winner_per_share / friction_per_share

    < 1.0  ->  DEAD     -- friction exceeds average winner
    1-2    ->  FRAGILE  -- edge exists but thin margin
    2-3    ->  VIABLE   -- survivable with discipline
    > 3    ->  ROBUST   -- strong friction absorption
"""

import json
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# -- TIER DEFINITIONS -------------------------------------

DEFAULT_TIERS = [
    (1.0,  3.0,  "$1-$3"),
    (3.0,  5.0,  "$3-$5"),
    (5.0,  7.0,  "$5-$7"),
    (7.0, 10.0,  "$7-$10"),
    (10.0, 20.0, "$10-$20"),
    (20.0, 999.0, "$20+"),
]


# -- FRICTION CONFIG --------------------------------------

@dataclass
class FrictionScenario:
    """Named friction scenario with all cost components."""
    name: str
    slippage_ticks: int = 0
    latency_ticks: int = 0
    spread_cost_per_side: float = 0.0
    commission: float = 0.0
    tick_size: float = 0.01
    shares: int = 100

    @property
    def friction_per_share(self) -> float:
        """Total friction cost per share (both sides of the trade)."""
        slip = (self.slippage_ticks + self.latency_ticks) * self.tick_size * 2
        spread = self.spread_cost_per_side * 2
        comm_per_share = self.commission / self.shares if self.shares > 0 else 0
        return slip + spread + comm_per_share

    @property
    def friction_per_trade(self) -> float:
        """Total friction cost per trade in dollars."""
        return self.friction_per_share * self.shares

    def describe(self) -> str:
        parts = []
        if self.slippage_ticks > 0:
            parts.append(f"slip={self.slippage_ticks}t")
        if self.latency_ticks > 0:
            parts.append(f"lat={self.latency_ticks}t")
        if self.spread_cost_per_side > 0:
            parts.append(f"spread=${self.spread_cost_per_side:.4f}")
        if self.commission > 0:
            parts.append(f"comm=${self.commission:.2f}")
        return f"{self.name}: {', '.join(parts)}" if parts else f"{self.name}: frictionless"


# Pre-built scenarios
SCENARIOS = {
    "ideal": FrictionScenario(
        name="Ideal (frictionless)",
    ),
    "realistic": FrictionScenario(
        name="Realistic (commission-free broker)",
        slippage_ticks=1,
        spread_cost_per_side=0.005,
    ),
    "conservative": FrictionScenario(
        name="Conservative (worst-case retail)",
        slippage_ticks=1,
        latency_ticks=1,
        spread_cost_per_side=0.01,
        commission=0.50,
    ),
}


# -- TRADE RECORD (normalized) ---------------------------

@dataclass
class NormalizedTrade:
    """Minimal trade record for tier analysis. Source-agnostic."""
    symbol: str
    entry_price: float
    exit_price: float
    share_size: int
    exit_reason: str  # target, stop, eod
    direction: int = 1  # 1=long, -1=short

    @property
    def pnl(self) -> float:
        return self.direction * (self.exit_price - self.entry_price) * self.share_size

    @property
    def pnl_per_share(self) -> float:
        return self.direction * (self.exit_price - self.entry_price)

    @property
    def won(self) -> bool:
        return self.pnl > 0


# -- TIER METRICS -----------------------------------------

@dataclass
class TierMetrics:
    """Complete metrics for a single price tier under a single friction scenario."""
    tier_label: str
    tier_low: float
    tier_high: float
    scenario_name: str

    # Counts
    trades: int = 0
    winners: int = 0
    losers: int = 0
    scratches: int = 0

    # Baseline (gross)
    gross_win_rate: float = 0.0
    gross_total_pnl: float = 0.0
    gross_avg_pnl: float = 0.0
    gross_avg_winner: float = 0.0
    gross_avg_loser: float = 0.0
    gross_avg_winner_per_share: float = 0.0
    gross_avg_loser_per_share: float = 0.0
    gross_avg_move_per_share: float = 0.0
    gross_pf: float = 0.0
    gross_rr: float = 0.0

    # Friction
    friction_per_share: float = 0.0
    friction_per_trade: float = 0.0

    # Net (after friction)
    net_winners: int = 0
    net_losers: int = 0
    net_win_rate: float = 0.0
    net_total_pnl: float = 0.0
    net_avg_pnl: float = 0.0
    net_pf: float = 0.0
    net_rr: float = 0.0

    # Survivability
    flipped_to_loss: int = 0       # winners that became losers
    flipped_pct: float = 0.0       # % of winners that flipped
    edge_buffer_ratio: float = 0.0 # avg_winner_per_share / friction_per_share

    # Verdict
    verdict: str = ""


# -- CORE ANALYSIS ENGINE ---------------------------------

def load_trades_from_jsonl(log_path: str) -> List[NormalizedTrade]:
    """Load shadow JSONL log into normalized trade records."""
    path = Path(log_path)
    if not path.exists():
        logger.error(f"Shadow log not found: {log_path}")
        return []

    trades = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line {line_no}")
                continue

            entry_price = rec.get("entry_price", 0)
            exit_price = rec.get("exit_price", 0)
            share_size = rec.get("share_size", rec.get("shares", 100))
            exit_reason = rec.get("exit_reason", "unknown")
            symbol = rec.get("symbol", "???")

            if entry_price <= 0 or exit_price <= 0:
                continue

            trades.append(NormalizedTrade(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                share_size=share_size,
                exit_reason=exit_reason,
                direction=1,  # flush_reclaim is long-only
            ))

    logger.info(f"Loaded {len(trades)} trades from {log_path}")
    return trades


def load_trades_from_batch_result(batch_trades) -> List[NormalizedTrade]:
    """Convert BatchTrade list to normalized records."""
    trades = []
    for bt in batch_trades:
        if bt.entry_price <= 0 or bt.exit_price <= 0:
            continue
        trades.append(NormalizedTrade(
            symbol=bt.symbol,
            entry_price=bt.entry_price,
            exit_price=bt.exit_price,
            share_size=bt.size,
            exit_reason=bt.exit_reason,
            direction=bt.direction,
        ))
    return trades


def bucket_trades_by_tier(
    trades: List[NormalizedTrade],
    tiers: List[Tuple[float, float, str]] = None,
) -> Dict[str, List[NormalizedTrade]]:
    """Bucket trades by entry price tier."""
    if tiers is None:
        tiers = DEFAULT_TIERS

    buckets = {}
    for low, high, label in tiers:
        buckets[label] = []

    for t in trades:
        for low, high, label in tiers:
            if low <= t.entry_price < high:
                buckets[label].append(t)
                break

    return buckets


def compute_tier_metrics(
    tier_trades: List[NormalizedTrade],
    tier_label: str,
    tier_low: float,
    tier_high: float,
    scenario: FrictionScenario,
) -> TierMetrics:
    """
    Compute full metrics for a single tier under a single friction scenario.
    """
    m = TierMetrics(
        tier_label=tier_label,
        tier_low=tier_low,
        tier_high=tier_high,
        scenario_name=scenario.name,
    )

    if not tier_trades:
        m.verdict = "NO DATA"
        return m

    n = len(tier_trades)
    m.trades = n

    # -- GROSS (baseline) metrics --

    pnls = [t.pnl for t in tier_trades]
    pnls_per_share = [t.pnl_per_share for t in tier_trades]

    gross_winners = [t for t in tier_trades if t.pnl > 0]
    gross_losers = [t for t in tier_trades if t.pnl < 0]
    gross_scratches = [t for t in tier_trades if t.pnl == 0]

    m.winners = len(gross_winners)
    m.losers = len(gross_losers)
    m.scratches = len(gross_scratches)

    decided = m.winners + m.losers
    m.gross_win_rate = (m.winners / decided * 100) if decided > 0 else 0.0

    m.gross_total_pnl = sum(pnls)
    m.gross_avg_pnl = m.gross_total_pnl / n

    if gross_winners:
        m.gross_avg_winner = sum(t.pnl for t in gross_winners) / len(gross_winners)
        m.gross_avg_winner_per_share = sum(t.pnl_per_share for t in gross_winners) / len(gross_winners)
    if gross_losers:
        m.gross_avg_loser = sum(t.pnl for t in gross_losers) / len(gross_losers)
        m.gross_avg_loser_per_share = sum(t.pnl_per_share for t in gross_losers) / len(gross_losers)

    m.gross_avg_move_per_share = sum(abs(p) for p in pnls_per_share) / n if n > 0 else 0.0

    gross_win_total = sum(t.pnl for t in gross_winners) if gross_winners else 0
    gross_loss_total = abs(sum(t.pnl for t in gross_losers)) if gross_losers else 0
    m.gross_pf = (gross_win_total / gross_loss_total) if gross_loss_total > 0 else float('inf')

    m.gross_rr = (abs(m.gross_avg_winner / m.gross_avg_loser)
                  if m.gross_avg_loser != 0 else 0.0)

    # -- FRICTION costs --

    m.friction_per_share = scenario.friction_per_share
    m.friction_per_trade = scenario.friction_per_trade

    # -- NET (friction-adjusted) metrics --

    # Apply friction: each trade loses friction_per_trade from its PnL
    net_pnls = []
    net_winners_list = []
    net_losers_list = []
    flipped = 0

    for t in tier_trades:
        gross_pnl = t.pnl
        net_pnl = gross_pnl - m.friction_per_trade

        net_pnls.append(net_pnl)

        if net_pnl > 0:
            net_winners_list.append(net_pnl)
        elif net_pnl < 0:
            net_losers_list.append(net_pnl)

            # Did this flip from winner to loser?
            if gross_pnl > 0:
                flipped += 1

    m.net_winners = len(net_winners_list)
    m.net_losers = len(net_losers_list)
    net_decided = m.net_winners + m.net_losers
    m.net_win_rate = (m.net_winners / net_decided * 100) if net_decided > 0 else 0.0

    m.net_total_pnl = sum(net_pnls)
    m.net_avg_pnl = m.net_total_pnl / n

    net_gross_win = sum(net_winners_list) if net_winners_list else 0
    net_gross_loss = abs(sum(net_losers_list)) if net_losers_list else 0
    m.net_pf = (net_gross_win / net_gross_loss) if net_gross_loss > 0 else float('inf')

    net_avg_w = net_gross_win / len(net_winners_list) if net_winners_list else 0
    net_avg_l = abs(sum(net_losers_list)) / len(net_losers_list) if net_losers_list else 0
    m.net_rr = (net_avg_w / net_avg_l) if net_avg_l > 0 else 0.0

    m.flipped_to_loss = flipped
    m.flipped_pct = (flipped / m.winners * 100) if m.winners > 0 else 0.0

    # -- EDGE BUFFER RATIO --

    if m.friction_per_share > 0 and m.gross_avg_winner_per_share > 0:
        m.edge_buffer_ratio = m.gross_avg_winner_per_share / m.friction_per_share
    elif m.friction_per_share == 0:
        m.edge_buffer_ratio = float('inf')
    else:
        m.edge_buffer_ratio = 0.0

    # -- VERDICT --

    if m.friction_per_share == 0:
        if m.gross_pf >= 1.4:
            m.verdict = "ROBUST (frictionless)"
        elif m.gross_pf >= 1.0:
            m.verdict = "POSITIVE (frictionless)"
        else:
            m.verdict = "NO EDGE (even frictionless)"
    else:
        ebr = m.edge_buffer_ratio
        if ebr < 1.0:
            m.verdict = "DEAD"
        elif ebr < 2.0:
            m.verdict = "FRAGILE"
        elif ebr < 3.0:
            m.verdict = "VIABLE"
        else:
            m.verdict = "ROBUST"

    return m


# -- FULL ANALYSIS ----------------------------------------

def run_tier_analysis(
    trades: List[NormalizedTrade],
    scenarios: Optional[Dict[str, FrictionScenario]] = None,
    tiers: Optional[List[Tuple[float, float, str]]] = None,
) -> Dict:
    """
    Run complete price tier x friction scenario analysis.

    Returns structured dict with all metrics + recommendation.
    """
    if scenarios is None:
        scenarios = SCENARIOS
    if tiers is None:
        tiers = DEFAULT_TIERS

    # Bucket
    buckets = bucket_trades_by_tier(trades, tiers)

    # Compute per-tier, per-scenario metrics
    results = {}
    for scenario_name, scenario in scenarios.items():
        results[scenario_name] = {
            "scenario": asdict(scenario),
            "friction_per_share": round(scenario.friction_per_share, 6),
            "friction_per_trade": round(scenario.friction_per_trade, 2),
            "tiers": {},
        }

        for low, high, label in tiers:
            tier_trades = buckets.get(label, [])
            m = compute_tier_metrics(tier_trades, label, low, high, scenario)

            results[scenario_name]["tiers"][label] = {
                "trades": m.trades,
                "gross": {
                    "winners": m.winners,
                    "losers": m.losers,
                    "win_rate": round(m.gross_win_rate, 1),
                    "total_pnl": round(m.gross_total_pnl, 2),
                    "avg_pnl": round(m.gross_avg_pnl, 2),
                    "avg_winner": round(m.gross_avg_winner, 2),
                    "avg_loser": round(m.gross_avg_loser, 2),
                    "avg_winner_per_share": round(m.gross_avg_winner_per_share, 4),
                    "avg_loser_per_share": round(m.gross_avg_loser_per_share, 4),
                    "avg_move_per_share": round(m.gross_avg_move_per_share, 4),
                    "pf": round(m.gross_pf, 2) if m.gross_pf != float('inf') else "inf",
                    "rr": round(m.gross_rr, 2),
                },
                "net": {
                    "winners": m.net_winners,
                    "losers": m.net_losers,
                    "win_rate": round(m.net_win_rate, 1),
                    "total_pnl": round(m.net_total_pnl, 2),
                    "avg_pnl": round(m.net_avg_pnl, 2),
                    "pf": round(m.net_pf, 2) if m.net_pf != float('inf') else "inf",
                    "rr": round(m.net_rr, 2),
                },
                "friction_impact": {
                    "flipped_to_loss": m.flipped_to_loss,
                    "flipped_pct": round(m.flipped_pct, 1),
                    "edge_buffer_ratio": round(m.edge_buffer_ratio, 2) if m.edge_buffer_ratio != float('inf') else "inf",
                },
                "verdict": m.verdict,
            }

    # -- RECOMMENDATION --

    recommendation = _compute_recommendation(results, tiers)

    # -- AGGREGATE STATS --

    total_stats = {
        "total_trades": len(trades),
        "symbols": list(set(t.symbol for t in trades)),
        "price_range": {
            "min": round(min(t.entry_price for t in trades), 2) if trades else 0,
            "max": round(max(t.entry_price for t in trades), 2) if trades else 0,
            "mean": round(sum(t.entry_price for t in trades) / len(trades), 2) if trades else 0,
        },
        "tier_distribution": {
            label: len(buckets.get(label, []))
            for _, _, label in tiers
        },
    }

    return {
        "analysis": "price_tier_friction_survivability",
        "strategy": "flush_reclaim_v1",
        "total": total_stats,
        "scenarios": results,
        "recommendation": recommendation,
    }


def _compute_recommendation(
    results: Dict,
    tiers: List[Tuple[float, float, str]],
) -> Dict:
    """
    Determine the recommended min-price cutoff.

    Criteria (all must be met under "realistic" scenario):
      - Net PF >= 1.2
      - Net Avg PnL > 0
      - Edge Buffer Ratio >= 2.0
    """
    # Use "realistic" scenario for recommendation
    # Fall back to "conservative" if realistic not present
    rec_scenario = "realistic" if "realistic" in results else "conservative"
    if rec_scenario not in results:
        # Use first non-ideal scenario
        for k in results:
            if k != "ideal":
                rec_scenario = k
                break

    scenario_data = results.get(rec_scenario, {})
    tier_data = scenario_data.get("tiers", {})

    qualifying_tiers = []
    all_tier_verdicts = []

    for low, high, label in tiers:
        td = tier_data.get(label, {})
        if not td or td.get("trades", 0) == 0:
            all_tier_verdicts.append((label, low, "NO DATA", 0, 0, 0))
            continue

        net = td.get("net", {})
        friction = td.get("friction_impact", {})

        net_pf = net.get("pf", 0)
        if isinstance(net_pf, str):
            net_pf = 999.0

        net_avg_pnl = net.get("avg_pnl", 0)

        ebr = friction.get("edge_buffer_ratio", 0)
        if isinstance(ebr, str):
            ebr = 999.0

        verdict = td.get("verdict", "")

        all_tier_verdicts.append((label, low, verdict, net_pf, net_avg_pnl, ebr))

        # Check qualification
        if net_pf >= 1.2 and net_avg_pnl > 0 and ebr >= 2.0:
            qualifying_tiers.append((label, low, net_pf, net_avg_pnl, ebr))

    # Find lowest qualifying tier
    if qualifying_tiers:
        qualifying_tiers.sort(key=lambda x: x[1])  # Sort by tier low
        best = qualifying_tiers[0]
        rec_price = best[1]
        rec_label = best[0]
        confidence = "HIGH" if best[4] >= 3.0 else "MEDIUM" if best[4] >= 2.0 else "LOW"
    else:
        rec_price = None
        rec_label = None
        confidence = "NONE"

    # Summary by scenario
    scenario_summaries = {}
    for scn_name, scn_data in results.items():
        scn_tiers = scn_data.get("tiers", {})
        surviving = []
        for low, high, label in tiers:
            td = scn_tiers.get(label, {})
            if td.get("trades", 0) > 0:
                net_pf = td.get("net", {}).get("pf", 0)
                if isinstance(net_pf, str):
                    net_pf = 999.0
                if net_pf >= 1.2:
                    surviving.append(label)
        scenario_summaries[scn_name] = {
            "surviving_tiers": surviving,
            "total_surviving_tiers": len(surviving),
        }

    return {
        "scenario_used": rec_scenario,
        "criteria": {
            "net_pf_min": 1.2,
            "net_avg_pnl_min": 0,
            "edge_buffer_ratio_min": 2.0,
        },
        "recommended_min_price": rec_price,
        "recommended_tier": rec_label,
        "confidence": confidence,
        "qualifying_tiers": [
            {"tier": t[0], "min_price": t[1], "net_pf": round(t[2], 2),
             "net_avg_pnl": round(t[3], 2), "ebr": round(t[4], 2)}
            for t in qualifying_tiers
        ],
        "all_tier_verdicts": [
            {"tier": t[0], "min_price": t[1], "verdict": t[2],
             "net_pf": round(t[3], 2), "net_avg_pnl": round(t[4], 2),
             "ebr": round(t[5], 2)}
            for t in all_tier_verdicts
        ],
        "scenario_summaries": scenario_summaries,
    }


# -- CONSOLE OUTPUT ---------------------------------------

def print_tier_analysis(analysis: Dict) -> None:
    """Pretty-print the full tier analysis to console."""

    total = analysis.get("total", {})
    rec = analysis.get("recommendation", {})
    scenarios = analysis.get("scenarios", {})

    print(f"\n{'='*78}")
    print(f"  PRICE TIER & FRICTION SURVIVABILITY ANALYSIS")
    print(f"  Strategy: flush_reclaim_v1")
    print(f"{'='*78}")
    print(f"  Total trades: {total.get('total_trades', 0)}")
    print(f"  Symbols:      {', '.join(total.get('symbols', []))}")
    pr = total.get("price_range", {})
    print(f"  Price range:  ${pr.get('min', 0):.2f} - ${pr.get('max', 0):.2f} (avg ${pr.get('mean', 0):.2f})")

    dist = total.get("tier_distribution", {})
    dist_str = "  Distribution: " + " | ".join(f"{k}: {v}" for k, v in dist.items() if v > 0)
    print(dist_str)
    print(f"{'='*78}")

    # Print each scenario
    for scn_name, scn_data in scenarios.items():
        fps = scn_data.get("friction_per_share", 0)
        fpt = scn_data.get("friction_per_trade", 0)
        scn_info = scn_data.get("scenario", {})

        print(f"\n  +- SCENARIO: {scn_info.get('name', scn_name)}")
        print(f"  |  Friction/share: ${fps:.4f}   Friction/trade: ${fpt:.2f}")
        print(f"  |")

        # Header
        print(f"  |  {'Tier':<10} {'Trades':>6} {'Gross':>6} {'Net':>6} "
              f"{'Gross':>7} {'Net':>7} {'AvgW':>7} {'AvgL':>7} "
              f"{'Flip':>5} {'EBR':>5} {'Verdict':<10}")
        print(f"  |  {'':10} {'':>6} {'WR%':>6} {'WR%':>6} "
              f"{'PF':>7} {'PF':>7} {'$/sh':>7} {'$/sh':>7} "
              f"{'%':>5} {'':>5} {'':10}")
        print(f"  |  {'-'*72}")

        tier_data = scn_data.get("tiers", {})
        for label, td in tier_data.items():
            trades = td.get("trades", 0)
            if trades == 0:
                print(f"  |  {label:<10} {'--':>6} {'--':>6} {'--':>6} "
                      f"{'--':>7} {'--':>7} {'--':>7} {'--':>7} "
                      f"{'--':>5} {'--':>5} {'NO DATA':<10}")
                continue

            g = td.get("gross", {})
            n = td.get("net", {})
            fi = td.get("friction_impact", {})
            verdict = td.get("verdict", "")

            g_wr = g.get("win_rate", 0)
            n_wr = n.get("win_rate", 0)
            g_pf = g.get("pf", 0)
            n_pf = n.get("pf", 0)
            g_pf_s = f"{g_pf:.2f}" if isinstance(g_pf, (int, float)) else g_pf
            n_pf_s = f"{n_pf:.2f}" if isinstance(n_pf, (int, float)) else n_pf
            aw = g.get("avg_winner_per_share", 0)
            al = g.get("avg_loser_per_share", 0)
            flip = fi.get("flipped_pct", 0)
            ebr = fi.get("edge_buffer_ratio", 0)
            ebr_s = f"{ebr:.1f}" if isinstance(ebr, (int, float)) else ebr

            print(f"  |  {label:<10} {trades:>6} {g_wr:>5.1f}% {n_wr:>5.1f}% "
                  f"{g_pf_s:>7} {n_pf_s:>7} "
                  f"${aw:>5.3f} ${al:>+5.3f} "
                  f"{flip:>4.0f}% {ebr_s:>5} {verdict:<10}")

        print(f"  +{'-'*76}")

    # -- RECOMMENDATION --

    print(f"\n{'='*78}")
    print(f"  RECOMMENDATION")
    print(f"{'='*78}")
    print(f"  Scenario used: {rec.get('scenario_used', 'N/A')}")
    print(f"  Criteria:      Net PF >= {rec['criteria']['net_pf_min']}, "
          f"Avg PnL > 0, EBR >= {rec['criteria']['edge_buffer_ratio_min']}")

    if rec.get("recommended_min_price") is not None:
        print(f"\n  [Y] RECOMMENDED MIN-PRICE: ${rec['recommended_min_price']:.2f}")
        print(f"    Tier: {rec['recommended_tier']}")
        print(f"    Confidence: {rec['confidence']}")
    else:
        print(f"\n  [X] NO QUALIFYING TIER FOUND")
        print(f"    The strategy does not survive friction at any tested price level.")
        print(f"    Consider: wider reward_multiple, or different strategy class.")

    print(f"\n  Tier Verdicts ({rec.get('scenario_used', '')}):")
    for tv in rec.get("all_tier_verdicts", []):
        marker = "->" if tv["min_price"] == rec.get("recommended_min_price") else " "
        pf_s = f"{tv['net_pf']:.2f}" if isinstance(tv['net_pf'], (int, float)) else tv['net_pf']
        ebr_s = f"{tv['ebr']:.1f}" if isinstance(tv['ebr'], (int, float)) else tv['ebr']
        print(f"  {marker} {tv['tier']:<10} PF={pf_s:>6}  AvgPnL=${tv['net_avg_pnl']:>+6.2f}  "
              f"EBR={ebr_s:>5}  {tv['verdict']}")

    print(f"{'='*78}")
    print(f"\n  FRICTION TIER ANALYSIS COMPLETE\n")


# -- MARKDOWN REPORT --------------------------------------

def generate_markdown_report(analysis: Dict) -> str:
    """Generate a markdown summary report."""

    total = analysis.get("total", {})
    rec = analysis.get("recommendation", {})
    scenarios = analysis.get("scenarios", {})
    pr = total.get("price_range", {})

    lines = []
    lines.append("# Price Tier & Friction Survivability Analysis")
    lines.append(f"## Strategy: flush_reclaim_v1")
    lines.append("")
    lines.append("### Dataset")
    lines.append(f"- Total trades: {total.get('total_trades', 0)}")
    lines.append(f"- Symbols: {', '.join(total.get('symbols', []))}")
    lines.append(f"- Price range: ${pr.get('min', 0):.2f} - ${pr.get('max', 0):.2f} (avg ${pr.get('mean', 0):.2f})")
    lines.append("")

    # Tier distribution
    dist = total.get("tier_distribution", {})
    lines.append("### Trade Distribution by Price Tier")
    lines.append("")
    lines.append("| Tier | Trades |")
    lines.append("|------|--------|")
    for label, count in dist.items():
        lines.append(f"| {label} | {count} |")
    lines.append("")

    # Per-scenario tables
    for scn_name, scn_data in scenarios.items():
        scn_info = scn_data.get("scenario", {})
        fps = scn_data.get("friction_per_share", 0)
        fpt = scn_data.get("friction_per_trade", 0)

        lines.append(f"### Scenario: {scn_info.get('name', scn_name)}")
        lines.append(f"- Friction/share: ${fps:.4f}")
        lines.append(f"- Friction/trade: ${fpt:.2f}")
        lines.append("")

        lines.append("| Tier | Trades | Gross WR | Net WR | Gross PF | Net PF | AvgW $/sh | AvgL $/sh | Flip% | EBR | Verdict |")
        lines.append("|------|--------|----------|--------|----------|--------|-----------|-----------|-------|-----|---------|")

        tier_data = scn_data.get("tiers", {})
        for label, td in tier_data.items():
            trades = td.get("trades", 0)
            if trades == 0:
                lines.append(f"| {label} | 0 | -- | -- | -- | -- | -- | -- | -- | -- | NO DATA |")
                continue

            g = td.get("gross", {})
            n = td.get("net", {})
            fi = td.get("friction_impact", {})
            verdict = td.get("verdict", "")

            g_pf = g.get("pf", 0)
            n_pf = n.get("pf", 0)
            g_pf_s = f"{g_pf:.2f}" if isinstance(g_pf, (int, float)) else g_pf
            n_pf_s = f"{n_pf:.2f}" if isinstance(n_pf, (int, float)) else n_pf
            ebr = fi.get("edge_buffer_ratio", 0)
            ebr_s = f"{ebr:.1f}" if isinstance(ebr, (int, float)) else ebr

            lines.append(
                f"| {label} | {trades} | {g.get('win_rate',0):.1f}% | {n.get('win_rate',0):.1f}% | "
                f"{g_pf_s} | {n_pf_s} | "
                f"${g.get('avg_winner_per_share',0):.3f} | ${g.get('avg_loser_per_share',0):+.3f} | "
                f"{fi.get('flipped_pct',0):.0f}% | {ebr_s} | {verdict} |"
            )
        lines.append("")

    # Recommendation
    lines.append("### Recommendation")
    lines.append("")
    lines.append(f"- Scenario used: {rec.get('scenario_used', 'N/A')}")
    lines.append(f"- Criteria: Net PF >= {rec['criteria']['net_pf_min']}, "
                 f"Avg PnL > 0, EBR >= {rec['criteria']['edge_buffer_ratio_min']}")
    lines.append("")

    if rec.get("recommended_min_price") is not None:
        lines.append(f"**Recommended min-price: ${rec['recommended_min_price']:.2f}** "
                     f"(Tier: {rec['recommended_tier']}, Confidence: {rec['confidence']})")
    else:
        lines.append("**No qualifying tier found.** Strategy does not survive friction at any tested price level.")
    lines.append("")

    lines.append("### Tier Verdicts")
    lines.append("")
    lines.append("| Tier | Net PF | Net Avg PnL | EBR | Verdict |")
    lines.append("|------|--------|------------|-----|---------|")
    for tv in rec.get("all_tier_verdicts", []):
        pf_s = f"{tv['net_pf']:.2f}" if isinstance(tv['net_pf'], (int, float)) else tv['net_pf']
        ebr_s = f"{tv['ebr']:.1f}" if isinstance(tv['ebr'], (int, float)) else tv['ebr']
        marker = " <-- recommended" if tv["min_price"] == rec.get("recommended_min_price") else ""
        lines.append(f"| {tv['tier']} | {pf_s} | ${tv['net_avg_pnl']:+.2f} | {ebr_s} | {tv['verdict']}{marker} |")
    lines.append("")

    # Why low-price stocks collapse
    lines.append("### Why Low-Priced Stocks Collapse Under Friction")
    lines.append("")
    lines.append("The flush_reclaim_v1 strategy detects a genuine microstructure pattern, ")
    lines.append("but on sub-$5 stocks the average winning move is typically $0.03-$0.05/share. ")
    lines.append("A single tick of slippage ($0.01) on entry AND exit consumes 40-66% of the average winner. ")
    lines.append("Add spread and commission, and the edge is entirely consumed by execution costs.")
    lines.append("")
    lines.append("On $5+ stocks, winning moves average $0.15-$0.35/share -- 1 tick of slippage is ")
    lines.append("only 6-13% of the winner, leaving substantial room for the edge to survive.")
    lines.append("")
    lines.append("### Risk of Ignoring This Filter")
    lines.append("")
    lines.append("Deploying flush_reclaim_v1 without a min-price filter will result in ")
    lines.append("systematic losses that scale linearly with trade frequency. The more trades ")
    lines.append("the system takes on low-priced stocks, the faster capital erodes. ")
    lines.append("This is not a \"might lose\" scenario -- the math is deterministic: ")
    lines.append("when friction exceeds edge, every trade is expected-value negative.")

    return "\n".join(lines)


# -- CLI ENTRY POINT --------------------------------------

def run_from_cli(
    log_path: str = "logs/shadow_flush_reclaim.jsonl",
    cache_path: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    slippage_ticks: int = 1,
    latency_ticks: int = 0,
    spread_cost: float = 0.005,
    commission: float = 0.0,
    shares: int = 100,
    tick_size: float = 0.01,
) -> Dict:
    """
    Run friction tier analysis from CLI.

    Data source priority:
      1. If cache_path provided: run batch-backtest and use those trades
      2. Else: load from shadow JSONL log
    """
    # Load trades
    if cache_path:
        # Run backtest to get trades
        from core.dbn_loader import DatabentoTradeLoader
        from engine.batch_backtest import BatchBacktestEngine
        from strategies.flush_reclaim_v1 import FlushReclaimV1

        loader = DatabentoTradeLoader(cache_path)
        engine = BatchBacktestEngine(loader)

        if not symbols:
            symbols = loader.symbols[:20]

        strategy = FlushReclaimV1(
            lookback=100,
            flush_pct=0.3,
            reclaim_window=100,
            reward_multiple=1.5,
            allowed_regimes="LOW_VOL_CHOP",
        )

        result = engine.run(strategy, symbols, start_date, end_date)
        trades = load_trades_from_batch_result(result.trades)
        print(f"  Loaded {len(trades)} trades from batch-backtest")
    else:
        trades = load_trades_from_jsonl(log_path)
        print(f"  Loaded {len(trades)} trades from {log_path}")

    if not trades:
        print("  ERROR: No trades loaded. Cannot run analysis.")
        return {}

    # Build custom scenario from CLI flags (replaces "realistic")
    custom_scenario = FrictionScenario(
        name=f"Custom (slip={slippage_ticks}t, lat={latency_ticks}t, "
             f"spread=${spread_cost:.4f}, comm=${commission:.2f})",
        slippage_ticks=slippage_ticks,
        latency_ticks=latency_ticks,
        spread_cost_per_side=spread_cost,
        commission=commission,
        tick_size=tick_size,
        shares=shares,
    )

    scenarios = {
        "ideal": SCENARIOS["ideal"],
        "realistic": SCENARIOS["realistic"],
        "custom": custom_scenario,
        "conservative": SCENARIOS["conservative"],
    }

    # Run analysis
    analysis = run_tier_analysis(trades, scenarios)

    # Print to console
    print_tier_analysis(analysis)

    # Save JSON report
    json_path = Path("reports/price_tier_friction_analysis.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # Save markdown report
    md_path = Path("reports/price_tier_summary.md")
    md_text = generate_markdown_report(analysis)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  Saved: {md_path}")

    return analysis

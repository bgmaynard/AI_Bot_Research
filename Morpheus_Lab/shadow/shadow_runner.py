"""
Morpheus Lab — Shadow Trading Runner
========================================
Runs flush_reclaim_v1 alongside Morpheus production without
executing any real orders. Logs hypothetical trades for
performance comparison.

Architecture:
  1. Loads promoted parameters from runtime_config.json
  2. Accumulates tick data per symbol (same feed as Morpheus)
  3. Runs fr1 strategy on accumulated batches at intervals
  4. Tracks hypothetical open positions
  5. Logs entries, exits, and daily summaries to JSONL
  6. Compares against production Morpheus trades

SAFETY:
  - strategy_mode MUST be "SHADOW"
  - No IBKR connection, no order objects, no execution path
  - Read-only access to market data
  - All trades are hypothetical

Usage:
  # From Morpheus_Lab directory
  python -m shadow.shadow_runner

  # Or integrated into Morpheus runtime:
  from shadow.shadow_runner import ShadowRunner
  runner = ShadowRunner.from_config("shadow/runtime_config.json")
  runner.on_tick(symbol, price, size, timestamp_ns)
"""

import json
import logging
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ShadowPosition:
    """Hypothetical open position tracked by shadow runner."""
    symbol: str
    entry_price: float
    entry_ts_ns: int
    stop_price: float
    target_price: float
    share_size: int
    regime: str
    risk_pct: float

    def check_exit(self, price: float, ts_ns: int) -> Optional[dict]:
        """Check if price hits target or stop. Returns exit info or None."""
        if price >= self.target_price:
            return {
                "exit_price": price,
                "exit_ts_ns": ts_ns,
                "exit_reason": "target",
            }
        elif price <= self.stop_price:
            return {
                "exit_price": price,
                "exit_ts_ns": ts_ns,
                "exit_reason": "stop",
            }
        return None


class ShadowRunner:
    """
    Shadow trading engine for flush_reclaim_v1.

    Accumulates ticks per symbol, runs strategy at batch intervals,
    manages hypothetical positions, logs everything.

    NO ORDER EXECUTION. NO CAPITAL RISK.
    """

    def __init__(
        self,
        strategy_params: dict,
        log_path: str = "logs/shadow_flush_reclaim.jsonl",
        summary_path: str = "logs/shadow_daily_summary.jsonl",
        production_log_path: str = "logs/production_trades.jsonl",
        batch_size: int = 500,
    ):
        """
        Args:
            strategy_params: Promoted fr1 parameters
            log_path: JSONL file for individual trades
            summary_path: JSONL file for daily summaries
            production_log_path: Path to Morpheus production log (read-only)
            batch_size: Ticks to accumulate before running strategy
        """
        # Safety check
        self._mode = "SHADOW"

        # Lazy import to avoid circular deps at module level
        from strategies.flush_reclaim_v1 import FlushReclaimV1
        from shadow.shadow_logger import ShadowLogger, ShadowTradeLog

        self._ShadowTradeLog = ShadowTradeLog

        # Initialize strategy with promoted params
        self.strategy = FlushReclaimV1(**strategy_params)
        self.batch_size = batch_size

        # Logger
        self.logger = ShadowLogger(
            trade_log_path=log_path,
            summary_log_path=summary_path,
            production_log_path=production_log_path,
        )

        # Per-symbol tick buffers
        self._buffers: Dict[str, dict] = {}

        # Open shadow positions (max 1 per symbol)
        self._positions: Dict[str, ShadowPosition] = {}

        # Cooldown tracker (symbol -> next allowed tick index)
        self._cooldowns: Dict[str, int] = {}

        # Stats
        self._total_ticks = 0
        self._total_signals = 0
        self._total_trades = 0

        logger.info(
            f"Shadow runner initialized: {self.strategy.name} "
            f"mode={self._mode} batch_size={batch_size}"
        )
        logger.info(f"Params: {strategy_params}")

    @classmethod
    def from_config(cls, config_path: str = "shadow/runtime_config.json") -> "ShadowRunner":
        """Create ShadowRunner from runtime config file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(path) as f:
            config = json.load(f)

        # Safety: verify shadow mode
        if config.get("strategy_mode") != "SHADOW":
            raise ValueError(
                f"SAFETY: strategy_mode must be SHADOW, got '{config.get('strategy_mode')}'. "
                f"Shadow runner refuses to start in non-SHADOW mode."
            )

        if not config.get("enable_shadow_flush_reclaim", False):
            raise ValueError(
                "Shadow flush reclaim is disabled in config. "
                "Set enable_shadow_flush_reclaim=true to enable."
            )

        params = config["promoted_params"]
        log_cfg = config.get("logging", {})
        comp_cfg = config.get("comparison", {})

        return cls(
            strategy_params=params,
            log_path=log_cfg.get("shadow_log_path", "logs/shadow_flush_reclaim.jsonl"),
            summary_path=log_cfg.get("daily_summary_path", "logs/shadow_daily_summary.jsonl"),
            production_log_path=comp_cfg.get("production_log_path", "logs/production_trades.jsonl"),
        )

    def on_tick(self, symbol: str, price: float, size: int, ts_ns: int) -> None:
        """
        Process a single market tick. Call this from Morpheus data feed.

        This is the main integration point. Morpheus calls this for every
        trade event on every symbol it monitors.

        Args:
            symbol: Ticker symbol (e.g., "CISS")
            price: Trade price
            size: Trade volume
            ts_ns: Timestamp in nanoseconds (epoch)
        """
        self._total_ticks += 1

        # Check existing position for exit
        if symbol in self._positions:
            pos = self._positions[symbol]
            exit_info = pos.check_exit(price, ts_ns)
            if exit_info:
                self._close_position(symbol, exit_info)

        # Accumulate tick into buffer
        if symbol not in self._buffers:
            self._buffers[symbol] = {
                "ts": [], "price": [], "size": [],
                "tick_count": 0,
            }

        buf = self._buffers[symbol]
        buf["ts"].append(ts_ns)
        buf["price"].append(price)
        buf["size"].append(size)
        buf["tick_count"] += 1

        # Run strategy when batch is full
        if buf["tick_count"] >= self.batch_size:
            self._run_batch(symbol)

    def on_eod(self, symbol: Optional[str] = None) -> None:
        """
        End-of-day processing. Close open positions, flush summaries.

        Call at market close or end of data feed.
        """
        symbols = [symbol] if symbol else list(self._positions.keys())

        for sym in symbols:
            if sym in self._positions:
                pos = self._positions[sym]
                # Close at last known price
                buf = self._buffers.get(sym)
                if buf and buf["price"]:
                    last_price = buf["price"][-1]
                    last_ts = buf["ts"][-1]
                else:
                    last_price = pos.entry_price
                    last_ts = pos.entry_ts_ns

                self._close_position(sym, {
                    "exit_price": last_price,
                    "exit_ts_ns": last_ts,
                    "exit_reason": "eod",
                })

        # Flush daily summary
        self.logger.flush()

    def shutdown(self) -> None:
        """Clean shutdown: close all positions and flush logs."""
        logger.info("Shadow runner shutting down...")
        self.on_eod()

        logger.info(
            f"Shadow session complete: "
            f"{self._total_ticks} ticks, "
            f"{self._total_signals} signals, "
            f"{self._total_trades} trades"
        )

    def status(self) -> dict:
        """Return current shadow runner status."""
        return {
            "mode": self._mode,
            "strategy": self.strategy.name,
            "total_ticks": self._total_ticks,
            "total_signals": self._total_signals,
            "total_trades": self._total_trades,
            "open_positions": {
                sym: {
                    "entry": pos.entry_price,
                    "stop": pos.stop_price,
                    "target": pos.target_price,
                    "regime": pos.regime,
                }
                for sym, pos in self._positions.items()
            },
            "buffered_symbols": list(self._buffers.keys()),
        }

    def _run_batch(self, symbol: str) -> None:
        """Run strategy on accumulated batch for a symbol."""
        buf = self._buffers[symbol]

        ts = np.array(buf["ts"], dtype=np.int64)
        price = np.array(buf["price"], dtype=np.float64)
        size = np.array(buf["size"], dtype=np.int64)

        # Run strategy
        trades = self.strategy.on_batch(ts, price, size, symbol)

        if trades:
            self._total_signals += len(trades)

            for t in trades:
                # Skip if we already have a position in this symbol
                if symbol in self._positions:
                    continue

                # Calculate structural stop and target from the trade
                risk = t.entry_price - t.exit_price if t.exit_reason == "stop" else 0
                stop_price = t.entry_price - abs(t.entry_price - t.exit_price) if t.exit_reason == "stop" else t.entry_price * (1 - self.strategy.max_risk_pct)

                # Use the strategy's own stop/target from the trade object
                # Reverse-engineer from the trade: the strategy computed structural stop
                # For shadow, we re-derive from entry_price and reward_multiple
                # But the backtest trade already has exit info — in live shadow we track differently

                # Open shadow position
                # We need the flush low for structural stop — extract from entry context
                # Since on_batch returns completed trades, for live shadow we track differently:
                # We open at entry, then monitor for target/stop in on_tick

                self._open_position_from_trade(t)

        # Keep recent ticks for context, trim old ones
        keep = max(self.strategy.lookback + self.strategy.reclaim_window, 500)
        if len(buf["ts"]) > keep * 2:
            buf["ts"] = buf["ts"][-keep:]
            buf["price"] = buf["price"][-keep:]
            buf["size"] = buf["size"][-keep:]
            buf["tick_count"] = len(buf["ts"])
        else:
            buf["tick_count"] = 0  # Reset counter, keep data for context

    def _open_position_from_trade(self, trade) -> None:
        """
        Convert a backtest trade signal into a live shadow position.

        In backtest mode, on_batch() returns completed trades (entry+exit).
        In shadow mode, we extract the entry signal and track it live.
        """
        symbol = trade.symbol
        entry_price = trade.entry_price

        # Reverse-engineer stop from the trade's PnL structure
        # The strategy uses structural stop (flush low), so for a losing trade:
        # exit_price < entry_price, and exit_price = stop_price
        # For shadow, we compute risk from the entry and exit of stop trades

        # For simplicity and accuracy: use the strategy's reward_multiple
        # to compute stop and target from entry price and risk
        # Risk = (entry - flush_low), target = entry + reward_multiple * risk

        # The backtest trade has this info embedded:
        if trade.exit_reason == "stop":
            stop_price = trade.exit_price
        else:
            # Estimate from R:R structure
            if trade.exit_reason == "target":
                target_distance = trade.exit_price - trade.entry_price
                risk = target_distance / self.strategy.reward_multiple
                stop_price = trade.entry_price - risk
            else:
                # EOD — use a default risk estimate
                stop_price = trade.entry_price * (1 - self.strategy.min_risk_pct)

        risk = entry_price - stop_price
        target_price = entry_price + (self.strategy.reward_multiple * risk)

        risk_pct = risk / entry_price if entry_price > 0 else 0

        pos = ShadowPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_ts_ns=trade.entry_ts,
            stop_price=stop_price,
            target_price=target_price,
            share_size=trade.size,
            regime=trade.entry_regime,
            risk_pct=risk_pct,
        )

        self._positions[symbol] = pos
        self._total_trades += 1

        logger.info(
            f"SHADOW OPEN {symbol} @ ${entry_price:.2f} "
            f"stop=${stop_price:.2f} target=${target_price:.2f} "
            f"risk={risk_pct*100:.2f}% [{pos.regime}]"
        )

    def _close_position(self, symbol: str, exit_info: dict) -> None:
        """Close a shadow position and log the trade."""
        if symbol not in self._positions:
            return

        pos = self._positions.pop(symbol)

        exit_price = exit_info["exit_price"]
        exit_ts_ns = exit_info["exit_ts_ns"]
        exit_reason = exit_info["exit_reason"]

        pnl = (exit_price - pos.entry_price) * pos.share_size
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price * 100
                    if pos.entry_price > 0 else 0)

        hold_ns = exit_ts_ns - pos.entry_ts_ns
        hold_seconds = hold_ns / 1e9

        # Compute reward/risk ratio achieved
        risk_distance = pos.entry_price - pos.stop_price
        rr = ((exit_price - pos.entry_price) / risk_distance
              if risk_distance > 0 else 0)

        entry_dt = datetime.fromtimestamp(
            pos.entry_ts_ns / 1e9, tz=timezone.utc
        )
        exit_dt = datetime.fromtimestamp(
            exit_ts_ns / 1e9, tz=timezone.utc
        )

        trade_log = self._ShadowTradeLog(
            timestamp=entry_dt.isoformat(),
            symbol=symbol,
            regime=pos.regime,
            entry_price=round(pos.entry_price, 4),
            stop_price=round(pos.stop_price, 4),
            target_price=round(pos.target_price, 4),
            exit_price=round(exit_price, 4),
            exit_reason=exit_reason,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            hold_seconds=round(hold_seconds, 1),
            risk_pct=round(pos.risk_pct * 100, 2),
            reward_risk=round(rr, 2),
            share_size=pos.share_size,
            exit_timestamp=exit_dt.isoformat(),
        )

        self.logger.log_trade(trade_log)

    # ── REPLAY MODE (for testing with Databento cache) ───

    def replay_from_cache(
        self,
        cache_path: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """
        Run shadow mode on historical Databento cache data.

        Uses full symbol-day batches (identical to batch backtest) to
        ensure trade counts match validated results exactly. Each
        symbol-day is processed as one contiguous array — no overlap,
        no duplicate detection.

        Returns: summary stats dict
        """
        from core.dbn_loader import DatabentoTradeLoader
        from datetime import timezone

        loader = DatabentoTradeLoader(cache_path)

        if symbols is None:
            symbols = loader.symbols[:20]

        print(f"\n{'='*60}")
        print(f"  SHADOW REPLAY — flush_reclaim_v1")
        print(f"{'='*60}")
        print(f"  Cache:    {cache_path}")
        print(f"  Symbols:  {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        print(f"  Start:    {start_date or 'all'}")
        print(f"  End:      {end_date or 'all'}")
        print(f"  Mode:     {self._mode}")
        print(f"{'='*60}\n")

        start_wall = time.perf_counter()
        total_events = 0

        for sym in symbols:
            self.strategy.reset()

            def _on_batch(ts_arr, price_arr, size_arr, symbol):
                nonlocal total_events
                total_events += len(ts_arr)

                # Run strategy on full symbol-day batch (matches backtest exactly)
                trades = self.strategy.on_batch(ts_arr, price_arr, size_arr, symbol)

                for t in trades:
                    self._total_signals += 1
                    self._log_completed_trade(t)

            loader.replay_symbol_batch_callback(
                sym, _on_batch,
                start_date=start_date, end_date=end_date,
            )

        elapsed = time.perf_counter() - start_wall

        # Final flush
        self.logger.flush()

        self._total_ticks = total_events

        stats = self.status()
        stats["elapsed_seconds"] = round(elapsed, 1)
        stats["symbols_processed"] = len(symbols)

        print(f"\n  Shadow replay complete: {elapsed:.1f}s")
        print(f"  Ticks: {total_events:,}")
        print(f"  Signals: {stats['total_signals']}")
        print(f"  Trades: {stats['total_trades']}")

        return stats

    def _log_completed_trade(self, trade) -> None:
        """
        Log a completed trade from on_batch() directly to shadow JSONL.

        Used in replay mode where on_batch() returns full entry-to-exit
        trades. No position tracking needed — the strategy already
        computed entry, exit, stop, target, and PnL.
        """
        entry_price = trade.entry_price
        exit_price = trade.exit_price

        # Reverse-engineer stop/target from trade structure
        if trade.exit_reason == "stop":
            stop_price = trade.exit_price
            risk = entry_price - stop_price
            target_price = entry_price + (self.strategy.reward_multiple * risk)
        elif trade.exit_reason == "target":
            target_distance = exit_price - entry_price
            risk = target_distance / self.strategy.reward_multiple
            stop_price = entry_price - risk
            target_price = exit_price
        else:
            # EOD
            risk = entry_price * self.strategy.min_risk_pct
            stop_price = entry_price - risk
            target_price = entry_price + (self.strategy.reward_multiple * risk)

        risk_pct = risk / entry_price if entry_price > 0 else 0

        pnl = (exit_price - entry_price) * trade.size
        pnl_pct = ((exit_price - entry_price) / entry_price * 100
                    if entry_price > 0 else 0)

        hold_ns = trade.exit_ts - trade.entry_ts
        hold_seconds = hold_ns / 1e9

        rr = ((exit_price - entry_price) / risk if risk > 0 else 0)

        entry_dt = datetime.fromtimestamp(trade.entry_ts / 1e9, tz=timezone.utc)
        exit_dt = datetime.fromtimestamp(trade.exit_ts / 1e9, tz=timezone.utc)

        trade_log = self._ShadowTradeLog(
            timestamp=entry_dt.isoformat(),
            symbol=trade.symbol,
            regime=trade.entry_regime,
            entry_price=round(entry_price, 4),
            stop_price=round(stop_price, 4),
            target_price=round(target_price, 4),
            exit_price=round(exit_price, 4),
            exit_reason=trade.exit_reason,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            hold_seconds=round(hold_seconds, 1),
            risk_pct=round(risk_pct * 100, 2),
            reward_risk=round(rr, 2),
            share_size=trade.size,
            exit_timestamp=exit_dt.isoformat(),
        )

        self.logger.log_trade(trade_log)
        self._total_trades += 1


# ── CLI ENTRY POINT ──────────────────────────────────────

def main():
    """Run shadow replay from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Shadow Trading Runner — flush_reclaim_v1"
    )
    parser.add_argument("--config", default="shadow/runtime_config.json",
                        help="Path to runtime config")
    parser.add_argument("--cache", required=True,
                        help="Databento cache path for replay")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (default: first 20)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    runner = ShadowRunner.from_config(args.config)
    runner.replay_from_cache(
        cache_path=args.cache,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()

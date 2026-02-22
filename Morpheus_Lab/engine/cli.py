"""
Morpheus Lab — CLI Interface
==============================
Command-line entrypoints for the full research pipeline.

Usage:
    python -m engine.cli inspect-databento --cache <path> [--deep]
    python -m engine.cli benchmark-replay --cache <path> --symbols CISS,BOXL --start 2026-02-05 --end 2026-02-05
    python -m engine.cli backtest --cache <path> --mode auto|bars_1s|trades
    python -m engine.cli load      <hypothesis.yaml>
    python -m Morpheus_Lab.cli run       <hypothesis.yaml>
    python -m Morpheus_Lab.cli score     <hypothesis_id>
    python -m Morpheus_Lab.cli promote   <hypothesis_id>
    python -m Morpheus_Lab.cli status    <hypothesis_id>
    python -m Morpheus_Lab.cli shadow    <hypothesis_id>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("morpheus_lab")


# ─────────────────────────────────────────────────────────────
#  DATABENTO COMMANDS
# ─────────────────────────────────────────────────────────────

def cmd_inspect_databento(args: argparse.Namespace) -> None:
    """Inspect Databento cache and produce dataset_profile.json."""
    from datafeeds.databento_inspector import inspect_cache, print_inspection_report

    report_path = args.report or "reports/dataset_profile.json"

    profile = inspect_cache(
        cache_path=args.cache,
        quick=not args.deep,
        report_path=report_path,
    )
    print_inspection_report(profile)
    print(f"  Report saved: {report_path}")


def cmd_benchmark_replay(args: argparse.Namespace) -> None:
    """Benchmark the trade-level replay engine."""
    import os
    import tracemalloc

    from core.dbn_loader import DatabentoTradeLoader
    from core.market_replay import MarketReplayEngine

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.symbols_file:
        sf = Path(args.symbols_file)
        if not sf.exists():
            print(f"Error: symbols file not found: {args.symbols_file}")
            sys.exit(1)
        with open(sf) as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    # Initialize loader
    print(f"\n{'='*64}")
    print(f"  MORPHEUS LAB — REPLAY BENCHMARK")
    print(f"{'='*64}")
    print(f"  Cache:    {args.cache}")

    loader = DatabentoTradeLoader(args.cache)

    if not symbols:
        symbols = loader.symbols
        print(f"  Symbols:  ALL ({len(symbols)} available)")
    else:
        print(f"  Symbols:  {', '.join(symbols)}")

    print(f"  Start:    {args.start or 'all'}")
    print(f"  End:      {args.end or 'all'}")
    single_mode = len(symbols) == 1
    print(f"  Mode:     {'single-symbol fast' if single_mode else 'multi-symbol heap merge'}")
    print(f"{'='*64}")
    print()

    # Track memory
    tracemalloc.start()

    engine = MarketReplayEngine(loader)
    stats = engine.benchmark(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        single_mode=single_mode,
    )

    # Memory snapshot
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Results
    print(f"  {'─'*58}")
    print(f"  Total events:     {stats.total_events:>14,}")
    print(f"  Symbols (data):   {stats.symbols_with_data:>14} / {stats.symbols_requested}")
    print(f"  Elapsed:          {stats.elapsed_seconds:>14.3f} s")
    print(f"  Throughput:       {stats.events_per_second:>14,.0f} evt/s")
    print(f"  Memory current:   {current_mem / 1024 / 1024:>14.1f} MB")
    print(f"  Memory peak:      {peak_mem / 1024 / 1024:>14.1f} MB")
    print(f"  {'─'*58}")

    # Performance assessment
    if stats.total_events > 0:
        if stats.events_per_second >= 1_000_000:
            grade = "EXCELLENT"
        elif stats.events_per_second >= 500_000:
            grade = "GOOD"
        elif stats.events_per_second >= 100_000:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS OPTIMIZATION"
        print(f"  Performance:      {grade}")

    if peak_mem / 1024 / 1024 > 500:
        print(f"  WARNING: Peak memory exceeded 500 MB — investigate streaming")

    print(f"{'='*64}\n")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtest over Databento cache data."""
    from datafeeds.databento_feed import DatabentoFeed
    from core.market_replay import MarketReplay, TradeCollector
    from engine.metrics import compute_metrics

    # Load symbols
    symbols = None
    if args.symbols_file:
        sf = Path(args.symbols_file)
        if not sf.exists():
            print(f"Error: symbols file not found: {args.symbols_file}")
            sys.exit(1)
        with open(sf) as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        print(f"Using {len(symbols)} symbols from --symbols flag")

    # Initialize feed
    print(f"\nInitializing Databento feed: {args.cache}")
    feed = DatabentoFeed(args.cache)

    print(f"Available schemas: {feed.available_schemas}")
    resolved_mode = feed._resolve_mode(args.mode)
    print(f"Mode: {args.mode} -> {resolved_mode}")

    # Initialize replay
    replay = MarketReplay(feed)
    collector = TradeCollector()

    # Event counter callback
    event_count = {"bars": 0, "trades": 0}

    def count_bars(bar):
        event_count["bars"] += 1

    def count_trades(trade):
        event_count["trades"] += 1

    replay.on_bar(count_bars)
    replay.on_trade(count_trades)

    # Run replay
    print(f"\nRunning replay: {args.start} -> {args.end}, mode={resolved_mode}")
    print("-" * 60)

    stats = replay.run(
        symbols=symbols,
        start=args.start,
        end=args.end,
        mode=args.mode,
    )

    print("-" * 60)
    print(f"\n{stats.summary()}")

    # If we have collected trades from a strategy, compute metrics
    if collector.trades:
        metrics = compute_metrics(collector.trades)
        print(f"\nMetrics:")
        for k, v in metrics.to_dict().items():
            print(f"  {k}: {v}")
    else:
        print(
            "\nNo strategy attached - replay ran in observation mode. "
            "Attach a strategy via replay.on_bar() to generate trades."
        )

    print()


# ─────────────────────────────────────────────────────────────
#  HYPOTHESIS COMMANDS
# ─────────────────────────────────────────────────────────────

def cmd_load(args: argparse.Namespace) -> None:
    """Load and validate a hypothesis file."""
    from engine.hypothesis_loader import (
        load_hypothesis,
        generate_grid_combinations,
        generate_execution_variants,
    )

    hyp = load_hypothesis(args.hypothesis_file)
    grid = generate_grid_combinations(hyp.parameter_grid)
    exec_vars = generate_execution_variants(hyp.execution_models)

    print(f"\n{'='*60}")
    print(f"  HYPOTHESIS LOADED: {hyp.hypothesis_id}")
    print(f"{'='*60}")
    print(f"  Description:  {hyp.description.strip()}")
    print(f"  Target:       {hyp.target_system}")
    print(f"  Author:       {hyp.author}")
    print(f"  Window:       {hyp.evaluation_window.start} -> {hyp.evaluation_window.end}")
    print(f"  Grid combos:  {len(grid)}")
    print(f"  Exec variants: {len(exec_vars)}")
    print(f"  Total runs:   {hyp.total_runs()}")
    print(f"{'='*60}\n")

    print("Parameter Grid:")
    for key, values in hyp.parameter_grid.items():
        print(f"  {key}: {values}")

    print(f"\nExecution Models:")
    print(f"  Slippage:  {hyp.execution_models.slippage}")
    print(f"  Latency:   {hyp.execution_models.latency_ms}")

    print(f"\n  Hypothesis is valid and ready for testing.\n")


def cmd_run(args: argparse.Namespace) -> None:
    """Run grid test for a hypothesis."""
    from engine.grid_runner import GridRunner
    from engine.hypothesis_loader import load_hypothesis

    hyp = load_hypothesis(args.hypothesis_file)

    print(f"\nStarting grid test for {hyp.hypothesis_id}")
    print(f"Total runs: {hyp.total_runs()}")
    print(f"Output: results/candidates/{hyp.hypothesis_id}/\n")

    def placeholder_backtest(config, eval_window, seed):
        """PLACEHOLDER - replace with actual backtest logic."""
        print("  WARNING: Using placeholder backtest. Wire your actual backtest function.")
        import random
        random.seed(seed)
        trades = []
        for _ in range(random.randint(20, 50)):
            entry = random.uniform(3.0, 15.0)
            pnl = random.uniform(-0.50, 0.60)
            trades.append({
                "pnl": round(pnl, 4),
                "date": eval_window["start"],
                "entry_price": round(entry, 4),
                "exit_price": round(entry + pnl, 4),
                "direction": "long",
                "shares": 100,
                "relative_volume": random.uniform(0.5, 5.0),
                "tape_speed": random.randint(10, 200),
            })
        return trades

    runner = GridRunner(
        hypothesis=hyp,
        backtest_fn=placeholder_backtest,
        output_dir="results/candidates",
        max_workers=args.workers,
        seed=args.seed,
    )

    results = runner.run(baseline_config_path=args.baseline)

    print(f"\n{'='*60}")
    print(f"  GRID TEST COMPLETE: {hyp.hypothesis_id}")
    print(f"  Total runs: {len(results)}")
    if results:
        best = max(results, key=lambda r: r.overall_metrics.get("expectancy", 0))
        print(f"  Best expectancy: {best.overall_metrics.get('expectancy', 0):.4f}")
        print(f"  Best config: {best.parameters}")
    print(f"{'='*60}\n")


def cmd_score(args: argparse.Namespace) -> None:
    """Score a hypothesis against baseline."""
    from scoring.promotion_score import compute_promotion_score, save_scoring_result

    print(f"\nScoring hypothesis: {args.hypothesis_id}")
    print("Provide normalized 0-1 scores for each dimension:\n")

    try:
        result = compute_promotion_score(
            hypothesis_id=args.hypothesis_id,
            expectancy_improvement=float(input("  Expectancy improvement (0-1): ")),
            stability_improvement=float(input("  Stability improvement (0-1): ")),
            drawdown_improvement=float(input("  Drawdown improvement (0-1): ")),
            regime_consistency=float(input("  Regime consistency (0-1): ")),
            execution_robustness=float(input("  Execution robustness (0-1): ")),
        )

        filepath = save_scoring_result(result, "results")

        print(f"\n{'='*60}")
        print(f"  PROMOTION SCORE: {result.promotion_score:.4f}")
        print(f"  STATUS:          {result.status.upper()}")
        print(f"  Saved to:        {filepath}")
        print(f"{'='*60}\n")

    except (ValueError, EOFError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_promote(args: argparse.Namespace) -> None:
    """Promote a validated hypothesis."""
    from promotion.promote import PromotionPipeline

    pipeline = PromotionPipeline()

    try:
        event = pipeline.promote(
            hypothesis_id=args.hypothesis_id,
            target_system=args.target,
            supervisor_approved=args.approved,
        )

        print(f"\n{'='*60}")
        print(f"  PROMOTED: {args.hypothesis_id}")
        print(f"  Target:   {args.target}")
        print(f"  Score:    {event['promotion_score']}")
        print(f"  Status:   {event['status']}")
        print(f"{'='*60}\n")

    except ValueError as e:
        print(f"\nPromotion BLOCKED: {e}\n")
        sys.exit(1)


def cmd_shadow(args: argparse.Namespace) -> None:
    """Generate shadow validation checklist."""
    from promotion.promote import PromotionPipeline

    pipeline = PromotionPipeline()
    checklist = pipeline.get_shadow_checklist(args.hypothesis_id)

    print(f"\n{'='*60}")
    print(f"  SHADOW VALIDATION CHECKLIST: {args.hypothesis_id}")
    print(f"{'='*60}")
    print(f"  Shadow period: {checklist['shadow_period_days']} days")
    print(f"  Supervisor signoff: {checklist['supervisor_signoff_required']}")
    print(f"\n  Items:")
    for key, item in checklist["checklist"].items():
        print(f"    [ ] {item['description']}")
    print(f"{'='*60}\n")


def cmd_status(args: argparse.Namespace) -> None:
    """Check status of a hypothesis."""
    hyp_id = args.hypothesis_id
    results_base = Path("results")

    for status_dir in ["validated", "candidates", "rejected"]:
        scoring_file = results_base / status_dir / f"{hyp_id}_scoring.json"
        if scoring_file.exists():
            with open(scoring_file) as f:
                data = json.load(f)
            print(f"\n  Hypothesis: {hyp_id}")
            print(f"  Status:     {data.get('status', 'unknown').upper()}")
            print(f"  Score:      {data.get('promotion_score', 'N/A')}")
            print(f"  Location:   {scoring_file}\n")
            return

    run_dir = results_base / "candidates" / hyp_id
    if run_dir.exists():
        completed = run_dir / "_completed.json"
        if completed.exists():
            with open(completed) as f:
                hashes = json.load(f)
            print(f"\n  Hypothesis: {hyp_id}")
            print(f"  Status:     RUNS COMPLETED (not yet scored)")
            print(f"  Runs:       {len(hashes)}")
            print(f"  Location:   {run_dir}\n")
            return

    print(f"\n  Hypothesis {hyp_id}: NOT FOUND\n")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="morpheus-lab",
        description="Morpheus Research & Promotion Framework",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- inspect-databento --
    p_inspect = subparsers.add_parser(
        "inspect-databento",
        help="Inspect Databento cache: schemas, symbols, date range",
    )
    p_inspect.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_inspect.add_argument("--deep", action="store_true", help="Deep scan (reads all records, slow)")
    p_inspect.add_argument("--report", default=None, help="Output report path")

    # -- benchmark-replay --
    p_bench = subparsers.add_parser(
        "benchmark-replay",
        help="Benchmark trade-level replay engine throughput",
    )
    p_bench.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_bench.add_argument("--symbols", default=None, help="Comma-separated symbol list (e.g. CISS,BOXL)")
    p_bench.add_argument("--symbols-file", default=None, help="File with one symbol per line")
    p_bench.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p_bench.add_argument("--end", default=None, help="End date YYYY-MM-DD")

    # -- backtest --
    p_bt = subparsers.add_parser("backtest", help="Run backtest over Databento data")
    p_bt.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_bt.add_argument("--mode", default="auto", choices=["auto", "bars_1s", "trades", "bars_1m"],
                       help="Data mode (default: auto)")
    p_bt.add_argument("--symbols-file", default=None, help="File with one symbol per line")
    p_bt.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p_bt.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p_bt.add_argument("--end", default=None, help="End date YYYY-MM-DD")

    # -- load --
    p_load = subparsers.add_parser("load", help="Load & validate hypothesis")
    p_load.add_argument("hypothesis_file", help="Path to hypothesis YAML file")

    # -- run --
    p_run = subparsers.add_parser("run", help="Run grid test")
    p_run.add_argument("hypothesis_file", help="Path to hypothesis YAML file")
    p_run.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--baseline", default=None, help="Override baseline config path")

    # -- score --
    p_score = subparsers.add_parser("score", help="Score hypothesis")
    p_score.add_argument("hypothesis_id", help="Hypothesis ID")

    # -- promote --
    p_promote = subparsers.add_parser("promote", help="Promote validated hypothesis")
    p_promote.add_argument("hypothesis_id", help="Hypothesis ID")
    p_promote.add_argument("--target", required=True, help="Target system")
    p_promote.add_argument("--approved", action="store_true", help="Supervisor approved")

    # -- shadow --
    p_shadow = subparsers.add_parser("shadow", help="Generate shadow checklist")
    p_shadow.add_argument("hypothesis_id", help="Hypothesis ID")

    # -- status --
    p_status = subparsers.add_parser("status", help="Check hypothesis status")
    p_status.add_argument("hypothesis_id", help="Hypothesis ID")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "inspect-databento": cmd_inspect_databento,
        "benchmark-replay": cmd_benchmark_replay,
        "backtest": cmd_backtest,
        "load": cmd_load,
        "run": cmd_run,
        "score": cmd_score,
        "promote": cmd_promote,
        "shadow": cmd_shadow,
        "status": cmd_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

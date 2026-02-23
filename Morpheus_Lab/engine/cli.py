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
    raw_pass = getattr(args, 'raw_pass', False)
    batch_mode = getattr(args, 'batch', False)
    callback_mode = getattr(args, 'callback_mode', False)
    batch_callback = getattr(args, 'batch_callback', False)

    if raw_pass:
        mode_label = "RAW PASSTHROUGH (count only, no Python objects)"
    elif batch_callback:
        mode_label = "BATCH CALLBACK (numpy arrays to function, no per-event loop)"
    elif callback_mode:
        mode_label = "CALLBACK (zero-object, raw primitives)"
    elif batch_mode:
        mode_label = "BATCH (numpy arrays, no per-event objects)"
    elif single_mode:
        mode_label = "single-symbol fast"
    else:
        mode_label = "multi-symbol heap merge"
    print(f"  Mode:     {mode_label}")
    print(f"{'='*64}")
    print()

    # Track memory
    tracemalloc.start()

    engine = MarketReplayEngine(loader)

    # Optional profiling
    use_profile = getattr(args, 'profile', False)

    if use_profile:
        import cProfile
        import pstats
        import io

        profiler = cProfile.Profile()
        profiler.enable()

    stats = engine.benchmark(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        single_mode=single_mode,
        raw_pass=raw_pass,
        batch_mode=batch_mode,
        callback_mode=callback_mode,
        batch_callback_mode=batch_callback,
    )

    if use_profile:
        profiler.disable()

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
    print(f"  Mode:             {stats.mode:>14}")
    print(f"  {'─'*58}")

    # Performance assessment
    if stats.total_events > 0:
        if stats.events_per_second >= 1_000_000:
            grade = "EXCELLENT (>1M evt/s)"
        elif stats.events_per_second >= 500_000:
            grade = "GREAT (>500k evt/s)"
        elif stats.events_per_second >= 300_000:
            grade = "GOOD (>300k evt/s)"
        elif stats.events_per_second >= 100_000:
            grade = "ACCEPTABLE (>100k evt/s)"
        else:
            grade = "NEEDS OPTIMIZATION"
        print(f"  Performance:      {grade}")

    if peak_mem / 1024 / 1024 > 500:
        print(f"  WARNING: Peak memory exceeded 500 MB")

    print(f"{'='*64}")

    # Profile output
    if use_profile:
        print(f"\n  PROFILE — Top 15 by cumulative time:")
        print(f"  {'─'*58}")
        stream = io.StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(15)
        for line in stream.getvalue().split("\n"):
            if line.strip():
                print(f"  {line}")
        print()

    print()


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


def cmd_shadow_replay(args: argparse.Namespace) -> None:
    """Run shadow flush_reclaim_v1 on historical Databento data."""
    from shadow.shadow_runner import ShadowRunner

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    config_path = getattr(args, 'config', 'shadow/runtime_config.json')

    runner = ShadowRunner.from_config(config_path)
    runner.replay_from_cache(
        cache_path=args.cache,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
    )


def cmd_friction_tier_analysis(args: argparse.Namespace) -> None:
    """Run price tier + friction survivability diagnostic."""
    from analysis.friction_price_tier_analysis import run_from_cli

    # Parse symbols if provided
    symbols = None
    if getattr(args, 'symbols', None):
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    run_from_cli(
        log_path=getattr(args, 'log', 'logs/shadow_flush_reclaim.jsonl'),
        cache_path=getattr(args, 'cache', None),
        symbols=symbols,
        start_date=getattr(args, 'start', None),
        end_date=getattr(args, 'end', None),
        slippage_ticks=getattr(args, 'slippage_ticks', 1),
        latency_ticks=getattr(args, 'latency_ticks', 0),
        spread_cost=getattr(args, 'spread_cost', 0.005),
        commission=getattr(args, 'commission', 0.0),
        shares=getattr(args, 'shares', 100),
        tick_size=getattr(args, 'tick_size', 0.01),
    )


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

def cmd_batch_backtest(args: argparse.Namespace) -> None:
    """Run vectorized batch backtest."""
    import tracemalloc

    from core.dbn_loader import DatabentoTradeLoader
    from engine.batch_backtest import BatchBacktestEngine

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

    # Initialize
    print(f"\n{'='*64}")
    print(f"  MORPHEUS LAB — BATCH BACKTEST")
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

    # Strategy selection
    strat_choice = getattr(args, 'strategy', 'v1')
    use_v2 = strat_choice == 'v2'
    use_fr1 = strat_choice == 'fr1'

    params = {
        'lookback': getattr(args, 'lookback', 200),
        'vol_surge': getattr(args, 'vol_surge', 2.0),
        'breakout_pct': getattr(args, 'breakout_pct', 0.5),
        'target_pct': getattr(args, 'target_pct', 2.0),
        'stop_pct': getattr(args, 'stop_pct', 1.0),
        'cooldown': getattr(args, 'cooldown', 50),
        'share_size': getattr(args, 'share_size', 100),
    }

    if use_fr1:
        from strategies.flush_reclaim_v1 import FlushReclaimV1
        fr_params = {
            'lookback': params['lookback'],
            'flush_pct': getattr(args, 'flush_pct', 0.5),
            'flush_window': getattr(args, 'flush_window', 50),
            'vol_surge': params['vol_surge'],
            'reclaim_window': getattr(args, 'reclaim_window', 100),
            'reward_multiple': getattr(args, 'reward_multiple', 2.0),
            'min_risk_pct': getattr(args, 'min_risk_pct', 0.3),
            'max_risk_pct': getattr(args, 'max_risk_pct', 3.0),
            'cooldown': params['cooldown'],
            'share_size': params['share_size'],
            'regime_window': getattr(args, 'regime_window', 200),
        }
        allowed = getattr(args, 'allowed_regimes', None)
        if allowed:
            fr_params['allowed_regimes'] = allowed
        strategy = FlushReclaimV1(**fr_params)
    elif use_v2:
        from strategies.momentum_continuation_v2 import MomentumContinuationV2
        v2_params = dict(params)
        v2_params['regime_window'] = getattr(args, 'regime_window', 200)
        v2_params['classify_entries'] = True
        allowed = getattr(args, 'allowed_regimes', None)
        if allowed:
            v2_params['allowed_regimes'] = allowed
        strategy = MomentumContinuationV2(**v2_params)
    else:
        from strategies.momentum_breakout import MomentumBreakout
        strategy = MomentumBreakout(**params)

    print(f"  Strategy: {strategy.name}")
    print(f"  Params:   {params}")

    if use_v2 or use_fr1:
        print(f"  Gate:     {strategy.allowed_regimes}")

    regime_breakdown = getattr(args, 'regime_breakdown', False)
    only_regime = getattr(args, 'only_regime', None)
    regime_window = getattr(args, 'regime_window', 200)

    # v2/fr1: strategy gates internally, no external regime filtering needed
    # v1: use engine's external regime classification
    has_internal_gate = use_v2 or use_fr1
    engine_regime_breakdown = regime_breakdown and not has_internal_gate
    engine_only_regime = only_regime if not has_internal_gate else None

    if not has_internal_gate and (regime_breakdown or only_regime):
        print(f"  Regime:   {'breakdown enabled' if regime_breakdown else ''}"
              f"{'filter=' + only_regime if only_regime else ''}"
              f" (window={regime_window})")

    print(f"{'='*64}")
    print()

    # Run
    tracemalloc.start()

    engine = BatchBacktestEngine(loader)

    use_profile = getattr(args, 'profile', False)
    if use_profile:
        import cProfile
        import pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()

    result = engine.run(
        strategy, symbols, args.start, args.end,
        regime_breakdown=engine_regime_breakdown,
        only_regime=engine_only_regime,
        regime_window=regime_window,
    )

    if use_profile:
        profiler.disable()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Results
    print(result.summary())
    print(f"  Memory:   {peak_mem / 1024 / 1024:.1f} MB peak")
    print(f"{'='*64}")

    # Regime breakdown (works for both v1 external tagging and v2 internal tagging)
    show_regime_breakdown = regime_breakdown or ((use_v2 or use_fr1) and result.trades)
    if show_regime_breakdown and result.trades:
        # v2 trades have entry_regime pre-filled; v1 needs external tagging
        has_regime_tags = any(t.entry_regime for t in result.trades)

        if has_regime_tags:
            from engine.batch_backtest import compute_regime_breakdown, save_regime_breakdown

            breakdown = compute_regime_breakdown(result.trades)

            print(f"\n  REGIME BREAKDOWN")
            print(f"  {'─'*62}")
            print(f"  {'Regime':<22} {'Trades':>6} {'WR%':>5} {'PF':>6} {'PnL':>10} {'R:R':>5} {'Sharpe':>6} {'AvgHold':>7}")
            print(f"  {'─'*62}")

            for regime, metrics in sorted(breakdown.items(), key=lambda x: -x[1].get('profit_factor', 0) if isinstance(x[1].get('profit_factor', 0), (int, float)) else 0):
                pf = metrics['profit_factor']
                pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else pf
                print(
                    f"  {regime:<22} {metrics['trades']:>6} "
                    f"{metrics['win_rate']:>4.0f}% "
                    f"{pf_str:>6} "
                    f"${metrics['total_pnl']:>9.2f} "
                    f"{metrics['rr']:>5.2f} "
                    f"{metrics['sharpe']:>6.1f} "
                    f"{metrics['avg_hold_seconds']:>6.0f}s"
                )

            print(f"  {'─'*62}")

            sym_tag = symbols[0] if len(symbols) == 1 else f"{len(symbols)}sym"
            report_path = f"reports/regime_breakdown_{sym_tag}.json"
            save_regime_breakdown(breakdown, report_path, params=params, symbols=symbols)
            print(f"  Saved: {report_path}")

    # Entry type breakdown (v2 only)
    entry_type_breakdown = getattr(args, 'entry_type_breakdown', False)
    if (entry_type_breakdown or use_v2 or use_fr1) and result.trades:
        has_entry_types = any(t.entry_type for t in result.trades)

        if has_entry_types:
            from engine.batch_backtest import compute_regime_breakdown

            # Group by entry_type using same machinery
            from engine.edge_metrics import compute_edge_metrics
            type_groups = {}
            for t in result.trades:
                et = t.entry_type or "unclassified"
                if et not in type_groups:
                    type_groups[et] = []
                type_groups[et].append(t)

            print(f"\n  ENTRY TYPE BREAKDOWN")
            print(f"  {'─'*62}")
            print(f"  {'EntryType':<22} {'Trades':>6} {'WR%':>5} {'PF':>6} {'PnL':>10} {'R:R':>5} {'Sharpe':>6} {'AvgHold':>7}")
            print(f"  {'─'*62}")

            for etype, group in sorted(type_groups.items(), key=lambda x: -len(x[1])):
                m = compute_edge_metrics(group, strategy_name="", params={})
                pf = m.profit_factor
                pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                print(
                    f"  {etype:<22} {m.total_trades:>6} "
                    f"{m.win_rate:>4.0f}% "
                    f"{pf_str:>6} "
                    f"${m.total_pnl:>9.2f} "
                    f"{m.reward_risk:>5.2f} "
                    f"{m.sharpe:>6.1f} "
                    f"{m.avg_hold_seconds:>6.0f}s"
                )

            print(f"  {'─'*62}")

    # Friction stress test (if any friction flags set)
    has_friction = (
        getattr(args, 'slippage_ticks', 0) > 0
        or getattr(args, 'latency_ticks', 0) > 0
        or getattr(args, 'spread_cost', 0.0) > 0
        or getattr(args, 'commission', 0.0) > 0
    )

    if has_friction and result.trades:
        from engine.friction_model import (
            FrictionModel, FrictionConfig,
            compute_friction_comparison, print_friction_comparison,
        )

        friction_cfg = FrictionConfig(
            slippage_ticks=getattr(args, 'slippage_ticks', 0),
            latency_ticks=getattr(args, 'latency_ticks', 0),
            spread_cost=getattr(args, 'spread_cost', 0.0),
            commission=getattr(args, 'commission', 0.0),
            tick_size=getattr(args, 'tick_size', 0.01),
        )

        friction = FrictionModel(friction_cfg)
        adjusted_trades = friction.apply(result.trades)

        comparison = compute_friction_comparison(
            result.trades, adjusted_trades, friction_cfg
        )
        print_friction_comparison(comparison)

        # Save comparison to JSON
        import json as _json
        friction_report_path = "reports/friction_stress_test.json"
        Path(friction_report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(friction_report_path, "w") as f:
            _json.dump(comparison, f, indent=2)
        print(f"  Saved: {friction_report_path}")

    # Trade details
    if result.trades and not getattr(args, 'no_details', False):
        print(f"\n  TRADES ({len(result.trades)}):")
        print(f"  {'─'*72}")
        for i, t in enumerate(result.trades[:50]):  # cap at 50
            win_str = "WIN " if t.won else "LOSS"
            regime_tag = f" [{t.entry_regime}]" if t.entry_regime else ""
            type_tag = f" ({t.entry_type})" if t.entry_type else ""
            print(
                f"  {i+1:>3}. {t.symbol:<6} {win_str} "
                f"${t.entry_price:.2f} → ${t.exit_price:.2f} "
                f"PnL=${t.pnl:>+8.2f} ({t.exit_reason}) "
                f"hold={t.hold_seconds:.1f}s{regime_tag}{type_tag}"
            )
        if len(result.trades) > 50:
            print(f"  ... and {len(result.trades) - 50} more")

    # Profile
    if use_profile:
        print(f"\n  PROFILE — Top 15 by cumulative time:")
        print(f"  {'─'*58}")
        stream = io.StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(15)
        for line in stream.getvalue().split("\n"):
            if line.strip():
                print(f"  {line}")

    print()


def cmd_batch_grid(args: argparse.Namespace) -> None:
    """Run systematic parameter grid search."""
    import tracemalloc

    from core.dbn_loader import DatabentoTradeLoader
    from engine.grid_engine import GridEngine, parse_grid_string, expand_grid
    from engine.report_generator import ReportGenerator

    strat_choice = getattr(args, 'strategy', 'v1')
    if strat_choice == 'fr1':
        from strategies.flush_reclaim_v1 import FlushReclaimV1 as StrategyClass
    elif strat_choice == 'v2':
        from strategies.momentum_continuation_v2 import MomentumContinuationV2 as StrategyClass
    else:
        from strategies.momentum_breakout import MomentumBreakout as StrategyClass

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.symbols_file:
        sf = Path(args.symbols_file)
        with open(sf) as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    # Parse grid
    if not args.grid:
        print("Error: --grid required. Example: --grid 'lookback=50,100 target_pct=2,3'")
        sys.exit(1)

    param_dict = parse_grid_string(args.grid)
    param_grid = expand_grid(param_dict)

    # v2/fr1: inject regime gate params into each combo
    if strat_choice in ('v2', 'fr1'):
        allowed = getattr(args, 'allowed_regimes', None)
        for combo in param_grid:
            if strat_choice == 'v2':
                combo['classify_entries'] = True
            if allowed:
                combo['allowed_regimes'] = allowed

    # Initialize
    print(f"\n{'='*64}")
    print(f"  MORPHEUS LAB — PARAMETER GRID SEARCH")
    print(f"{'='*64}")
    print(f"  Cache:    {args.cache}")

    loader = DatabentoTradeLoader(args.cache)

    if not symbols:
        symbols = loader.symbols[:20]
        print(f"  Symbols:  first {len(symbols)}")
    else:
        print(f"  Symbols:  {', '.join(symbols)}")

    print(f"  Start:    {args.start or 'all'}")
    print(f"  End:      {args.end or 'all'}")
    print(f"  Grid:     {param_dict}")
    print(f"  Combos:   {len(param_grid)}")
    min_trades = getattr(args, 'min_trades', 10)
    print(f"  Min trades: {min_trades}")
    print(f"{'='*64}")
    print()

    tracemalloc.start()

    engine = GridEngine(loader)

    # Progress callback
    import time
    grid_start = time.perf_counter()

    def _progress(idx, total, params, metrics):
        pf = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "inf"
        elapsed = time.perf_counter() - grid_start
        print(
            f"  [{idx:>3}/{total}] {elapsed:>6.1f}s | "
            f"trades={metrics.total_trades:>4} PF={pf:>6} "
            f"PnL=${metrics.total_pnl:>8.2f} | {params}"
        )

    results = engine.run_grid(
        strategy_class=StrategyClass,
        param_grid=param_grid,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        min_trades=min_trades,
        progress_callback=_progress,
    )

    grid_elapsed = time.perf_counter() - grid_start

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Filter and display top results
    qualified = [m for m in results if m.total_trades >= min_trades]
    top_n = min(10, len(qualified))

    print(f"\n{'='*64}")
    print(f"  GRID RESULTS — Top {top_n} by Profit Factor")
    print(f"  ({len(qualified)} combos with ≥{min_trades} trades)")
    print(f"{'='*64}")

    for i, m in enumerate(qualified[:top_n]):
        pf = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "inf"
        print(
            f"  #{i+1:>2}  PF={pf:>6}  R:R={m.reward_risk:.2f}  "
            f"WR={m.win_rate:.0f}%  Trades={m.total_trades:>4}  "
            f"PnL=${m.total_pnl:>9.2f}  DD=${m.max_drawdown:>7.2f}  "
            f"Sharpe={m.sharpe:.1f}  {m.params}"
        )

    # Save CSV
    sym_tag = symbols[0] if len(symbols) == 1 else f"{len(symbols)}sym"
    date_tag = f"{args.start or 'all'}_{args.end or 'all'}"
    csv_path = f"reports/grid_results_{sym_tag}_{date_tag}.csv"
    json_path = f"reports/grid_results_{sym_tag}_{date_tag}.json"

    rows_csv = engine.save_grid_csv(results, csv_path, min_trades=min_trades)
    rows_json = engine.save_grid_json(results, json_path, min_trades=min_trades)

    print(f"\n  Grid time:  {grid_elapsed:.1f}s ({len(param_grid)} combos)")
    print(f"  Memory:     {peak_mem / 1024 / 1024:.1f} MB peak")
    print(f"  CSV saved:  {csv_path} ({rows_csv} rows)")
    print(f"  JSON saved: {json_path} ({rows_json} rows)")
    print(f"{'='*64}\n")


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run walk-forward validation."""
    from core.dbn_loader import DatabentoTradeLoader
    from engine.grid_engine import parse_grid_string, expand_grid
    from engine.walk_forward import WalkForwardAnalyzer
    from core.promotion_gate import qualifies_for_shadow

    strat_choice = getattr(args, 'strategy', 'v1')
    if strat_choice == 'fr1':
        from strategies.flush_reclaim_v1 import FlushReclaimV1 as StrategyClass
    elif strat_choice == 'v2':
        from strategies.momentum_continuation_v2 import MomentumContinuationV2 as StrategyClass
    else:
        from strategies.momentum_breakout import MomentumBreakout as StrategyClass

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    if not args.grid:
        print("Error: --grid required")
        sys.exit(1)

    param_dict = parse_grid_string(args.grid)
    param_grid = expand_grid(param_dict)

    # v2/fr1: inject regime gate params
    if strat_choice in ('v2', 'fr1'):
        allowed = getattr(args, 'allowed_regimes', None)
        for combo in param_grid:
            if strat_choice == 'v2':
                combo['classify_entries'] = True
            if allowed:
                combo['allowed_regimes'] = allowed

    # Initialize
    print(f"\n{'='*64}")
    print(f"  MORPHEUS LAB — WALK-FORWARD VALIDATION")
    print(f"{'='*64}")
    print(f"  Cache:    {args.cache}")

    loader = DatabentoTradeLoader(args.cache)

    if not symbols:
        symbols = loader.symbols[:20]
    print(f"  Symbols:  {', '.join(symbols)}")
    print(f"  Start:    {args.start}")
    print(f"  End:      {args.end}")
    print(f"  Grid:     {param_dict} ({len(param_grid)} combos)")

    train_pct = getattr(args, 'train_pct', 0.7)
    top_n = getattr(args, 'top_n', 5)
    min_trades = getattr(args, 'min_trades', 10)

    print(f"  Train:    {train_pct*100:.0f}%")
    print(f"  Top N:    {top_n}")
    print(f"{'='*64}")
    print()

    import time
    wf_start = time.perf_counter()

    analyzer = WalkForwardAnalyzer(loader)

    def _progress(idx, total, params, metrics):
        elapsed = time.perf_counter() - wf_start
        pf = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "inf"
        print(
            f"  [Train {idx:>3}/{total}] {elapsed:>5.1f}s | "
            f"trades={metrics.total_trades:>4} PF={pf:>6} | {params}"
        )

    result = analyzer.analyze(
        strategy_class=StrategyClass,
        param_grid=param_grid,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        train_pct=train_pct,
        top_n=top_n,
        min_trades=min_trades,
        progress_callback=_progress,
    )

    wf_elapsed = time.perf_counter() - wf_start

    # Display results
    print(f"\n{result.summary()}")

    # Promotion gate check for each candidate
    print(f"\n{'─'*60}")
    print(f"  PROMOTION GATE ASSESSMENT")
    print(f"{'─'*60}")

    for i, (tm, vm) in enumerate(zip(result.train_metrics, result.val_metrics)):
        is_pf = tm.profit_factor
        oos_pf = vm.profit_factor
        stability = (oos_pf / is_pf) if (is_pf > 0 and is_pf != float('inf') and oos_pf != float('inf')) else 0.0

        decision = qualifies_for_shadow(
            metrics=tm,
            oos_metrics=vm,
            stability_ratio=stability,
        )

        status = "QUALIFIES" if decision.qualifies else "REJECTED"
        print(f"\n  Candidate #{i+1}: {status} (score: {decision.score:.0f}/100)")
        print(f"  Params: {tm.params}")
        if decision.rejections:
            for r in decision.rejections:
                print(f"    ✗ {r}")
        if decision.warnings:
            for w in decision.warnings:
                print(f"    ⚠ {w}")

    # Save results
    sym_tag = symbols[0] if len(symbols) == 1 else f"{len(symbols)}sym"
    json_path = f"reports/walkforward_{sym_tag}.json"
    analyzer.save_results(result, json_path)

    print(f"\n  Walk-forward time: {wf_elapsed:.1f}s")
    print(f"  Results saved: {json_path}")
    print(f"{'='*64}\n")


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
    p_bench.add_argument("--raw-pass", action="store_true", dest="raw_pass",
                         help="Raw passthrough: count only, no Python objects (decode speed test)")
    p_bench.add_argument("--callback-mode", action="store_true", dest="callback_mode",
                         help="Zero-object callback: raw primitives to function, no tuples created")
    p_bench.add_argument("--batch-callback", action="store_true", dest="batch_callback",
                         help="Batch callback: numpy arrays passed to function, no per-event loop")
    p_bench.add_argument("--batch", action="store_true",
                         help="Batch mode: yield numpy arrays, no per-event objects")
    p_bench.add_argument("--profile", action="store_true",
                         help="Run cProfile and show top 15 hotspots")

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

    # -- batch-backtest --
    p_bbt = subparsers.add_parser("batch-backtest", help="Vectorized batch backtest (fastest)")
    p_bbt.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_bbt.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p_bbt.add_argument("--symbols-file", default=None, help="File with one symbol per line")
    p_bbt.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p_bbt.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p_bbt.add_argument("--lookback", type=int, default=200, help="Rolling window (trade count)")
    p_bbt.add_argument("--vol-surge", type=float, default=2.0, dest="vol_surge", help="Volume surge multiplier")
    p_bbt.add_argument("--breakout-pct", type=float, default=0.5, dest="breakout_pct", help="Breakout threshold %%")
    p_bbt.add_argument("--target-pct", type=float, default=2.0, dest="target_pct", help="Profit target %%")
    p_bbt.add_argument("--stop-pct", type=float, default=1.0, dest="stop_pct", help="Stop loss %%")
    p_bbt.add_argument("--cooldown", type=int, default=50, help="Min trades between entries")
    p_bbt.add_argument("--share-size", type=int, default=100, dest="share_size", help="Position size")
    p_bbt.add_argument("--no-details", action="store_true", dest="no_details", help="Hide trade details")
    p_bbt.add_argument("--profile", action="store_true", help="Run cProfile")
    p_bbt.add_argument("--regime-breakdown", action="store_true", dest="regime_breakdown",
                         help="Classify regimes and show PF per regime")
    p_bbt.add_argument("--only-regime", default=None, dest="only_regime",
                         help="Only keep trades entering in this regime (e.g. HIGH_VOL_BREAKOUT)")
    p_bbt.add_argument("--regime-window", type=int, default=200, dest="regime_window",
                         help="Lookback window for regime classifier (default 200)")
    p_bbt.add_argument("--strategy", default="v1", choices=["v1", "v2", "fr1"],
                         help="Strategy: v1=momentum_breakout, v2=momentum_continuation_v2, fr1=flush_reclaim_v1 (regime-gated)")
    p_bbt.add_argument("--allowed-regimes", default=None, dest="allowed_regimes",
                         help="v2 only: comma-separated regimes to allow (e.g. TREND_DOWN,TREND_UP,LOW_VOL_CHOP)")
    p_bbt.add_argument("--entry-type-breakdown", action="store_true", dest="entry_type_breakdown",
                         help="v2 only: show PF breakdown by entry type")
    # fr1-specific args
    p_bbt.add_argument("--flush-pct", type=float, default=0.5, dest="flush_pct",
                         help="fr1: Min %% drop below VWAP for flush (default 0.5)")
    p_bbt.add_argument("--flush-window", type=int, default=50, dest="flush_window",
                         help="fr1: Max ticks for flush event (default 50)")
    p_bbt.add_argument("--reclaim-window", type=int, default=100, dest="reclaim_window",
                         help="fr1: Max ticks after flush low to reclaim VWAP (default 100)")
    p_bbt.add_argument("--reward-multiple", type=float, default=2.0, dest="reward_multiple",
                         help="fr1: Target = risk x N (default 2.0)")
    p_bbt.add_argument("--min-risk-pct", type=float, default=0.3, dest="min_risk_pct",
                         help="fr1: Min risk %% to filter micro-stops (default 0.3)")
    p_bbt.add_argument("--max-risk-pct", type=float, default=3.0, dest="max_risk_pct",
                         help="fr1: Max risk %% to filter outsized stops (default 3.0)")
    # Friction model flags (stress testing)
    p_bbt.add_argument("--slippage-ticks", type=int, default=0, dest="slippage_ticks",
                         help="Friction: adverse fill slippage in ticks (default 0)")
    p_bbt.add_argument("--latency-ticks", type=int, default=0, dest="latency_ticks",
                         help="Friction: synthetic latency in ticks (default 0)")
    p_bbt.add_argument("--spread-cost", type=float, default=0.0, dest="spread_cost",
                         help="Friction: half-spread penalty per side in $ (default 0)")
    p_bbt.add_argument("--commission", type=float, default=0.0,
                         help="Friction: flat fee per trade in $ (default 0)")
    p_bbt.add_argument("--tick-size", type=float, default=0.01, dest="tick_size",
                         help="Friction: tick size in $ (default 0.01)")

    # -- batch-grid --
    p_grid = subparsers.add_parser("batch-grid", help="Systematic parameter grid search")
    p_grid.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_grid.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p_grid.add_argument("--symbols-file", default=None, help="File with one symbol per line")
    p_grid.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p_grid.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p_grid.add_argument("--grid", required=True,
                         help="Parameter grid: 'lookback=50,100,200 target_pct=2,3,4 stop_pct=0.5,1.0'")
    p_grid.add_argument("--min-trades", type=int, default=10, dest="min_trades",
                         help="Minimum trades to include in results (default 10)")
    p_grid.add_argument("--strategy", default="v1", choices=["v1", "v2", "fr1"],
                         help="Strategy: v1=momentum_breakout, v2=momentum_continuation_v2, fr1=flush_reclaim_v1")
    p_grid.add_argument("--allowed-regimes", default=None, dest="allowed_regimes",
                         help="v2 only: comma-separated regimes to allow")

    # -- walk-forward --
    p_wf = subparsers.add_parser("walk-forward", help="Walk-forward edge validation")
    p_wf.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_wf.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p_wf.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p_wf.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p_wf.add_argument("--grid", required=True,
                         help="Parameter grid: 'lookback=50,100,200 target_pct=2,3,4'")
    p_wf.add_argument("--train-pct", type=float, default=0.7, dest="train_pct",
                         help="Training period percentage (default 0.7)")
    p_wf.add_argument("--top-n", type=int, default=5, dest="top_n",
                         help="Top N candidates to validate OOS (default 5)")
    p_wf.add_argument("--min-trades", type=int, default=10, dest="min_trades",
                         help="Minimum trades per candidate (default 10)")
    p_wf.add_argument("--strategy", default="v1", choices=["v1", "v2", "fr1"],
                         help="Strategy: v1=momentum_breakout, v2=momentum_continuation_v2, fr1=flush_reclaim_v1")
    p_wf.add_argument("--allowed-regimes", default=None, dest="allowed_regimes",
                         help="v2 only: comma-separated regimes to allow")

    # -- shadow-replay --
    p_shadow = subparsers.add_parser("shadow-replay",
        help="Run shadow flush_reclaim_v1 on historical data (no execution)")
    p_shadow.add_argument("--cache", required=True, help="Path to Databento cache root")
    p_shadow.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p_shadow.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p_shadow.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p_shadow.add_argument("--config", default="shadow/runtime_config.json",
                           help="Shadow runtime config path")

    # -- friction-tier-analysis --
    p_fta = subparsers.add_parser("friction-tier-analysis",
        help="Price tier + friction survivability diagnostic (capital protection gate)")
    p_fta.add_argument("--log", default="logs/shadow_flush_reclaim.jsonl",
                         help="Path to shadow trade JSONL log")
    p_fta.add_argument("--cache", default=None,
                         help="Databento cache path (runs batch-backtest instead of reading JSONL)")
    p_fta.add_argument("--symbols", default=None,
                         help="Comma-separated symbol list (only with --cache)")
    p_fta.add_argument("--start", default=None, help="Start date YYYY-MM-DD (only with --cache)")
    p_fta.add_argument("--end", default=None, help="End date YYYY-MM-DD (only with --cache)")
    p_fta.add_argument("--slippage-ticks", type=int, default=1, dest="slippage_ticks",
                         help="Slippage ticks per side (default 1)")
    p_fta.add_argument("--latency-ticks", type=int, default=0, dest="latency_ticks",
                         help="Latency ticks (default 0)")
    p_fta.add_argument("--spread-cost", type=float, default=0.005, dest="spread_cost",
                         help="Half-spread cost per side (default 0.005)")
    p_fta.add_argument("--commission", type=float, default=0.0,
                         help="Flat commission per trade (default 0)")
    p_fta.add_argument("--shares", type=int, default=100,
                         help="Share size (default 100)")
    p_fta.add_argument("--tick-size", type=float, default=0.01, dest="tick_size",
                         help="Tick size in $ (default 0.01)")

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
        "batch-backtest": cmd_batch_backtest,
        "batch-grid": cmd_batch_grid,
        "walk-forward": cmd_walk_forward,
        "load": cmd_load,
        "run": cmd_run,
        "score": cmd_score,
        "promote": cmd_promote,
        "shadow": cmd_shadow,
        "shadow-replay": cmd_shadow_replay,
        "friction-tier-analysis": cmd_friction_tier_analysis,
        "status": cmd_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

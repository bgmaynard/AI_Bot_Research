"""
Morpheus Lab — CLI Interface
==============================
Command-line entrypoints for hypothesis testing, scoring, and promotion.

Usage:
    python -m engine.cli load    <hypothesis.yaml>       # Load & validate hypothesis
    python -m engine.cli run     <hypothesis.yaml>       # Run full grid test
    python -m engine.cli score   <hypothesis_id>         # Score against baseline
    python -m engine.cli promote <hypothesis_id>         # Promote validated hypothesis
    python -m engine.cli status  <hypothesis_id>         # Check hypothesis status
    python -m engine.cli shadow  <hypothesis_id>         # Generate shadow checklist
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from engine.hypothesis_loader import (
    load_hypothesis,
    generate_grid_combinations,
    generate_execution_variants,
)

logger = logging.getLogger("morpheus_lab")


def cmd_load(args: argparse.Namespace) -> None:
    """Load and validate a hypothesis file."""
    hyp = load_hypothesis(args.hypothesis_file)
    grid = generate_grid_combinations(hyp.parameter_grid)
    exec_vars = generate_execution_variants(hyp.execution_models)

    print(f"\n{'='*60}")
    print(f"  HYPOTHESIS LOADED: {hyp.hypothesis_id}")
    print(f"{'='*60}")
    print(f"  Description:  {hyp.description.strip()}")
    print(f"  Target:       {hyp.target_system}")
    print(f"  Author:       {hyp.author}")
    print(f"  Window:       {hyp.evaluation_window.start} → {hyp.evaluation_window.end}")
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

    print(f"\n✓ Hypothesis is valid and ready for testing.\n")


def cmd_run(args: argparse.Namespace) -> None:
    """Run grid test for a hypothesis."""
    from engine.grid_runner import GridRunner

    hyp = load_hypothesis(args.hypothesis_file)

    print(f"\nStarting grid test for {hyp.hypothesis_id}")
    print(f"Total runs: {hyp.total_runs()}")
    print(f"Output: results/candidates/{hyp.hypothesis_id}/\n")

    # NOTE: You must provide your own backtest function.
    # This is a placeholder that demonstrates the interface.
    def placeholder_backtest(config, eval_window, seed):
        """
        Replace this with your actual backtest function.

        Must accept:
            config: dict - merged config parameters
            eval_window: dict with 'start' and 'end' date strings
            seed: int - for deterministic reproducibility

        Must return:
            List of trade dicts, each with at minimum:
                - pnl: float
                - date: str (YYYY-MM-DD)
                - entry_price: float
                - exit_price: float
                - direction: str ("long" or "short")
                - shares: int
        """
        print(
            "  WARNING: Using placeholder backtest. "
            "Integrate your actual backtest function in engine/cli.py"
        )
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
    print("NOTE: Provide normalized 0-1 scores for each dimension.\n")

    # In production, these would be computed from baseline_comparator
    # For now, prompt for manual input or load from files
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
    print(f"  Supervisor signoff required: {checklist['supervisor_signoff_required']}")
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

    # Check run results
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


def main():
    parser = argparse.ArgumentParser(
        prog="morpheus-lab",
        description="Morpheus Research & Promotion Framework CLI",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # load
    p_load = subparsers.add_parser("load", help="Load & validate hypothesis")
    p_load.add_argument("hypothesis_file", help="Path to hypothesis YAML file")

    # run
    p_run = subparsers.add_parser("run", help="Run grid test")
    p_run.add_argument("hypothesis_file", help="Path to hypothesis YAML file")
    p_run.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_run.add_argument("--baseline", default=None, help="Override baseline config path")

    # score
    p_score = subparsers.add_parser("score", help="Score hypothesis")
    p_score.add_argument("hypothesis_id", help="Hypothesis ID")

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote validated hypothesis")
    p_promote.add_argument("hypothesis_id", help="Hypothesis ID")
    p_promote.add_argument("--target", required=True, help="Target system")
    p_promote.add_argument("--approved", action="store_true", help="Supervisor approved")

    # shadow
    p_shadow = subparsers.add_parser("shadow", help="Generate shadow checklist")
    p_shadow.add_argument("hypothesis_id", help="Hypothesis ID")

    # status
    p_status = subparsers.add_parser("status", help="Check hypothesis status")
    p_status.add_argument("hypothesis_id", help="Hypothesis ID")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
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

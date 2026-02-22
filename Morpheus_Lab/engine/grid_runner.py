"""
Morpheus Lab — Grid Test Engine
=================================
Expands parameter combinations, runs backtests per config,
stores per-run and regime-segmented metrics.
Supports multiprocessing, resume capability, and deterministic seeding.
"""

import json
import hashlib
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.hypothesis_loader import (
    Hypothesis,
    generate_grid_combinations,
    generate_execution_variants,
    merge_config,
    load_baseline_config,
)
from engine.metrics import compute_metrics, MetricsResult
from engine.regime_segmenter import segment_trades_by_regime
from execution_models.slippage_model import apply_execution_model

logger = logging.getLogger(__name__)


def _config_hash(config: Dict[str, Any], execution: Dict[str, Any]) -> str:
    """Generate deterministic hash for a config + execution variant combo."""
    combined = json.dumps({"config": config, "execution": execution}, sort_keys=True)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


@dataclass
class RunResult:
    """Result from a single backtest run."""
    run_id: str
    config_hash: str
    parameters: Dict[str, Any]
    execution_variant: Dict[str, Any]
    overall_metrics: Dict[str, Any]
    regime_metrics: Dict[str, Dict[str, Any]]
    trade_count: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GridRunner:
    """
    Grid test engine that expands parameter combinations,
    runs backtests, and stores structured results.
    """

    def __init__(
        self,
        hypothesis: Hypothesis,
        backtest_fn: Callable,
        output_dir: str = "results/candidates",
        max_workers: int = 4,
        seed: int = 42,
    ):
        """
        Args:
            hypothesis: Loaded Hypothesis object.
            backtest_fn: Callable that accepts (config_dict, eval_window, seed)
                         and returns a list of trade dicts.
            output_dir: Directory to store run results.
            max_workers: Number of parallel workers.
            seed: Base seed for deterministic reproducibility.
        """
        self.hypothesis = hypothesis
        self.backtest_fn = backtest_fn
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.base_seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = self.output_dir / hypothesis.hypothesis_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._completed_hashes: set = set()
        self._load_completed()

    def _load_completed(self) -> None:
        """Load previously completed run hashes for resume capability."""
        completed_file = self._run_dir / "_completed.json"
        if completed_file.exists():
            with open(completed_file) as f:
                self._completed_hashes = set(json.load(f))
            logger.info(f"Resuming: {len(self._completed_hashes)} runs already completed")

    def _save_completed(self) -> None:
        """Persist completed run hashes."""
        completed_file = self._run_dir / "_completed.json"
        with open(completed_file, "w") as f:
            json.dump(list(self._completed_hashes), f)

    def _generate_all_runs(self) -> List[Tuple[int, Dict[str, Any], Dict[str, Any], str]]:
        """
        Generate all (index, config_override, execution_variant, config_hash) tuples.
        Skips already-completed runs.
        """
        grid_combos = generate_grid_combinations(self.hypothesis.parameter_grid)
        exec_variants = generate_execution_variants(self.hypothesis.execution_models)

        runs = []
        idx = 0
        for combo in grid_combos:
            for variant in exec_variants:
                ch = _config_hash(combo, variant)
                if ch not in self._completed_hashes:
                    runs.append((idx, combo, variant, ch))
                idx += 1

        return runs

    def _execute_single_run(
        self,
        run_idx: int,
        config_overrides: Dict[str, Any],
        execution_variant: Dict[str, Any],
        config_hash: str,
        baseline_config: Dict[str, Any],
    ) -> RunResult:
        """Execute a single backtest run."""
        start_time = time.time()

        # Merge config
        test_config = merge_config(baseline_config, config_overrides)

        # Deterministic seed per run
        run_seed = self.base_seed + run_idx

        # Run backtest
        trades = self.backtest_fn(
            test_config,
            {
                "start": self.hypothesis.evaluation_window.start,
                "end": self.hypothesis.evaluation_window.end,
            },
            run_seed,
        )

        # Apply execution model (slippage + latency)
        trades = apply_execution_model(
            trades,
            slippage=execution_variant["slippage"],
            latency_ms=execution_variant["latency_ms"],
        )

        # Compute overall metrics
        overall = compute_metrics(trades)

        # Segment by regime and compute per-regime metrics
        regime_trades = segment_trades_by_regime(trades)
        regime_metrics = {}
        for regime_name, regime_trade_list in regime_trades.items():
            if len(regime_trade_list) > 0:
                regime_metrics[regime_name] = compute_metrics(regime_trade_list).to_dict()

        elapsed = time.time() - start_time

        run_id = f"{self.hypothesis.hypothesis_id}_run{run_idx:04d}"

        return RunResult(
            run_id=run_id,
            config_hash=config_hash,
            parameters=config_overrides,
            execution_variant=execution_variant,
            overall_metrics=overall.to_dict(),
            regime_metrics=regime_metrics,
            trade_count=len(trades),
            elapsed_seconds=round(elapsed, 2),
        )

    def _save_run_result(self, result: RunResult) -> None:
        """Persist individual run result."""
        result_file = self._run_dir / f"{result.run_id}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self._completed_hashes.add(result.config_hash)
        self._save_completed()

    def run(self, baseline_config_path: Optional[str] = None) -> List[RunResult]:
        """
        Execute all grid test runs.

        Args:
            baseline_config_path: Override for baseline config path.
                                  Defaults to hypothesis.baseline_config.

        Returns:
            List of RunResult objects.
        """
        config_path = baseline_config_path or self.hypothesis.baseline_config
        baseline_config = load_baseline_config(config_path)

        all_runs = self._generate_all_runs()
        total = self.hypothesis.total_runs()
        remaining = len(all_runs)

        logger.info(
            f"Grid Runner: {total} total runs, "
            f"{total - remaining} completed, "
            f"{remaining} remaining"
        )

        if remaining == 0:
            logger.info("All runs already completed. Loading results.")
            return self._load_all_results()

        results = []

        if self.max_workers <= 1:
            # Sequential execution
            for i, (run_idx, combo, variant, ch) in enumerate(all_runs):
                logger.info(f"Run {i+1}/{remaining}: {combo} | {variant}")
                result = self._execute_single_run(
                    run_idx, combo, variant, ch, baseline_config
                )
                self._save_run_result(result)
                results.append(result)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for run_idx, combo, variant, ch in all_runs:
                    future = executor.submit(
                        self._execute_single_run,
                        run_idx, combo, variant, ch, baseline_config,
                    )
                    futures[future] = (run_idx, ch)

                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self._save_run_result(result)
                    results.append(result)
                    logger.info(
                        f"Completed {i+1}/{remaining}: {result.run_id} "
                        f"({result.elapsed_seconds}s)"
                    )

        # Save summary
        self._save_summary(results)
        return results

    def _load_all_results(self) -> List[RunResult]:
        """Load all completed run results from disk."""
        results = []
        for f in sorted(self._run_dir.glob("H-*.json")):
            with open(f) as fh:
                data = json.load(fh)
                results.append(RunResult(**data))
        return results

    def _save_summary(self, results: List[RunResult]) -> None:
        """Save a summary of all runs."""
        summary = {
            "hypothesis_id": self.hypothesis.hypothesis_id,
            "total_runs": len(results),
            "target_system": self.hypothesis.target_system,
            "best_expectancy": None,
            "best_config": None,
            "all_runs": [r.run_id for r in results],
        }

        # Find best by expectancy
        best = None
        for r in results:
            exp = r.overall_metrics.get("expectancy", float("-inf"))
            if best is None or exp > best.overall_metrics.get("expectancy", float("-inf")):
                best = r

        if best:
            summary["best_expectancy"] = best.overall_metrics.get("expectancy")
            summary["best_config"] = best.parameters

        summary_file = self._run_dir / "_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    print("GridRunner is a library module. Use via CLI: python -m engine.cli run <hypothesis.yaml>")

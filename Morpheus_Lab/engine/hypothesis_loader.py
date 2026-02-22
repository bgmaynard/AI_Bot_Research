"""
Morpheus Lab — Hypothesis Loader
=================================
Loads YAML hypothesis definitions, validates required fields,
and generates parameter grid combinations for testing.
"""

import yaml
import json
import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationWindow:
    """Date range for backtesting."""
    start: str
    end: str


@dataclass
class ExecutionModels:
    """Execution sensitivity parameters."""
    slippage: List[float]
    latency_ms: List[int]


@dataclass
class Hypothesis:
    """Complete hypothesis definition."""
    hypothesis_id: str
    description: str
    author: str
    created: str
    baseline_config: str
    parameter_grid: Dict[str, List[Any]]
    evaluation_window: EvaluationWindow
    execution_models: ExecutionModels
    target_system: str
    notes: Optional[str] = None

    def total_combinations(self) -> int:
        """Total number of parameter grid combinations."""
        if not self.parameter_grid:
            return 0
        counts = [len(v) for v in self.parameter_grid.values()]
        result = 1
        for c in counts:
            result *= c
        return result

    def total_execution_variants(self) -> int:
        """Total execution model combinations."""
        return len(self.execution_models.slippage) * len(self.execution_models.latency_ms)

    def total_runs(self) -> int:
        """Total backtests required (grid × execution variants)."""
        return self.total_combinations() * self.total_execution_variants()


REQUIRED_FIELDS = [
    "hypothesis_id", "description", "author", "created",
    "baseline_config", "parameter_grid", "evaluation_window",
    "execution_models", "target_system"
]

VALID_TARGET_SYSTEMS = ["morpheus_ai", "ibkr_morpheus", "max_ai"]


def validate_hypothesis(data: Dict[str, Any]) -> List[str]:
    """
    Validate hypothesis data. Returns list of error messages (empty = valid).
    """
    errors = []

    for field_name in REQUIRED_FIELDS:
        if field_name not in data or data[field_name] is None:
            errors.append(f"Missing required field: {field_name}")

    if "target_system" in data and data["target_system"] not in VALID_TARGET_SYSTEMS:
        errors.append(
            f"Invalid target_system: {data['target_system']}. "
            f"Must be one of: {VALID_TARGET_SYSTEMS}"
        )

    if "parameter_grid" in data and isinstance(data["parameter_grid"], dict):
        for key, values in data["parameter_grid"].items():
            if not isinstance(values, list) or len(values) == 0:
                errors.append(f"Parameter grid '{key}' must be a non-empty list")

    if "evaluation_window" in data and isinstance(data["evaluation_window"], dict):
        for field_name in ["start", "end"]:
            if field_name not in data["evaluation_window"]:
                errors.append(f"evaluation_window missing '{field_name}'")

    if "execution_models" in data and isinstance(data["execution_models"], dict):
        for field_name in ["slippage", "latency_ms"]:
            if field_name not in data["execution_models"]:
                errors.append(f"execution_models missing '{field_name}'")

    return errors


def load_hypothesis(filepath: str) -> Hypothesis:
    """
    Load and validate a hypothesis from a YAML file.

    Args:
        filepath: Path to the hypothesis YAML file.

    Returns:
        Validated Hypothesis object.

    Raises:
        FileNotFoundError: If hypothesis file doesn't exist.
        ValueError: If hypothesis fails validation.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis file not found: {filepath}")

    logger.info(f"Loading hypothesis from: {filepath}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    errors = validate_hypothesis(data)
    if errors:
        error_msg = f"Hypothesis validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    hypothesis = Hypothesis(
        hypothesis_id=data["hypothesis_id"],
        description=data["description"],
        author=data["author"],
        created=str(data["created"]),
        baseline_config=data["baseline_config"],
        parameter_grid=data["parameter_grid"],
        evaluation_window=EvaluationWindow(**data["evaluation_window"]),
        execution_models=ExecutionModels(**data["execution_models"]),
        target_system=data["target_system"],
        notes=data.get("notes"),
    )

    logger.info(
        f"Loaded hypothesis {hypothesis.hypothesis_id}: "
        f"{hypothesis.total_combinations()} grid combos × "
        f"{hypothesis.total_execution_variants()} execution variants = "
        f"{hypothesis.total_runs()} total runs"
    )

    return hypothesis


def generate_grid_combinations(parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand parameter grid into all combinations.

    Args:
        parameter_grid: Dict mapping parameter names to lists of values.

    Returns:
        List of dicts, each representing one combination.

    Example:
        Input:  {"a": [1, 2], "b": ["x", "y"]}
        Output: [{"a": 1, "b": "x"}, {"a": 1, "b": "y"},
                 {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]
    """
    keys = list(parameter_grid.keys())
    values = list(parameter_grid.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    logger.info(f"Generated {len(combinations)} parameter combinations")
    return combinations


def generate_execution_variants(execution_models: ExecutionModels) -> List[Dict[str, Any]]:
    """
    Generate all execution model combinations.

    Returns:
        List of dicts with slippage and latency_ms values.
    """
    variants = []
    for slip, lat in itertools.product(execution_models.slippage, execution_models.latency_ms):
        variants.append({"slippage": slip, "latency_ms": lat})

    logger.info(f"Generated {len(variants)} execution variants")
    return variants


def merge_config(baseline: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge parameter overrides into a baseline config (shallow merge).

    Args:
        baseline: The baseline runtime config dict.
        overrides: Parameter overrides from grid combination.

    Returns:
        New merged config dict (baseline is not mutated).
    """
    merged = baseline.copy()
    merged.update(overrides)
    return merged


def load_baseline_config(filepath: str) -> Dict[str, Any]:
    """
    Load baseline runtime config from JSON.

    Args:
        filepath: Path to baseline config JSON file.

    Returns:
        Config dict.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Baseline config not found: {filepath}")

    with open(path, "r") as f:
        config = json.load(f)

    logger.info(f"Loaded baseline config from: {filepath}")
    return config


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python hypothesis_loader.py <path_to_hypothesis.yaml>")
        sys.exit(1)

    hyp = load_hypothesis(sys.argv[1])
    print(f"\nHypothesis: {hyp.hypothesis_id}")
    print(f"Description: {hyp.description}")
    print(f"Target: {hyp.target_system}")
    print(f"Total runs: {hyp.total_runs()}")

    combos = generate_grid_combinations(hyp.parameter_grid)
    print(f"\nFirst 3 grid combinations:")
    for c in combos[:3]:
        print(f"  {c}")

    exec_vars = generate_execution_variants(hyp.execution_models)
    print(f"\nFirst 3 execution variants:")
    for v in exec_vars[:3]:
        print(f"  {v}")

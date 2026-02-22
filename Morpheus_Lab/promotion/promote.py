"""
Morpheus Lab — Promotion Pipeline
====================================
Validates scoring thresholds, copies configs into runtime candidate folders,
logs promotion events, and attaches hypothesis IDs.

Runtime configs must never be edited manually.
"""

import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PromotionPipeline:
    """
    Manages the promotion of validated hypotheses from lab to runtime.
    """

    def __init__(
        self,
        results_dir: str = "results",
        promotion_dir: str = "promotion",
        promotion_log: str = "promotion/promotion_log.json",
    ):
        self.results_dir = Path(results_dir)
        self.promotion_dir = Path(promotion_dir)
        self.promotion_log_path = Path(promotion_log)

        self.promotion_dir.mkdir(parents=True, exist_ok=True)
        self._log = self._load_log()

    def _load_log(self) -> list:
        """Load existing promotion log."""
        if self.promotion_log_path.exists():
            with open(self.promotion_log_path) as f:
                return json.load(f)
        return []

    def _save_log(self) -> None:
        """Persist promotion log."""
        with open(self.promotion_log_path, "w") as f:
            json.dump(self._log, f, indent=2)

    def promote(
        self,
        hypothesis_id: str,
        target_system: str,
        runtime_candidate_dir: Optional[str] = None,
        supervisor_approved: bool = False,
    ) -> Dict[str, Any]:
        """
        Promote a validated hypothesis to runtime candidate status.

        Args:
            hypothesis_id: ID of the validated hypothesis.
            target_system: Which bot this promotes to
                          (morpheus_ai, ibkr_morpheus, max_ai).
            runtime_candidate_dir: Path to copy validated config into.
                                   If None, copies to promotion/<hypothesis_id>/
            supervisor_approved: Whether supervisor has signed off.

        Returns:
            Promotion event dict.

        Raises:
            ValueError: If hypothesis is not validated or supervisor hasn't approved.
        """
        # Verify hypothesis is validated
        validated_dir = self.results_dir / "validated"
        scoring_file = validated_dir / f"{hypothesis_id}_scoring.json"

        if not scoring_file.exists():
            raise ValueError(
                f"Hypothesis {hypothesis_id} not found in validated results. "
                f"Only validated hypotheses can be promoted."
            )

        with open(scoring_file) as f:
            scoring = json.load(f)

        if scoring.get("status") != "validated":
            raise ValueError(
                f"Hypothesis {hypothesis_id} has status '{scoring.get('status')}', "
                f"not 'validated'. Cannot promote."
            )

        if not supervisor_approved:
            raise ValueError(
                "Supervisor approval is required for promotion. "
                "Set supervisor_approved=True after manual review."
            )

        # Create promotion package
        package_dir = self.promotion_dir / hypothesis_id
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy all artifacts
        artifacts_copied = []
        for artifact_file in validated_dir.glob(f"{hypothesis_id}*"):
            dest = package_dir / artifact_file.name
            shutil.copy2(artifact_file, dest)
            artifacts_copied.append(str(dest))

        # Copy to runtime candidate dir if specified
        if runtime_candidate_dir:
            runtime_dir = Path(runtime_candidate_dir)
            runtime_dir.mkdir(parents=True, exist_ok=True)

            config_src = validated_dir / f"{hypothesis_id}_config.json"
            if config_src.exists():
                runtime_dest = runtime_dir / f"{hypothesis_id}_config.json"
                shutil.copy2(config_src, runtime_dest)
                artifacts_copied.append(str(runtime_dest))
                logger.info(f"Config copied to runtime: {runtime_dest}")

        # Log promotion event
        event = {
            "hypothesis_id": hypothesis_id,
            "target_system": target_system,
            "promotion_score": scoring.get("promotion_score"),
            "promoted_at": datetime.now().isoformat(),
            "supervisor_approved": supervisor_approved,
            "artifacts": artifacts_copied,
            "status": "promoted_to_shadow",
        }

        self._log.append(event)
        self._save_log()

        logger.info(
            f"PROMOTED: {hypothesis_id} → {target_system} "
            f"(score: {scoring.get('promotion_score')})"
        )

        return event

    def get_promotion_history(self) -> list:
        """Return full promotion log."""
        return self._log.copy()

    def get_shadow_checklist(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Generate shadow validation checklist for a promoted hypothesis.

        Shadow promotion requires manual supervisor signoff.
        """
        return {
            "hypothesis_id": hypothesis_id,
            "checklist": {
                "performance_deviation_under_5pct": {
                    "description": "Shadow P&L within 5% of lab projection",
                    "status": "pending",
                    "verified_by": None,
                    "verified_at": None,
                },
                "disable_events_not_increased": {
                    "description": "No increase in gate disables or circuit breakers",
                    "status": "pending",
                    "verified_by": None,
                    "verified_at": None,
                },
                "no_flipflop_instability": {
                    "description": "No rapid signal flip-flopping or oscillation",
                    "status": "pending",
                    "verified_by": None,
                    "verified_at": None,
                },
                "trade_frequency_within_tolerance": {
                    "description": "Trade frequency within ±15% of baseline",
                    "status": "pending",
                    "verified_by": None,
                    "verified_at": None,
                },
            },
            "shadow_period_days": "3-5",
            "supervisor_signoff_required": True,
            "created_at": datetime.now().isoformat(),
        }

    def complete_shadow_validation(
        self,
        hypothesis_id: str,
        checklist_results: Dict[str, bool],
        supervisor: str,
    ) -> Dict[str, Any]:
        """
        Complete shadow validation and approve for live deployment.

        Args:
            hypothesis_id: Hypothesis being validated.
            checklist_results: Dict mapping checklist item keys to pass/fail.
            supervisor: Name of approving supervisor.

        Returns:
            Final deployment event dict.

        Raises:
            ValueError: If any checklist item failed.
        """
        failures = [k for k, v in checklist_results.items() if not v]
        if failures:
            raise ValueError(
                f"Shadow validation FAILED for {hypothesis_id}. "
                f"Failed items: {failures}"
            )

        event = {
            "hypothesis_id": hypothesis_id,
            "status": "live_deployment_approved",
            "shadow_validation_passed": True,
            "checklist_results": checklist_results,
            "approved_by": supervisor,
            "approved_at": datetime.now().isoformat(),
        }

        self._log.append(event)
        self._save_log()

        logger.info(
            f"LIVE DEPLOYMENT APPROVED: {hypothesis_id} "
            f"by {supervisor}"
        )

        return event

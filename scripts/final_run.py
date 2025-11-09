"""Aggregate runner for DGCE API endpoints.

Executing this script will:
  • Run the enhanced quick simulation
  • Run the comprehensive analysis
  • Run a sector-focused analysis (Manufacturing by default)
  • Evaluate all strategic policy scenarios
  • Execute representative subpolicy analyses (single and combined)

Outputs are written to ``outputs/final_run_<timestamp>.json``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgce_model.orchestration.policy_package import run_policy_package
from dgce_model.api.json_serialization import json_default


def run_all() -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()

    common_params = {
        "standard_rate": 0.09,
        "threshold": 375_000,
        "small_biz_threshold": 3_000_000,
        "oil_gas_rate": 0.55,
        "fz_qualifying_rate": 0.0,
        "sme_election_rate": 0.80,
        "compliance_rate": 0.75,
        "vat_rate": 0.05,
        "government_spending_rel_change": 0.0,
        "years": 5,
    }

    payload = {
        "quick_params": common_params,
        "comprehensive_params": {
            "standard_rate": 0.09,
            "compliance": 0.75,
            "threshold": 375_000,
            "small_biz_threshold": 3_000_000,
            "oil_gas_rate": 0.55,
            "vat_rate": 0.05,
            "government_spending_rel_change": 0.0,
            "time_horizon": 10,
        },
        "sector_params": [
            {"sector": "Manufacturing", "params": {"compliance_rate": 0.75}}
        ],
        "strategic": {},
        "subpolicy": {
            "single": {"rule_type": "small_business", "parameters": {"threshold": 4_000_000}},
            "multiple": [
                {"type": "small_business", "parameters": {"threshold": 4_500_000}},
                {"type": "free_zone", "parameters": {"qualifying_income_rate": 0.60}},
            ],
        },
    }

    result = run_policy_package(payload)
    result["timestamp_utc"] = timestamp
    return result


def main() -> None:
    output_payload = run_all()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"final_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path = output_dir / file_name

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, default=json_default)

    print(f"Final run output written to {output_path}")


if __name__ == "__main__":
    main()

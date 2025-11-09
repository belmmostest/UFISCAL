"""Machine Control Protocol endpoints for DGCE policy simulations.

The primary entry point is :func:`run_policy_package`, which accepts a JSON-like
payload describing policy shocks and returns the combined output of all core
analysis surfaces (quick, comprehensive, sectoral, strategic, subpolicy).

Example payload::

    {
        "quick_params": {"standard_rate": 0.095},
        "comprehensive_params": {"time_horizon": 7},
        "sector_params": [
            {"sector": "Manufacturing", "params": {"compliance_rate": 0.8}},
            {"sector": "Information and communication"}
        ],
        "strategic": {
            "scenarios": [
                {"id": "growth_focused", "overrides": {"standard_rate": 0.085}}
            ]
        },
        "subpolicy": {
            "single": {"rule_type": "small_business", "parameters": {"threshold": 4_500_000}},
            "multiple": [
                {"type": "small_business", "parameters": {"threshold": 4_250_000}},
                {"type": "free_zone", "parameters": {"qualifying_income_rate": 0.65}}
            ]
        }
    }

Any omitted sections fall back to sensible defaults (baseline tax design,
Manufacturing sector check, all strategic scenarios, and default subpolicy probes).
"""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Dict, List

from dgce_model.api.quick_simulation_enhanced import EnhancedQuickSimulation
from dgce_model.api.comprehensive_analysis_enhanced import ComprehensivePolicyAnalyzer
from dgce_model.api.sector_analyzer_enhanced import SectorPolicyAnalyzer
from dgce_model.api.strategic_policy_scenarios import list_scenarios, run_scenario
from dgce_model.api.subpolicy_analyzer import (
    analyze_multiple_rules,
    analyze_single_rule,
)


DEFAULT_QUICK_PARAMS: Dict[str, Any] = {
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
    "incentives": {},
}

DEFAULT_COMPREHENSIVE_PARAMS: Dict[str, Any] = {
    "standard_rate": 0.09,
    "compliance": 0.75,
    "threshold": 375_000,
    "small_biz_threshold": 3_000_000,
    "oil_gas_rate": 0.55,
    "vat_rate": 0.05,
    "government_spending_rel_change": 0.0,
    "time_horizon": 10,
}

DEFAULT_SECTOR_PARAMS: Dict[str, Any] = {
    "standard_rate": 0.09,
    "compliance_rate": 0.75,
    "threshold": 375_000,
    "small_biz_threshold": 3_000_000,
    "oil_gas_rate": 0.55,
    "incentives": {},
}

DEFAULT_SUBPOLICY_SINGLE: Dict[str, Any] = {
    "rule_type": "small_business",
    "parameters": {"threshold": 4_000_000},
}

DEFAULT_SUBPOLICY_MULTIPLE = [
    {"type": "small_business", "parameters": {"threshold": 4_250_000}},
    {"type": "free_zone", "parameters": {"qualifying_income_rate": 0.65}},
]

EXAMPLE_PAYLOAD: Dict[str, Any] = {
    "quick_params": {
        "standard_rate": 0.095,
        "threshold": 400_000,
        "small_biz_threshold": 3_500_000,
        "oil_gas_rate": 0.58,
        "fz_qualifying_rate": 0.10,
        "sme_election_rate": 0.75,
        "compliance_rate": 0.78,
        "vat_rate": 0.05,
        "government_spending_rel_change": 0.01,
        "years": 7,
        "incentives": {"training_grant": 0.02},
    },
    "comprehensive_params": {
        "standard_rate": 0.095,
        "compliance": 0.78,
        "threshold": 400_000,
        "small_biz_threshold": 3_500_000,
        "oil_gas_rate": 0.58,
        "vat_rate": 0.05,
        "government_spending_rel_change": 0.01,
        "time_horizon": 8,
        "incentives": {"green_credit": 0.01},
    },
    "sector_params": [
        {
            "sector": "Manufacturing",
            "params": {
                "standard_rate": 0.09,
                "compliance_rate": 0.80,
                "threshold": 375_000,
                "small_biz_threshold": 3_000_000,
                "oil_gas_rate": 0.55,
                "incentives": {"export_rebate": 0.02},
            },
        },
        {
            "sector": "Information and communication",
            "params": {
                "standard_rate": 0.085,
                "compliance_rate": 0.82,
                "threshold": 350_000,
            },
        },
    ],
    "strategic": {
        "scenarios": [
            {"id": "growth_focused", "overrides": {"standard_rate": 0.085, "compliance": 0.78}},
            {"id": "revenue_focused", "overrides": {"standard_rate": 0.11}},
        ]
    },
    "subpolicy": {
        "single": {"rule_type": "small_business", "parameters": {"threshold": 4_200_000}},
        "multiple": [
            {"type": "small_business", "parameters": {"threshold": 4_300_000}},
            {"type": "free_zone", "parameters": {"qualifying_income_rate": 0.55}},
            {"type": "loss_carryforward", "parameters": {"duration_years": 5}},
        ],
    },
}


def _merge_params(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    merged.update(override)
    return merged


def run_policy_package(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full DGCE pipeline using user-supplied parameters."""

    timestamp = datetime.now(timezone.utc).isoformat()

    quick_params = _merge_params(
        DEFAULT_QUICK_PARAMS,
        payload.get("quick_params", {}),
    )
    quick_sim = EnhancedQuickSimulation()
    quick_result = quick_sim.run_simulation(quick_params)

    comprehensive_params = _merge_params(
        DEFAULT_COMPREHENSIVE_PARAMS,
        payload.get("comprehensive_params", {}),
    )
    comprehensive_analyzer = ComprehensivePolicyAnalyzer()
    comprehensive_result = comprehensive_analyzer.analyze_comprehensive_policy(
        comprehensive_params
    )

    sector_requests = payload.get("sector_params")
    if not sector_requests:
        sector_requests = [{"sector": "Manufacturing", "params": {}}]
    else:
        sector_requests = list(deepcopy(sector_requests))

    sector_analyzer = SectorPolicyAnalyzer()
    sector_results: Dict[str, Any] = {}
    for sector_entry in sector_requests:
        sector_name = sector_entry.get("sector", "Manufacturing")
        overrides = sector_entry.get("params", {})
        merged_params = _merge_params(DEFAULT_SECTOR_PARAMS, overrides)
        sector_results[sector_name] = sector_analyzer.analyze_sector_policy(
            sector_name, merged_params
        )

    strategic_config = payload.get("strategic", {})
    strategic_requests = deepcopy(strategic_config.get("scenarios"))
    strategic_outputs: Dict[str, Any] = {}

    if strategic_requests:
        for scenario_request in strategic_requests:
            scenario_id = scenario_request.get("id")
            if not scenario_id:
                continue
            overrides = scenario_request.get("overrides")
            strategic_outputs[scenario_id] = run_scenario(
                scenario_id,
                overrides=overrides,
                analyzer=comprehensive_analyzer,
            )
    else:
        for scenario in list_scenarios():
            scenario_id = scenario["id"]
            strategic_outputs[scenario_id] = run_scenario(
                scenario_id,
                analyzer=comprehensive_analyzer,
            )

    subpolicy_config = payload.get("subpolicy", {})
    single_cfg = deepcopy(subpolicy_config.get("single", DEFAULT_SUBPOLICY_SINGLE))
    multiple_cfg = deepcopy(subpolicy_config.get("multiple", DEFAULT_SUBPOLICY_MULTIPLE))

    subpolicy_single_result = analyze_single_rule(
        single_cfg.get("rule_type", DEFAULT_SUBPOLICY_SINGLE["rule_type"]),
        single_cfg.get("parameters", DEFAULT_SUBPOLICY_SINGLE["parameters"]),
    )
    subpolicy_multiple_result = analyze_multiple_rules(multiple_cfg)

    return {
        "status": "success",
        "timestamp_utc": timestamp,
        "quick_simulation_enhanced": quick_result,
        "comprehensive_analysis_enhanced": comprehensive_result,
        "sector_analyzer_enhanced": sector_results,
        "strategic_policy_scenarios": strategic_outputs,
        "subpolicy_single_rule": subpolicy_single_result,
        "subpolicy_multiple_rules": subpolicy_multiple_result,
    }


__all__ = ["run_policy_package", "EXAMPLE_PAYLOAD"]

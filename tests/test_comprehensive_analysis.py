import pandas as pd
import pytest

import dgce_model.model.dgce_model_enhanced_fixed as dgce_model_module
import dgce_model.data_loader.new_data_loader as data_loader_module
import dgce_model.api.comprehensive_analysis_enhanced as analysis_module
from dgce_model.api.comprehensive_analysis_enhanced import ComprehensivePolicyAnalyzer


class StubDGCEModel:
    def __init__(self):
        self.calls = {"apply_scenario": 0, "simulate": 0, "solve_policy_impact": 0}
        self.steady_state = {
            "corporate_tax_base": 120_000.0,
            "consumption": 200_000.0,
            "oil_revenue": 400_000.0,
        }

    def apply_scenario(self, params):
        self.calls["apply_scenario"] += 1
        return {
            "gdp_impact": 0.5,
            "employment_impact": 0.2,
            "investment_impact": 0.3,
            "consumption_impact": 0.1,
            "revenue_analysis": {
                "corporate_component": 150.0,
                "baseline_revenue": 1000.0,
                "revenue_change": 25.0,
                "revenue_change_pct": 2.5,
            },
            "sectoral_impacts": {"Manufacturing": -0.02},
            "fiscal_balance": 10.0,
            "oil_sector": {"production": 100.0, "revenue": 220.0},
        }

    def simulate(self, policy_shock, *, years):
        self.calls["simulate"] += 1
        return pd.DataFrame(
            {
                "year_index": list(range(1, years + 1)),
                "gdp": [100.0] * years,
                "consumption": [60.0] * years,
                "investment": [20.0] * years,
                "employment": [5.0] * years,
            }
        )

    def solve_policy_impact(self, shocks):
        self.calls["solve_policy_impact"] += 1
        return {
            "gdp_growth": 0.004,
            "employment_growth": 0.002,
            "investment_growth": 0.003,
            "consumption_growth": 0.001,
            "revenue_analysis": {
                "corporate_component": 160.0,
                "revenue_change": 30.0,
            },
            "sectoral_impacts": {"Manufacturing": -0.015},
            "fiscal_balance": 12.0,
            "oil_production": 95.0,
            "oil_revenue": 210.0,
            "corporate_tax_base": 118_000.0,
        }


class StubLoader:
    def __init__(self, *args, **kwargs):
        self.commerce_registry = pd.DataFrame(
            {
                "id": [1, 2],
                "ISIC_level_1": ["Manufacturing", "Mining and quarrying"],
                "status": ["Active", "Active"],
                "annual_revenue": [2_500_000.0, 4_500_000.0],
                "employee_count": [40, 50],
                "is_free_zone": [False, False],
            }
        )
        self.sectoral_panel = pd.DataFrame(
            {
                "economic_activity": ["Manufacturing", "Mining and quarrying"],
                "year": [2023, 2023],
                "output_in_aed": [8_000_000.0, 10_000_000.0],
                "intermediate_consumption_in_aed": [4_000_000.0, 5_000_000.0],
                "value_added_in_aed": [4_000_000.0, 5_000_000.0],
                "compensation_of_employees_in_aed": [2_000_000.0, 2_500_000.0],
                "gross_fixed_capital_formation_in_aed": [600_000.0, 700_000.0],
                "number_of_employees": [80, 90],
            }
        )


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr(dgce_model_module, "SimplifiedDGCEModel", StubDGCEModel)
    monkeypatch.setattr(data_loader_module, "RealUAEDataLoader", StubLoader)
    monkeypatch.setattr(analysis_module, "SimplifiedDGCEModel", StubDGCEModel)
    monkeypatch.setattr(analysis_module, "RealUAEDataLoader", StubLoader)
    yield


def test_comprehensive_analysis_returns_expected_sections():
    analyzer = ComprehensivePolicyAnalyzer()
    params = {
        "standard_rate": 0.09,
        "compliance": 0.75,
        "threshold": 375_000,
        "small_biz_threshold": 3_000_000,
        "oil_gas_rate": 0.55,
        "vat_rate": 0.05,
        "government_spending_rel_change": 0.0,
        "time_horizon": 3,
    }

    result = analyzer.analyze_comprehensive_policy(params)

    assert "microsimulation" in result
    assert result["microsimulation"]["validation_passed"] is True
    assert "revenue_analysis" in result
    assert "dynamic_impacts" in result and result["dynamic_impacts"]["years"] == 3
    assert result["policy_parameters"]["standard_rate"] == params["standard_rate"]

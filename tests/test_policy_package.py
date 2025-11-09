import pytest

from dgce_model.orchestration import policy_package as mcp_endpoints


def test_run_policy_package_defaults(monkeypatch):
    class DummyQuick:
        def run_simulation(self, params):
            return {"tag": "quick", "params": params}

    class DummyComprehensive:
        def analyze_comprehensive_policy(self, params):
            return {"tag": "comprehensive", "params": params}

    class DummySector:
        def analyze_sector_policy(self, sector, params):
            return {"tag": "sector", "sector": sector, "params": params}

    def dummy_run_scenario(name, overrides=None, analyzer=None):
        return {"tag": "strategic", "name": name, "overrides": overrides}

    def dummy_single(rule_type, parameters):
        return {"tag": "single", "rule_type": rule_type, "parameters": parameters}

    def dummy_multiple(rules):
        return {"tag": "multiple", "rules": rules}

    monkeypatch.setattr(mcp_endpoints, "EnhancedQuickSimulation", lambda: DummyQuick())
    monkeypatch.setattr(mcp_endpoints, "ComprehensivePolicyAnalyzer", lambda: DummyComprehensive())
    monkeypatch.setattr(mcp_endpoints, "SectorPolicyAnalyzer", lambda: DummySector())
    monkeypatch.setattr(mcp_endpoints, "run_scenario", dummy_run_scenario)
    monkeypatch.setattr(mcp_endpoints, "list_scenarios", lambda: [{"id": "alpha"}, {"id": "beta"}])
    monkeypatch.setattr(mcp_endpoints, "analyze_single_rule", dummy_single)
    monkeypatch.setattr(mcp_endpoints, "analyze_multiple_rules", dummy_multiple)

    output = mcp_endpoints.run_policy_package({})

    assert output["quick_simulation_enhanced"]["tag"] == "quick"
    assert output["comprehensive_analysis_enhanced"]["tag"] == "comprehensive"
    assert "alpha" in output["strategic_policy_scenarios"]
    assert output["subpolicy_multiple_rules"]["tag"] == "multiple"

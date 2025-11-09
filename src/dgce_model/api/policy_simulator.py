"""
Policy simulator for UAE corporate tax system.

This module provides comprehensive policy simulation capabilities including:
  - Quick simulation
  - Comprehensive analysis
  - Sector-specific analysis
  - Strategic scenario execution
  - Subpolicy rule analysis
"""
from typing import Dict, List, Any

from .quick_simulation_enhanced import EnhancedQuickSimulation
from .comprehensive_analysis_enhanced import ComprehensivePolicyAnalyzer
from .sector_analyzer_enhanced import EnhancedSectorAnalyzer
from .strategic_policy_scenarios import run_scenario, list_scenarios
from .subpolicy_analyzer import analyze_single_rule, analyze_multiple_rules

class PolicySimulator:
    """Wrapper class to orchestrate policy simulations."""
    def __init__(self):
        self.quick = EnhancedQuickSimulation()
        self.comprehensive = ComprehensivePolicyAnalyzer()
        self.sector = EnhancedSectorAnalyzer()

    def simulate_quick(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.quick.run_simulation(params)

    def simulate_comprehensive(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.comprehensive.analyze_comprehensive_policy(params)

    def simulate_sector(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sector = params.get('sector')
        return self.sector.analyze_sector(sector, params)

    def simulate_strategic(self, scenario: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        return run_scenario(scenario, overrides)

    def list_strategic(self) -> List[Dict[str, Any]]:
        return list_scenarios()

    def analyze_rule(self, rule_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return analyze_single_rule(rule_type, parameters)

    def analyze_rules(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        return analyze_multiple_rules(rules)

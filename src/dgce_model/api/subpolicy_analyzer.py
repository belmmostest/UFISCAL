"""
Subpolicy analyzer for specific corporate tax rule simulations.

This module provides functionality to analyze individual CT parameters and their impacts,
including threshold adjustments, relief provisions, deduction limitations, and more.
"""
from typing import Dict, Any, List
from .quick_simulation_enhanced import EnhancedQuickSimulation

_quick = EnhancedQuickSimulation()

def analyze_single_rule(rule_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single corporate tax rule change."""
    params = {
        'standard_rate': 0.09,
        'threshold': 0.0,
        'small_biz_threshold': 3000000.0,
        'oil_gas_rate': 0.55,
        'compliance_rate': 0.70
    }
    # Apply rule-specific overrides
    if rule_type == 'small_business':
        params['small_biz_threshold'] = float(parameters.get('threshold', params['small_biz_threshold']))
    elif rule_type == 'free_zone':
        params['fz_qualifying_rate'] = float(parameters.get('qualifying_income_rate', 0.80))
    else:
        # other rules can be mapped here
        pass
    # Run quick simulation to capture revenue and GE effects
    result = _quick.run_simulation(params)
    # Attach rule metadata
    result['rule_analysis'] = {
        'rule_type': rule_type,
        'parameters': parameters
    }
    return result

def analyze_multiple_rules(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze multiple corporate tax rules in combination."""
    # Start with base params
    params = {
        'standard_rate': 0.09,
        'threshold': 0.0,
        'small_biz_threshold': 3000000.0,
        'oil_gas_rate': 0.55,
        'compliance_rate': 0.70
    }
    # Apply each rule override
    for rule in rules:
        rtype = rule.get('type')
        prms = rule.get('parameters', {})
        if rtype == 'small_business':
            params['small_biz_threshold'] = float(prms.get('threshold', params['small_biz_threshold']))
        elif rtype == 'free_zone':
            params['fz_qualifying_rate'] = float(prms.get('qualifying_income_rate', 0.80))
        # add other rule types as needed
    # Run combined quick simulation
    result = _quick.run_simulation(params)
    result['rules_analysis'] = rules
    return result

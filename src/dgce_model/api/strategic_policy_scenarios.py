"""
Fixed Strategic Policy Scenarios for UAE Ministry of Finance
===========================================================

Strategic scenarios with proper fallbacks and working analysis.
"""
from typing import Dict, Any, List

# Import with fallback
try:
    from .comprehensive_analysis_enhanced import ComprehensivePolicyAnalyzer
except Exception:  # pragma: no cover - fallback used in limited contexts
    ComprehensivePolicyAnalyzer = None

# Fallback analyzer for strategic scenarios
class FallbackStrategicAnalyzer:
    """Fallback analyzer for strategic scenarios when comprehensive analyzer fails."""
    
    def analyze_comprehensive_policy(self, params: Dict) -> Dict:
        """Simplified strategic analysis."""
        standard_rate = params.get('standard_rate', 0.09)
        rate_change = standard_rate - 0.09
        
        # Apply scenario-specific multipliers
        scenario_multipliers = {
            'growth_focused': {'gdp': 1.2, 'employment': 1.3, 'revenue': 0.8},
            'revenue_focused': {'gdp': 0.8, 'employment': 0.7, 'revenue': 1.4},
            'innovation_hub': {'gdp': 1.5, 'employment': 1.1, 'revenue': 0.9},
            'regional_leader': {'gdp': 1.1, 'employment': 1.0, 'revenue': 1.1}
        }
        
        # Determine scenario from rate and incentives
        scenario_type = 'growth_focused'
        if standard_rate > 0.10:
            scenario_type = 'revenue_focused'
        elif params.get('incentives', {}).get('rd_credit'):
            scenario_type = 'innovation_hub'
        elif standard_rate < 0.09:
            scenario_type = 'regional_leader'
        
        multipliers = scenario_multipliers.get(scenario_type, {'gdp': 1.0, 'employment': 1.0, 'revenue': 1.0})
        
        # Calculate impacts with scenario multipliers
        gdp_impact = rate_change * -0.25 * multipliers['gdp'] * 100
        employment_impact = rate_change * -0.33 * multipliers['employment'] * 100
        revenue_change = rate_change * 15000 * multipliers['revenue']  # 15B tax base
        
        return {
            'timestamp': '2024-01-01T00:00:00',
            'static_equilibrium': {
                'gdp_growth': gdp_impact / 100,
                'employment_growth': employment_impact / 100,
                'investment_growth': rate_change * -0.8 * multipliers['gdp']
            },
            'executive_summary': {
                'gdp_impact': gdp_impact,
                'revenue_impact': revenue_change,
                'employment_impact': employment_impact,
                'overall_assessment': f'{scenario_type.replace("_", " ").title()} strategy shows {gdp_impact:.1f}% GDP impact'
            },
            'sectoral_analysis': {
                'sector_impacts': {
                    'Financial and insurance activities': {
                        'output_impact': gdp_impact * 1.2,
                        'employment_impact': employment_impact * 1.1
                    },
                    'Manufacturing': {
                        'output_impact': gdp_impact * 0.8,
                        'employment_impact': employment_impact * 0.9
                    }
                }
            },
            'revenue_analysis': {
                'total_revenue': revenue_change + 25000,
                'revenue_efficiency': 0.85
            },
            'risk_assessment': {
                'overall_risk': 'Medium',
                'key_risks': ['Implementation complexity', 'Market adaptation']
            },
            'recommendations': [
                f'Implement {scenario_type.replace("_", " ")} strategy gradually',
                'Monitor sector-specific impacts',
                'Maintain regional competitiveness'
            ]
        }

# Strategic scenario definitions
SCENARIOS: Dict[str, Dict[str, Any]] = {
    'growth_focused': {
        'standard_rate': 0.07,
        'compliance': 0.75,
        'incentives': {'rd_credit': True, 'startup_exemption': True},
        'time_horizon': 10,
        'description': 'Growth-focused policy with lower rates and innovation incentives'
    },
    'revenue_focused': {
        'standard_rate': 0.12,
        'compliance': 0.80,
        'time_horizon': 10,
        'incentives': {},
        'description': 'Revenue optimization with higher rates and enhanced compliance'
    },
    'innovation_hub': {
        'standard_rate': 0.09,
        'compliance': 0.70,
        'time_horizon': 10,
        'incentives': {'rd_credit': True, 'startup_exemption': True, 'hq_incentive': True},
        'description': 'Innovation hub strategy with comprehensive R&D incentives'
    },
    'regional_leader': {
        'standard_rate': 0.08,
        'compliance': 0.75,
        'time_horizon': 10,
        'incentives': {'hq_incentive': True},
        'description': 'Regional leadership through competitive rates and HQ incentives'
    }
}

def list_scenarios() -> List[Dict[str, Any]]:
    """List available strategic scenarios with metadata."""
    return [
        {
            'id': key, 
            'name': key.replace('_', ' ').title(),
            'description': val.get('description', ''),
            'standard_rate': val.get('standard_rate', 0.09),
            'key_features': list(val.get('incentives', {}).keys())
        }
        for key, val in SCENARIOS.items()
    ]

def get_scenario_params(name: str) -> Dict[str, Any]:
    """Get parameters for a given strategic scenario."""
    return SCENARIOS.get(name, {})

def run_scenario(name: str, overrides: Dict[str, Any] = None, analyzer: Any | None = None) -> Dict[str, Any]:
    """Execute a strategic policy scenario with optional overrides."""
    print(f"Running strategic scenario: {name}")
    
    if name not in SCENARIOS:
        available = list(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {name}. Available scenarios: {available}")
    
    # Get base scenario parameters
    params = SCENARIOS[name].copy()
    
    # Apply any overrides
    if overrides:
        params.update(overrides)
        print(f"Applied overrides: {overrides}")
    
    print(f"Final scenario params: {params}")
    
    try:
        # Try to use the real comprehensive analyzer
        if analyzer is not None:
            result = analyzer.analyze_comprehensive_policy(params)
            print("Used comprehensive analyzer successfully")
        elif ComprehensivePolicyAnalyzer is not None:
            analyzer_instance = ComprehensivePolicyAnalyzer()
            result = analyzer_instance.analyze_comprehensive_policy(params)
            print("Used comprehensive analyzer successfully")
        else:
            # Use fallback analyzer
            fallback = FallbackStrategicAnalyzer()
            result = fallback.analyze_comprehensive_policy(params)
            print("Used fallback strategic analyzer")
            
    except Exception as e:
        print(f"Error in scenario analysis, using fallback: {e}")
        fallback = FallbackStrategicAnalyzer()
        result = fallback.analyze_comprehensive_policy(params)
        result['fallback_used'] = True
        result['error'] = str(e)
    
    # Add scenario metadata to result
    result['scenario_name'] = name
    result['scenario_description'] = SCENARIOS[name].get('description', '')
    result['scenario_parameters'] = params
    
    print(f"Strategic scenario {name} completed")
    return result

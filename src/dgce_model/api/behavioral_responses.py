"""
Behavioral response models for UAE corporate tax simulations.

These models predict how businesses respond to tax policy changes based on
economic theory and empirical evidence.
"""

from typing import Dict, Optional

def output_response(base_output: float,
                    old_tax_rate: float,
                    new_tax_rate: float,
                    elasticity: float) -> float:
    """
    Estimate new output based on tax rate change and output elasticity.
    """
    if old_tax_rate <= 0:
        return base_output
    rate_change = (new_tax_rate - old_tax_rate) / old_tax_rate
    return base_output * (1 + elasticity * rate_change)

def investment_response(base_investment: float,
                        old_tax_rate: float,
                        new_tax_rate: float,
                        elasticity: float) -> float:
    """
    Estimate new investment based on tax rate change and investment elasticity.
    """
    if old_tax_rate <= 0:
        return base_investment
    rate_change = (new_tax_rate - old_tax_rate) / old_tax_rate
    return base_investment * (1 + elasticity * rate_change)

def employment_response(base_employment: float,
                        old_tax_rate: float,
                        new_tax_rate: float,
                        elasticity: float) -> float:
    """
    Estimate new employment based on tax rate change and employment elasticity.
    """
    if old_tax_rate <= 0:
        return base_employment
    rate_change = (new_tax_rate - old_tax_rate) / old_tax_rate
    return base_employment * (1 + elasticity * rate_change)

def compliance_response(base_compliance: float,
                        enforcement_strength: float,
                        elasticity: float) -> float:
    """
    Estimate compliance rate change given enforcement strength and elasticity.
    enforcement_strength: relative improvement (0-1) in enforcement capacity.
    """
    return max(0.0, min(1.0, base_compliance * (1 + elasticity * enforcement_strength)))

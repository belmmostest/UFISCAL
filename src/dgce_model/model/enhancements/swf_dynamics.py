"""
Sovereign Wealth Fund (SWF) Dynamics for UAE DGCE Model
=======================================================

Implements dynamic SWF accumulation and return flows for:
- Abu Dhabi Investment Authority (ADIA)
- Mubadala Investment Company
- Investment Corporation of Dubai (ICD)
- Emirates Investment Authority (EIA)
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SWFParameters:
    """Parameters for UAE sovereign wealth funds."""
    
    # SWF sizes (estimated, billion USD)
    adia_size: float = 850.0
    mubadala_size: float = 285.0
    icd_size: float = 300.0
    eia_size: float = 45.0
    
    # Return assumptions (annual %)
    adia_return: float = 0.065  # Conservative long-term
    mubadala_return: float = 0.08  # More aggressive
    icd_return: float = 0.07
    eia_return: float = 0.06
    
    # Domestic investment share
    domestic_share: float = 0.15  # 15% invested domestically
    
    # Oil revenue allocation rules
    oil_allocation_rules = {
        'adia': 0.60,      # 60% of surplus to ADIA
        'mubadala': 0.20,  # 20% to Mubadala
        'icd': 0.15,       # 15% to ICD
        'eia': 0.05        # 5% to EIA
    }


class SWFDynamicsBlock:
    """Dynamic sovereign wealth fund management."""
    
    def __init__(self, initial_assets: Dict[str, float] = None):
        self.params = SWFParameters()
        
        # Initialize SWF assets (billion AED)
        if initial_assets:
            self.swf_assets = initial_assets
        else:
            # Convert from USD to AED (rate: 3.67)
            self.swf_assets = {
                'adia': self.params.adia_size * 3.67,
                'mubadala': self.params.mubadala_size * 3.67,
                'icd': self.params.icd_size * 3.67,
                'eia': self.params.eia_size * 3.67
            }
        
        # Track flows
        self.swf_flows = {
            'inflows': [],
            'returns': [],
            'domestic_investment': [],
            'foreign_investment': []
        }
        
        # Calculate total assets
        self.total_assets = sum(self.swf_assets.values())
    
    def update_swf_assets(self, 
                         fiscal_surplus: float,
                         oil_windfall: float = 0) -> Dict:
        """
        Update SWF assets based on fiscal surplus and oil windfalls.
        
        Args:
            fiscal_surplus: Government surplus (million AED)
            oil_windfall: Additional oil revenue above baseline
            
        Returns:
            Dictionary with updated assets and flows
        """
        
        # Total inflow to SWFs
        total_inflow = max(0, fiscal_surplus + oil_windfall)
        
        # Allocate to different funds
        inflows = {}
        returns = {}
        domestic_inv = {}
        foreign_inv = {}
        
        for fund, allocation in self.params.oil_allocation_rules.items():
            # New inflows
            inflows[fund] = total_inflow * allocation / 1000  # Convert to billion
            
            # Investment returns
            return_rate = getattr(self.params, f'{fund}_return')
            returns[fund] = self.swf_assets[fund] * return_rate
            
            # Total available for investment
            total_available = inflows[fund] + returns[fund]
            
            # Domestic vs foreign investment
            domestic_inv[fund] = total_available * self.params.domestic_share
            foreign_inv[fund] = total_available * (1 - self.params.domestic_share)
            
            # Update assets
            self.swf_assets[fund] += inflows[fund] + returns[fund]
        
        # Record flows
        self.swf_flows['inflows'].append(inflows)
        self.swf_flows['returns'].append(returns)
        self.swf_flows['domestic_investment'].append(domestic_inv)
        self.swf_flows['foreign_investment'].append(foreign_inv)
        
        # Calculate totals
        total_domestic = sum(domestic_inv.values())
        total_foreign = sum(foreign_inv.values())
        total_returns = sum(returns.values())
        
        return {
            'assets': self.swf_assets.copy(),
            'total_assets': sum(self.swf_assets.values()),
            'returns': total_returns,
            'domestic_investment': total_domestic,
            'foreign_investment': total_foreign,
            'domestic_impact_gdp': total_domestic * 1000 / 1_600_000  # As % of GDP
        }
    
    def calculate_wealth_effect(self) -> float:
        """
        Calculate wealth effect on consumption from SWF assets.
        
        Returns:
            Wealth effect multiplier for household consumption
        """
        total_wealth = sum(self.swf_assets.values())
        
        # Assume 3% wealth effect on consumption
        wealth_effect_rate = 0.03
        
        # Only domestic portion affects consumption
        domestic_wealth = total_wealth * self.params.domestic_share
        
        # Wealth effect in million AED
        wealth_effect = domestic_wealth * wealth_effect_rate * 1000
        
        return wealth_effect
    
    def simulate_oil_price_scenario(self, 
                                  oil_prices: List[float],
                                  years: int = 10) -> Dict:
        """
        Simulate SWF evolution under oil price scenario.
        
        Args:
            oil_prices: List of oil prices by year
            years: Number of years to simulate
            
        Returns:
            SWF trajectories
        """
        results = {
            'years': list(range(years)),
            'total_assets': [],
            'domestic_investment': [],
            'wealth_effect': []
        }
        
        # Reset to initial state
        self.__init__(self.swf_assets.copy())
        
        for year, oil_price in enumerate(oil_prices[:years]):
            # Simple fiscal surplus model
            baseline_oil_price = 80
            oil_sensitivity = 2000  # Million AED per $1 oil price
            
            fiscal_surplus = (oil_price - baseline_oil_price) * oil_sensitivity
            
            # Update SWF
            swf_update = self.update_swf_assets(fiscal_surplus)
            
            results['total_assets'].append(swf_update['total_assets'])
            results['domestic_investment'].append(swf_update['domestic_investment'])
            results['wealth_effect'].append(self.calculate_wealth_effect())
        
        return results
    
    def get_stabilization_capacity(self) -> float:
        """
        Calculate fiscal stabilization capacity from SWF assets.
        
        Returns:
            Maximum annual drawdown sustainable for 5 years (million AED)
        """
        # Assume can draw down 20% of assets over 5 years
        total_assets = sum(self.swf_assets.values())
        max_drawdown = total_assets * 0.20
        
        # Annual sustainable drawdown
        annual_drawdown = max_drawdown / 5 * 1000  # Convert to million AED
        
        return annual_drawdown


# Example usage
if __name__ == "__main__":
    # Initialize SWF dynamics
    swf = SWFDynamicsBlock()
    
    print("Initial SWF Assets (billion AED):")
    for fund, assets in swf.swf_assets.items():
        print(f"  {fund.upper()}: {assets:,.0f}")
    print(f"  Total: {sum(swf.swf_assets.values()):,.0f}")
    
    # Simulate oil boom scenario
    print("\nOil Boom Scenario ($100/barrel for 5 years):")
    boom_prices = [100] * 5 + [80] * 5
    results = swf.simulate_oil_price_scenario(boom_prices)
    
    for i in range(10):
        print(f"Year {i+1}: Assets {results['total_assets'][i]:,.0f}B AED, "
              f"Domestic Inv {results['domestic_investment'][i]:,.0f}B AED")
    
    # Calculate stabilization capacity
    capacity = swf.get_stabilization_capacity()
    print(f"\nFiscal stabilization capacity: {capacity:,.0f} million AED/year")
"""
Price Dynamics Module for UAE DGCE Model
========================================

Implements endogenous price determination including:
- Wage determination by skill/nationality
- Capital rental rates
- Sectoral price indices
- Real exchange rate (under AED peg)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass 
class PriceParameters:
    """Parameters for price dynamics."""
    
    # Fixed exchange rate (AED per USD) - pegged since 1997
    AED_USD_PEG: float = 3.6725
    
    # Price adjustment speeds
    wage_adjustment_speed: float = 0.3
    rental_adjustment_speed: float = 0.4
    goods_price_adjustment: float = 0.5
    
    # Wage bargaining power
    worker_bargaining_power = {
        'emirati': 0.6,          # Higher due to regulations
        'expat_high': 0.4,       # Some bargaining power
        'expat_medium': 0.2,     # Limited power
        'expat_low': 0.1        # Minimal power
    }
    
    # Price elasticities
    labor_demand_elasticity: float = -0.5
    labor_supply_elasticity: float = 0.3
    
    # Minimum wages (AED/month)
    minimum_wages = {
        'emirati': 10000,       # Implicit minimum
        'expat_high': 8000,
        'expat_medium': 3000,
        'expat_low': 1500
    }


class PriceDynamics:
    """Dynamic price determination system."""
    
    def __init__(self):
        self.params = PriceParameters()
        
        # Initialize price indices (base = 1.0)
        self.price_indices = {
            'consumer': 1.0,
            'producer': 1.0,
            'import': 1.0,
            'export': 1.0
        }
        
        # Wage history for persistence
        self.wage_history = []
        
    def determine_equilibrium_wage(self,
                                 labor_demand: float,
                                 labor_supply: float,
                                 current_wage: float,
                                 worker_type: str) -> float:
        """
        Determine equilibrium wage through supply-demand dynamics.
        
        Args:
            labor_demand: Quantity of labor demanded
            labor_supply: Quantity of labor supplied
            current_wage: Current wage level
            worker_type: Type of worker (for bargaining power)
            
        Returns:
            New equilibrium wage
        """
        # Calculate excess demand
        if labor_supply > 0:
            excess_demand_rate = (labor_demand - labor_supply) / labor_supply
        else:
            excess_demand_rate = 1.0
        
        # Wage adjustment based on excess demand
        # Positive excess demand -> wage increases
        wage_pressure = excess_demand_rate * self.params.wage_adjustment_speed
        
        # Bargaining power adjustment
        bargaining = self.params.worker_bargaining_power.get(worker_type, 0.2)
        wage_pressure *= (1 + bargaining)
        
        # Calculate new wage
        new_wage = current_wage * (1 + wage_pressure)
        
        # Apply minimum wage constraint
        min_wage = self.params.minimum_wages.get(worker_type, 1500)
        new_wage = max(new_wage, min_wage)
        
        # Add persistence (wages are sticky)
        if self.wage_history:
            last_wage = self.wage_history[-1].get(worker_type, current_wage)
            persistence = 0.7  # 70% persistence
            new_wage = persistence * last_wage + (1 - persistence) * new_wage
        
        return new_wage
    
    def determine_capital_rental_rate(self,
                                    capital_demand: float,
                                    capital_stock: float,
                                    current_rate: float,
                                    depreciation: float = 0.05) -> float:
        """
        Determine capital rental rate based on utilization.
        
        Args:
            capital_demand: Demanded capital services
            capital_stock: Available capital stock
            current_rate: Current rental rate
            depreciation: Depreciation rate
            
        Returns:
            New rental rate
        """
        # Capital utilization rate
        if capital_stock > 0:
            utilization = min(capital_demand / capital_stock, 1.0)
        else:
            utilization = 1.0
        
        # Target utilization
        target_utilization = 0.85
        
        # Rental rate adjustment
        utilization_gap = utilization - target_utilization
        rate_adjustment = utilization_gap * self.params.rental_adjustment_speed
        
        # New rental rate
        new_rate = current_rate * (1 + rate_adjustment)
        
        # Must cover depreciation
        min_rate = depreciation * 1.2  # 20% markup over depreciation
        new_rate = max(new_rate, min_rate)
        
        return new_rate
    
    def determine_sectoral_prices(self,
                                unit_costs: Dict[str, float],
                                demand_pressure: Dict[str, float],
                                import_share: Dict[str, float]) -> Dict[str, float]:
        """
        Determine sectoral output prices.
        
        Args:
            unit_costs: Unit production costs by sector
            demand_pressure: Demand/supply ratio by sector
            import_share: Import penetration by sector
            
        Returns:
            Sectoral price indices
        """
        sectoral_prices = {}
        
        for sector in unit_costs:
            # Base price from costs (with markup)
            markup = 1.15  # 15% markup
            cost_based_price = unit_costs[sector] * markup
            
            # Demand pressure adjustment
            pressure = demand_pressure.get(sector, 1.0)
            if pressure > 1:  # Excess demand
                price_adjustment = (pressure - 1) * 0.2
            else:  # Excess supply
                price_adjustment = (pressure - 1) * 0.1  # Prices sticky downward
            
            demand_adjusted_price = cost_based_price * (1 + price_adjustment)
            
            # Import competition effect
            import_competition = import_share.get(sector, 0)
            if import_competition > 0.3:  # Significant import competition
                # Prices constrained by world prices
                world_price = 1.0  # Normalized
                final_price = (1 - import_competition) * demand_adjusted_price + \
                            import_competition * world_price
            else:
                final_price = demand_adjusted_price
            
            sectoral_prices[sector] = final_price
        
        return sectoral_prices
    
    def update_price_indices(self,
                           sectoral_prices: Dict[str, float],
                           sectoral_weights: Dict[str, float]) -> None:
        """
        Update aggregate price indices.
        
        Args:
            sectoral_prices: Prices by sector
            sectoral_weights: GDP weights by sector
        """
        # Producer price index (GDP deflator)
        self.price_indices['producer'] = sum(
            price * sectoral_weights.get(sector, 0)
            for sector, price in sectoral_prices.items()
        )
        
        # Consumer price index (simplified)
        # Weighted more toward services and trade
        consumer_weights = {
            'trade': 0.3,
            'services': 0.4,
            'transport': 0.1,
            'other': 0.2
        }
        
        self.price_indices['consumer'] = sum(
            sectoral_prices.get(sector, 1.0) * weight
            for sector, weight in consumer_weights.items()
        )
        
        # Import price index (exogenous under AED peg)
        # Moves with US inflation + oil prices
        us_inflation = 0.02  # 2% assumption
        oil_price_impact = 0.01  # 1% pass-through
        self.price_indices['import'] *= (1 + us_inflation + oil_price_impact)
        
        # Export price index
        # Weighted toward oil and manufacturing
        export_weights = {'oil': 0.6, 'manufacturing': 0.2, 'services': 0.2}
        self.price_indices['export'] = sum(
            sectoral_prices.get(sector, 1.0) * weight
            for sector, weight in export_weights.items()
        )
    
    def calculate_real_exchange_rate(self, 
                                   foreign_price_level: float = 1.0) -> float:
        """
        Calculate real exchange rate (under AED peg).
        
        RER = (E × P*) / P
        where E = nominal rate (fixed at 3.6725 AED/USD), P* = foreign prices, P = domestic prices
        
        Note: The AED has been pegged to USD at 3.6725 since 1997.
        This means nominal exchange rate adjustments are not available as a policy tool.
        
        Args:
            foreign_price_level: Foreign price index
            
        Returns:
            Real exchange rate index
        """
        # Nominal rate is fixed under peg (normalized to 1.0 for index)
        nominal_rate = 1.0  # Fixed at AED/USD = 3.6725
        domestic_prices = self.price_indices['consumer']
        
        rer = (nominal_rate * foreign_price_level) / domestic_prices
        
        return rer
    
    def simulate_inflation_dynamics(self,
                                  demand_shocks: List[float],
                                  supply_shocks: List[float],
                                  periods: int = 10) -> Dict:
        """
        Simulate inflation dynamics over time.
        
        Args:
            demand_shocks: List of demand shocks by period
            supply_shocks: List of supply shocks by period  
            periods: Number of periods
            
        Returns:
            Inflation trajectories
        """
        results = {
            'periods': list(range(periods)),
            'cpi_inflation': [],
            'ppi_inflation': [],
            'wage_inflation': [],
            'rer': []
        }
        
        # Initial values
        cpi = 1.0
        ppi = 1.0
        wages = {'emirati': 20000, 'expat_high': 15000}
        
        for t in range(periods):
            # Demand and supply shocks
            demand_shock = demand_shocks[t] if t < len(demand_shocks) else 0
            supply_shock = supply_shocks[t] if t < len(supply_shocks) else 0
            
            # Price dynamics
            # Phillips curve relationship
            output_gap = demand_shock
            inflation_pressure = 0.3 * output_gap - 0.5 * supply_shock
            
            # Update CPI
            cpi_inflation = 0.02 + inflation_pressure  # 2% base + pressure
            cpi *= (1 + cpi_inflation)
            results['cpi_inflation'].append(cpi_inflation)
            
            # Update PPI (more volatile)
            ppi_inflation = 0.02 + 1.5 * inflation_pressure
            ppi *= (1 + ppi_inflation)
            results['ppi_inflation'].append(ppi_inflation)
            
            # Wage inflation (lagged)
            if t > 0:
                wage_inflation = 0.7 * results['cpi_inflation'][t-1] + 0.3 * output_gap * 0.1
            else:
                wage_inflation = 0.02
            
            for worker_type in wages:
                wages[worker_type] *= (1 + wage_inflation)
            
            results['wage_inflation'].append(wage_inflation)
            
            # Real exchange rate
            rer = self.calculate_real_exchange_rate(1.0 + 0.02)  # 2% foreign inflation
            results['rer'].append(rer)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize price dynamics
    price_system = PriceDynamics()
    
    print("=== WAGE DETERMINATION TEST ===")
    
    # Test wage determination
    scenarios = [
        ("High demand", 110000, 100000),
        ("Balanced", 100000, 100000),
        ("Low demand", 90000, 100000)
    ]
    
    current_wage = 15000
    for name, demand, supply in scenarios:
        new_wage = price_system.determine_equilibrium_wage(
            demand, supply, current_wage, 'expat_high'
        )
        change = (new_wage / current_wage - 1) * 100
        print(f"{name}: Demand={demand:,}, Supply={supply:,}")
        print(f"  New wage: {new_wage:,.0f} AED ({change:+.1f}%)")
    
    print("\n=== SECTORAL PRICE TEST ===")
    
    # Test sectoral pricing
    unit_costs = {
        'manufacturing': 0.85,
        'services': 0.90,
        'trade': 0.88,
        'construction': 0.82
    }
    
    demand_pressure = {
        'manufacturing': 0.95,  # Slight excess supply
        'services': 1.10,       # Excess demand
        'trade': 1.02,
        'construction': 1.15    # High demand
    }
    
    import_share = {
        'manufacturing': 0.45,  # High import competition
        'services': 0.10,
        'trade': 0.35,
        'construction': 0.05
    }
    
    prices = price_system.determine_sectoral_prices(
        unit_costs, demand_pressure, import_share
    )
    
    for sector, price in prices.items():
        print(f"{sector}: {price:.3f}")
    
    print("\n=== INFLATION SIMULATION ===")
    
    # Simulate inflation with oil boom
    demand_shocks = [0.05, 0.08, 0.10, 0.08, 0.05, 0.02, 0, 0, 0, 0]
    supply_shocks = [0, -0.02, -0.03, -0.02, 0, 0, 0, 0, 0, 0]
    
    inflation_sim = price_system.simulate_inflation_dynamics(
        demand_shocks, supply_shocks
    )
    
    print("\nYear  CPI Infl  PPI Infl  Wage Infl  RER")
    print("-" * 45)
    for t in range(10):
        print(f"{t+1:4d}  {inflation_sim['cpi_inflation'][t]:7.1%}  "
              f"{inflation_sim['ppi_inflation'][t]:7.1%}  "
              f"{inflation_sim['wage_inflation'][t]:8.1%}  "
              f"{inflation_sim['rer'][t]:.3f}")
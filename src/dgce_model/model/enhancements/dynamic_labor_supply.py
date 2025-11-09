"""
Dynamic Labor Supply with Entry/Exit for UAE DGCE Model
=======================================================

Implements dynamic labor flows including:
- Expatriate visa-based entry/exit
- Skill-based migration decisions
- Emiratization dynamics
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class VisaParameters:
    """UAE visa and labor mobility parameters."""
    
    # Visa costs by skill level (AED/year)
    visa_costs = {
        'high_skilled': 15000,
        'medium_skilled': 8000,
        'low_skilled': 5000
    }
    
    # Processing times (months)
    processing_time = {
        'new_visa': 2,
        'renewal': 1,
        'cancellation': 0.5
    }
    
    # Visa duration (years)
    visa_duration = {
        'high_skilled': 3,
        'medium_skilled': 2,
        'low_skilled': 2
    }
    
    # Emiratization quotas by sector
    emiratization_quotas = {
        'banking': 0.04,      # 4% annual increase
        'insurance': 0.05,    # 5% annual increase
        'telecom': 0.03,      # 3% annual increase
        'other': 0.02         # 2% annual increase
    }


class DynamicLaborSupply:
    """Dynamic labor supply with entry/exit margins."""
    
    def __init__(self, initial_labor: Dict[str, float]):
        self.params = VisaParameters()
        
        # Initial labor stocks - now with Emirati split
        if 'emirati' in initial_labor and 'emirati_public' not in initial_labor:
            # Split Emirati workforce: 70% public, 30% private
            total_emirati = initial_labor.pop('emirati')
            initial_labor['emirati_public'] = int(total_emirati * 0.70)
            initial_labor['emirati_private'] = int(total_emirati * 0.30)
        
        self.labor_stock = initial_labor.copy()
        
        # Track flows
        self.labor_flows = {
            'entries': [],
            'exits': [],
            'net_flows': []
        }
        
        # Reservation wages (global market) - Based on 2022 data
        self.reservation_wages = {
            'high_skilled': 20000,    # AED/month - competitive with regional markets
            'medium_skilled': 6000,   # Skilled workers baseline
            'low_skilled': 2500       # Above home country wages
        }
        
        # Emirati wages (different for public/private) - Based on 2022 data
        self.emirati_wages = {
            'public': 40000,     # Public sector with Emirati premium
            'private': 30000     # Private sector with Emirati premium
        }
    
    def calculate_entry_probability(self, 
                                  wage: float, 
                                  skill: str,
                                  job_availability: float = 1.0) -> float:
        """
        Calculate probability of labor entry based on wage differential.
        
        Args:
            wage: Offered wage in UAE (AED/month)
            skill: Skill level
            job_availability: Job finding probability (0-1)
            
        Returns:
            Entry probability (0-1)
        """
        # Wage differential
        reservation = self.reservation_wages[skill]
        wage_premium = (wage - reservation) / reservation
        
        # Account for visa costs
        annual_wage = wage * 12
        visa_cost = self.params.visa_costs[skill]
        net_premium = (annual_wage - visa_cost) / (reservation * 12)
        
        # Logistic function for entry probability
        # Higher premium = higher entry probability
        if net_premium > 0:
            entry_prob = 1 / (1 + np.exp(-3 * net_premium))
        else:
            entry_prob = 0.1  # Minimal entry if no premium
        
        # Adjust for job availability
        entry_prob *= job_availability
        
        return min(entry_prob, 0.95)  # Cap at 95%
    
    def calculate_exit_probability(self,
                                 wage: float,
                                 skill: str,
                                 unemployment_rate: float = 0.02) -> float:
        """
        Calculate probability of labor exit.
        
        Args:
            wage: Current wage in UAE
            skill: Skill level
            unemployment_rate: Sector unemployment rate
            
        Returns:
            Exit probability (0-1)
        """
        # Base exit rate (visa expiry, personal reasons)
        base_exit = 0.15  # 15% annual
        
        # Wage-induced exit
        reservation = self.reservation_wages[skill]
        if wage < reservation:
            wage_penalty = (reservation - wage) / reservation
            wage_exit = wage_penalty * 0.3
        else:
            wage_exit = 0
        
        # Unemployment-induced exit
        unemployment_exit = unemployment_rate * 0.5
        
        # Total exit probability
        exit_prob = base_exit + wage_exit + unemployment_exit
        
        return min(exit_prob, 0.50)  # Cap at 50%
    
    def update_labor_flows(self,
                         wages: Dict[str, float],
                         labor_demand: Dict[str, float],
                         time_step: float = 1.0) -> Dict:
        """
        Update labor stocks based on entry/exit flows.
        
        Args:
            wages: Current wages by skill
            labor_demand: Labor demand by skill
            time_step: Time period (1.0 = 1 year)
            
        Returns:
            Updated labor stocks and flows
        """
        entries = {}
        exits = {}
        new_stocks = {}
        
        for skill in ['high_skilled', 'medium_skilled', 'low_skilled']:
            current_stock = self.labor_stock.get(f'expat_{skill}', 0)
            demand = labor_demand.get(skill, current_stock)
            wage = wages.get(skill, self.reservation_wages[skill])
            
            # Job availability based on demand/supply
            if current_stock > 0:
                job_availability = min(demand / current_stock, 1.0)
                unemployment = max(0, 1 - demand / current_stock)
            else:
                job_availability = 1.0
                unemployment = 0
            
            # Calculate flows
            entry_prob = self.calculate_entry_probability(wage, skill, job_availability)
            exit_prob = self.calculate_exit_probability(wage, skill, unemployment)
            
            # Potential entrants pool (simplified)
            potential_entrants = current_stock * 0.2  # 20% of current stock
            
            # Actual flows
            entries[skill] = potential_entrants * entry_prob * time_step
            exits[skill] = current_stock * exit_prob * time_step
            
            # Update stock
            new_stock = current_stock + entries[skill] - exits[skill]
            new_stocks[f'expat_{skill}'] = max(0, new_stock)
        
        # Update Emirati labor (no entry/exit, just participation changes)
        emirati_growth = 0.02  # 2% annual growth
        
        # Handle both old single category and new split categories
        if 'emirati' in self.labor_stock:
            new_stocks['emirati'] = self.labor_stock.get('emirati', 0) * (1 + emirati_growth * time_step)
        else:
            # Update public and private separately
            new_stocks['emirati_public'] = self.labor_stock.get('emirati_public', 0) * (1 + emirati_growth * time_step)
            new_stocks['emirati_private'] = self.labor_stock.get('emirati_private', 0) * (1 + emirati_growth * time_step)
            
            # Allow for some public-private mobility (1% annual)
            mobility_rate = 0.01 * time_step
            public_to_private = new_stocks['emirati_public'] * mobility_rate
            new_stocks['emirati_public'] -= public_to_private
            new_stocks['emirati_private'] += public_to_private
        
        # Record flows
        self.labor_flows['entries'].append(entries)
        self.labor_flows['exits'].append(exits)
        self.labor_flows['net_flows'].append({
            skill: entries[skill] - exits[skill] for skill in entries
        })
        
        # Update stocks
        self.labor_stock = new_stocks
        
        return {
            'stocks': new_stocks.copy(),
            'entries': entries,
            'exits': exits,
            'net_flows': self.labor_flows['net_flows'][-1],
            'total_labor': sum(new_stocks.values())
        }
    
    def apply_emiratization_policy(self, 
                                  sector: str,
                                  current_emirati_share: float) -> Dict:
        """
        Apply Emiratization quota requirements.
        
        Args:
            sector: Economic sector
            current_emirati_share: Current share of Emiratis
            
        Returns:
            Required adjustments
        """
        # Get quota requirement
        annual_increase = self.params.emiratization_quotas.get(sector, 0.02)
        target_share = current_emirati_share + annual_increase
        
        # Calculate required Emirati hiring
        total_employment = sum(self.labor_stock.values())
        current_emiratis = self.labor_stock.get('emirati', 0)
        
        target_emiratis = total_employment * target_share
        required_hiring = max(0, target_emiratis - current_emiratis)
        
        # Calculate implied expat reduction
        expat_reduction = required_hiring * 0.5  # Assume 50% replacement
        
        return {
            'target_emirati_share': target_share,
            'required_emirati_hiring': required_hiring,
            'implied_expat_reduction': expat_reduction,
            'feasible': required_hiring < (current_emiratis * 0.1)  # Max 10% growth
        }
    
    def simulate_wage_shock(self,
                          wage_changes: Dict[str, float],
                          periods: int = 5) -> Dict:
        """
        Simulate labor market response to wage shocks.
        
        Args:
            wage_changes: Percentage wage changes by skill
            periods: Number of years to simulate
            
        Returns:
            Labor market trajectories
        """
        results = {
            'periods': list(range(periods)),
            'labor_stocks': [],
            'net_flows': [],
            'total_employment': []
        }
        
        # Base wages - Based on 2022 sector averages
        base_wages = {
            'high_skilled': 25000,    # Finance/tech/professional avg
            'medium_skilled': 8000,   # Median across all sectors
            'low_skilled': 3000       # Construction/hospitality avg
        }
        
        # Apply shocks
        shocked_wages = {
            skill: wage * (1 + wage_changes.get(skill, 0))
            for skill, wage in base_wages.items()
        }
        
        # Simulate
        for period in range(periods):
            # Gradually adjust wages back to equilibrium
            adjustment_speed = 0.2
            current_wages = {}
            for skill in base_wages:
                base = base_wages[skill]
                shocked = shocked_wages[skill]
                current_wages[skill] = shocked + (base - shocked) * adjustment_speed * period
            
            # Simple labor demand (inversely related to wage)
            labor_demand = {
                skill: self.labor_stock.get(f'expat_{skill}', 100000) * (base_wages[skill] / current_wages[skill]) ** 0.5
                for skill in ['high_skilled', 'medium_skilled', 'low_skilled']
            }
            
            # Update flows
            update = self.update_labor_flows(current_wages, labor_demand)
            
            results['labor_stocks'].append(update['stocks'].copy())
            results['net_flows'].append(update['net_flows'])
            results['total_employment'].append(update['total_labor'])
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize with current labor stocks
    initial_labor = {
        'emirati': 677_300,
        'expat_high_skilled': 2_133_495,
        'expat_medium_skilled': 2_438_280,
        'expat_low_skilled': 1_523_925
    }
    
    labor_market = DynamicLaborSupply(initial_labor)
    
    print("Initial Labor Stocks:")
    for category, count in initial_labor.items():
        print(f"  {category}: {count:,}")
    
    # Test entry/exit probabilities
    print("\nEntry Probabilities at Different Wages:")
    for skill in ['high_skilled', 'medium_skilled', 'low_skilled']:
        for wage_mult in [0.8, 1.0, 1.2, 1.5]:
            wage = labor_market.reservation_wages[skill] * wage_mult
            prob = labor_market.calculate_entry_probability(wage, skill)
            print(f"  {skill} at {wage_mult}x reservation: {prob:.1%}")
    
    # Simulate wage shock
    print("\nSimulating 20% Wage Increase for High-Skilled:")
    shock_results = labor_market.simulate_wage_shock(
        {'high_skilled': 0.20, 'medium_skilled': 0, 'low_skilled': 0}
    )
    
    for i, (stocks, flows) in enumerate(zip(
        shock_results['labor_stocks'],
        shock_results['net_flows']
    )):
        print(f"\nYear {i+1}:")
        print(f"  High-skilled stock: {stocks['expat_high_skilled']:,.0f}")
        print(f"  Net flow: {flows['high_skilled']:+,.0f}")
        print(f"  Total employment: {shock_results['total_employment'][i]:,.0f}")
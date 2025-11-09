"""
Enhanced Labor Categories for DGCE Model
========================================

Implements refined labor categories:
- Split Emiratis: Public vs Private
- Expand Expats: 5 skill levels instead of 3
"""

import numpy as np
from typing import Dict, Tuple


class EnhancedLaborCategories:
    """Enhanced labor categorization with finer granularity."""
    
    def __init__(self):
        # Standard 3-category mapping to 5-category
        self.skill_expansion_map = {
            'high_skilled': {
                'very_high_skilled': 0.20,    # Top 20% - C-suite, specialists
                'high_skilled': 0.80           # Remaining 80% - professionals
            },
            'medium_skilled': {
                'medium_skilled': 1.00         # Keep as single category
            },
            'low_skilled': {
                'low_skilled': 0.80,           # 80% - semi-skilled
                'very_low_skilled': 0.20       # 20% - unskilled
            }
        }
        
        # Wage structure for enhanced categories - Based on 2022 UAE sector data
        self.wage_structure = {
            'emirati_public': 40000,           # Public sector with Emirati premium (2x avg)
            'emirati_private': 30000,          # Private sector with Emirati premium (1.5x avg)
            'expat_very_high_skilled': 80000,  # C-suite executives (3-4x finance/tech avg)
            'expat_high_skilled': 25000,       # Professionals (finance/tech/consulting avg)
            'expat_medium_skilled': 8000,      # Skilled workers (median across sectors)
            'expat_low_skilled': 4000,         # Semi-skilled (hospitality/retail avg)
            'expat_very_low_skilled': 2000     # Unskilled (construction/agriculture avg)
        }
        
        # Visa costs for enhanced categories
        self.visa_costs = {
            'very_high_skilled': 20000,    # Premium visa
            'high_skilled': 15000,
            'medium_skilled': 8000,
            'low_skilled': 5000,
            'very_low_skilled': 3000       # Basic visa
        }
        
        # Labor demand elasticities by category
        self.demand_elasticities = {
            'emirati_public': -0.1,        # Very inelastic (government jobs)
            'emirati_private': -0.3,       # Somewhat elastic
            'expat_very_high_skilled': -0.4,
            'expat_high_skilled': -0.5,
            'expat_medium_skilled': -0.7,
            'expat_low_skilled': -0.9,
            'expat_very_low_skilled': -1.2  # Most elastic
        }
    
    def expand_labor_categories(self, standard_labor: Dict[str, float]) -> Dict[str, float]:
        """
        Convert standard labor categories to enhanced categories.
        
        Args:
            standard_labor: Dictionary with standard categories
            
        Returns:
            Dictionary with enhanced categories
        """
        enhanced = {}
        
        # Handle Emiratis
        if 'emirati' in standard_labor:
            total_emirati = standard_labor['emirati']
            enhanced['emirati_public'] = int(total_emirati * 0.70)
            enhanced['emirati_private'] = int(total_emirati * 0.30)
        elif 'emirati_public' in standard_labor:
            # Already split
            enhanced['emirati_public'] = standard_labor['emirati_public']
            enhanced['emirati_private'] = standard_labor['emirati_private']
        
        # Handle Expats
        for std_category, total in standard_labor.items():
            if std_category.startswith('expat_'):
                skill_level = std_category.replace('expat_', '')
                
                if skill_level in self.skill_expansion_map:
                    splits = self.skill_expansion_map[skill_level]
                    for new_category, share in splits.items():
                        enhanced[f'expat_{new_category}'] = int(total * share)
        
        return enhanced
    
    def calculate_wage_bill(self, labor_stocks: Dict[str, float]) -> float:
        """Calculate total wage bill for enhanced categories."""
        total_wage_bill = 0
        
        for category, count in labor_stocks.items():
            if category in self.wage_structure:
                monthly_wage = self.wage_structure[category]
                annual_wage_bill = count * monthly_wage * 12
                total_wage_bill += annual_wage_bill
        
        return total_wage_bill
    
    def calculate_vat_incidence(self, labor_stocks: Dict[str, float], 
                               vat_rate: float = 0.05) -> Dict[str, float]:
        """
        Calculate VAT burden by labor category.
        
        Different categories have different consumption patterns.
        """
        vat_burden = {}
        
        # Consumption shares of income (lower income = higher share)
        consumption_shares = {
            'emirati_public': 0.60,
            'emirati_private': 0.65,
            'expat_very_high_skilled': 0.50,
            'expat_high_skilled': 0.60,
            'expat_medium_skilled': 0.75,
            'expat_low_skilled': 0.85,
            'expat_very_low_skilled': 0.95
        }
        
        for category, count in labor_stocks.items():
            if category in self.wage_structure:
                wage = self.wage_structure[category]
                consumption_share = consumption_shares.get(category, 0.7)
                
                # Annual consumption
                annual_consumption = wage * 12 * consumption_share
                
                # VAT burden
                vat_paid = annual_consumption * vat_rate
                
                # Per capita burden
                vat_burden[category] = vat_paid * count
        
        return vat_burden
    
    def simulate_emiratization_impact(self, 
                                    current_stocks: Dict[str, float],
                                    emiratization_target: float = 0.02) -> Dict:
        """
        Simulate impact of Emiratization policies.
        
        Args:
            current_stocks: Current labor distribution
            emiratization_target: Annual increase in Emirati share
            
        Returns:
            New distribution and displacement effects
        """
        results = {
            'new_stocks': current_stocks.copy(),
            'displaced_expats': {},
            'wage_impact': {}
        }
        
        # Calculate required Emirati increase
        total_private = sum(count for cat, count in current_stocks.items() 
                          if 'public' not in cat)
        
        current_emirati_private = current_stocks.get('emirati_private', 0)
        current_share = current_emirati_private / total_private if total_private > 0 else 0
        
        target_share = current_share + emiratization_target
        target_emiratis = int(total_private * target_share)
        needed_emiratis = target_emiratis - current_emirati_private
        
        if needed_emiratis > 0:
            # Move some public sector Emiratis to private
            available_public = current_stocks.get('emirati_public', 0) * 0.05  # 5% willing to move
            
            actual_increase = min(needed_emiratis, available_public)
            results['new_stocks']['emirati_public'] -= actual_increase
            results['new_stocks']['emirati_private'] += actual_increase
            
            # Displace expats (mainly medium and low skilled)
            displacement_shares = {
                'expat_very_high_skilled': 0.05,
                'expat_high_skilled': 0.10,
                'expat_medium_skilled': 0.30,
                'expat_low_skilled': 0.35,
                'expat_very_low_skilled': 0.20
            }
            
            for category, share in displacement_shares.items():
                if category in results['new_stocks']:
                    displaced = int(actual_increase * share)
                    results['new_stocks'][category] -= displaced
                    results['displaced_expats'][category] = displaced
            
            # Wage impacts (Emiratization typically increases wages)
            results['wage_impact'] = {
                'emirati_private': 0.05,  # 5% increase due to demand
                'expat_medium_skilled': -0.02,  # Slight decrease
                'expat_low_skilled': -0.03
            }
        
        return results

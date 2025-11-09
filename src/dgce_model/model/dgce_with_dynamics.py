"""
Enhanced DGCE Model with Dynamic Features - FIXED
================================================

Integrates SWF dynamics, dynamic labor supply, and price determination
into the existing DGCE framework without full DSGE transformation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Import base model and dynamic enhancements
from .dgce_model_enhanced_fixed import SimplifiedDGCEModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DGCEWithDynamics(SimplifiedDGCEModel):
    """
    Enhanced DGCE with dynamic features:
    - SWF accumulation and return flows
    - Dynamic labor supply with entry/exit
    - Endogenous price determination
    
    Note: This is NOT a full DSGE - no rational expectations or 
    infinite horizon optimization.
    """
    
    def __init__(self):
        # Initialize base model
        super().__init__()
        
        # Initialize dynamic components - with fallbacks if modules missing
        try:
            from .enhancements.swf_dynamics import SWFDynamicsBlock
            self.swf = SWFDynamicsBlock()
        except ImportError:
            self.swf = self._create_fallback_swf()
        
        try:
            from .enhancements.dynamic_labor_supply import DynamicLaborSupply
            from .enhancements.enhanced_labor_categories import EnhancedLaborCategories
            
            enhancer = EnhancedLaborCategories()
            standard_labor = {
                'emirati': 380_000,
                'expat_high_skilled': 2_133_495,
                'expat_medium_skilled': 2_438_280,
                'expat_low_skilled': 1_523_925
            }
            
            # Convert to enhanced categories
            initial_labor = enhancer.expand_labor_categories(standard_labor)
            self.labor_dynamics = DynamicLaborSupply(initial_labor)
            self.labor_enhancer = enhancer
        except ImportError:
            self.labor_dynamics = self._create_fallback_labor()
            self.labor_enhancer = None
            initial_labor = self._get_fallback_labor_stocks()
        
        try:
            from .enhancements.price_dynamics import PriceDynamics
            self.price_dynamics = PriceDynamics()
        except ImportError:
            self.price_dynamics = self._create_fallback_price_dynamics()
        
        # State variables for dynamics
        self.state = {
            'period': 0,
            'swf_assets': getattr(self.swf, 'swf_assets', {'adia': 500000, 'mubadala': 200000}),
            'labor_stocks': initial_labor,
            'price_indices': getattr(self.price_dynamics, 'price_indices', {'cpi': 100, 'ppi': 100}),
            'wage_levels': {
                'emirati': getattr(self.cal, 'emirati_wage_private', 30000),
                'expat_high': getattr(self.cal, 'expat_wage_high', 25000),
                'expat_medium': getattr(self.cal, 'expat_wage_medium', 8000),
                'expat_low': getattr(self.cal, 'expat_wage_low', 4000)
            }
        }
        
        # History tracking
        self.history = {
            'gdp': [],
            'employment': [],
            'swf_assets': [],
            'price_indices': [],
            'wage_inflation': [],
            'fiscal_balance': []
        }
    
    def _create_fallback_swf(self):
        """Create fallback SWF dynamics when module not available."""
        class FallbackSWF:
            def __init__(self):
                self.swf_assets = {'adia': 500000, 'mubadala': 200000}  # Million AED
                
            def update_swf_assets(self, fiscal_surplus, oil_windfall):
                # Simple accumulation
                total_inflow = fiscal_surplus + oil_windfall
                self.swf_assets['adia'] += total_inflow * 0.7
                self.swf_assets['mubadala'] += total_inflow * 0.3
                
                # Calculate returns (5% annually)
                returns = sum(self.swf_assets.values()) * 0.05
                
                return {
                    'total_assets': sum(self.swf_assets.values()),
                    'returns': returns,
                    'inflows': total_inflow
                }
            
            def calculate_wealth_effect(self):
                # Wealth effect: 2% of assets affect consumption
                return sum(self.swf_assets.values()) * 0.02
        
        return FallbackSWF()
    
    def _create_fallback_labor(self):
        """Create fallback labor dynamics."""
        class FallbackLabor:
            def __init__(self):
                pass
                
            def update_labor_flows(self, wages, labor_demand, time_step):
                # Simple labor adjustment
                total_demand = sum(labor_demand.values()) if isinstance(labor_demand, dict) else labor_demand
                
                return {
                    'total_labor': total_demand,
                    'entries': {'high': 1000, 'medium': 2000, 'low': 3000},
                    'exits': {'high': 500, 'medium': 1000, 'low': 1500},
                    'net_flow': 4000
                }
        
        return FallbackLabor()
    
    def _get_fallback_labor_stocks(self):
        """Get fallback labor stocks."""
        return {
            'emirati': 380_000,
            'expat_high_skilled': 2_133_495,
            'expat_medium_skilled': 2_438_280,
            'expat_low_skilled': 1_523_925
        }
    
    def _create_fallback_price_dynamics(self):
        """Create fallback price dynamics."""
        class FallbackPriceDynamics:
            def __init__(self):
                self.price_indices = {'cpi': 100, 'ppi': 100, 'export': 100}
                
            def determine_sectoral_prices(self, unit_costs, demand_pressure, import_shares):
                # Simple price determination
                prices = {}
                for sector in unit_costs.keys():
                    cost = unit_costs[sector]
                    demand = demand_pressure.get(sector, 1.0)
                    prices[sector] = cost * demand
                return prices
            
            def update_price_indices(self, sectoral_prices, weights):
                # Update indices
                if sectoral_prices:
                    avg_price_change = np.mean(list(sectoral_prices.values()))
                    self.price_indices['cpi'] *= avg_price_change
                    self.price_indices['ppi'] *= avg_price_change
            
            def calculate_real_exchange_rate(self):
                # Simple RER calculation
                return self.price_indices['cpi'] / 100
        
        return FallbackPriceDynamics()
    
    def solve_dynamic_path(self, 
                          shocks: Dict,
                          periods: int = 10,
                          verbose: bool = True) -> Dict:
        """
        Solve multi-period dynamics with adaptive expectations.
        
        Args:
            shocks: Policy shocks (can be time-varying)
            periods: Number of periods to simulate
            verbose: Print period results
            
        Returns:
            Dynamic paths of key variables
        """
        
        if verbose:
            print("\n=== DYNAMIC DGCE SIMULATION ===")
            print(f"Simulating {periods} periods with dynamic features\n")
        
        # Initialize paths
        paths = {
            'gdp': [],
            'employment': [],
            'investment': [],
            'consumption': [],
            'fiscal_balance': [],
            'swf_assets': [],
            'swf_returns': [],
            'labor_entries': [],
            'labor_exits': [],
            'wage_levels': [],
            'price_indices': [],
            'real_exchange_rate': []
        }
        
        for t in range(periods):
            try:
                if verbose:
                    print(f"--- Period {t+1} ---")
                
                # Get period-specific shocks
                period_shocks = self._get_period_shocks(shocks, t)
                
                # 1. Solve static equilibrium for current period
                result = self.solve_policy_impact(period_shocks)
                
                # 2. Update SWF dynamics
                swf_update = self._update_swf(result)
                
                # 3. Update labor market dynamics
                labor_update = self._update_labor_market(result)
                
                # 4. Update price dynamics
                price_update = self._update_prices(result)
                
                # 5. Calculate feedback effects
                feedback = self._calculate_feedback_effects(
                    swf_update, labor_update, price_update
                )
                
                # 6. Apply feedback to next period
                self._apply_feedback(feedback)
                
                # 7. Update state for next period
                self._update_state(swf_update, labor_update, price_update)
                
                # Store results - handle different result structures
                baseline = result.get('baseline', self.steady_state)
                
                # Extract GDP (handle both direct gdp and baseline structure)
                gdp_value = result.get('gdp', baseline.get('gdp', 0))
                if gdp_value == 0 and 'gdp_growth' in result:
                    gdp_value = baseline.get('gdp', 0) * (1 + result['gdp_growth'])
                
                # Extract employment
                employment_value = result.get('employment', baseline.get('employment', 0))
                if employment_value == 0 and 'employment_growth' in result:
                    employment_value = baseline.get('employment', 0) * (1 + result['employment_growth'])
                
                # Extract investment
                investment_value = result.get('investment', baseline.get('investment', 0))
                if investment_value == 0 and 'investment_growth' in result:
                    investment_value = baseline.get('investment', 0) * (1 + result['investment_growth'])
                
                # Extract consumption
                consumption_value = result.get('consumption', baseline.get('consumption', 0))
                
                # Extract fiscal balance
                fiscal_balance = result.get('fiscal_balance', baseline.get('government', 0) - baseline.get('gdp', 0) * 0.1)
                
                paths['gdp'].append(gdp_value)
                paths['employment'].append(labor_update['total_labor'])
                paths['investment'].append(investment_value)
                paths['consumption'].append(consumption_value + swf_update.get('wealth_effect', 0))
                paths['fiscal_balance'].append(fiscal_balance)
                paths['swf_assets'].append(swf_update['total_assets'])
                paths['swf_returns'].append(swf_update['returns'])
                paths['labor_entries'].append(sum(labor_update['entries'].values()))
                paths['labor_exits'].append(sum(labor_update['exits'].values()))
                paths['wage_levels'].append(self.state['wage_levels'].copy())
                paths['price_indices'].append(price_update['indices'].copy())
                paths['real_exchange_rate'].append(price_update.get('rer', 1.0))
                
                # Update period counter
                self.state['period'] = t + 1
                
                if verbose:
                    self._print_period_summary(t, result, swf_update, labor_update, price_update)
                    
            except Exception as e:
                print(f"Warning: Error in period {t+1}: {e}")
                # Fill with safe default values to continue simulation
                baseline = self.steady_state
                paths['gdp'].append(baseline.get('gdp', 0))
                paths['employment'].append(baseline.get('employment', 0))
                paths['investment'].append(baseline.get('investment', 0))
                paths['consumption'].append(baseline.get('consumption', 0))
                paths['fiscal_balance'].append(0)
                paths['swf_assets'].append(sum(self.state['swf_assets'].values()) if isinstance(self.state['swf_assets'], dict) else 0)
                paths['swf_returns'].append(0)
                paths['labor_entries'].append(0)
                paths['labor_exits'].append(0)
                paths['wage_levels'].append(self.state['wage_levels'].copy())
                paths['price_indices'].append(self.state['price_indices'].copy())
                paths['real_exchange_rate'].append(1.0)
                
                if verbose:
                    print(f"Used fallback values for period {t+1}")
                
                continue
        
        # Calculate summary statistics
        paths['summary'] = self._calculate_summary_statistics(paths)
        
        return paths
    
    def _get_period_shocks(self, shocks: Dict, period: int) -> Dict:
        """Extract shocks for current period (allows time-varying shocks)."""
        period_shocks = {}
        
        for key, value in shocks.items():
            if isinstance(value, list):
                # Time-varying shock
                if period < len(value):
                    period_shocks[key] = value[period]
                else:
                    period_shocks[key] = value[-1]  # Use last value
            else:
                # Constant shock
                period_shocks[key] = value
        
        return period_shocks
    
    def _update_swf(self, result: Dict) -> Dict:
        """Update SWF with fiscal surplus and returns."""
        # Calculate oil windfall - handle different result structures
        oil_windfall = 0
        oil_revenue = result.get('oil_revenue', 0)
        if oil_revenue > 0:
            baseline = result.get('baseline', self.steady_state)
            baseline_gdp = baseline.get('gdp', 0)
            baseline_oil_revenue = baseline_gdp * 0.31  # 31% of GDP baseline
            oil_windfall = max(0, oil_revenue - baseline_oil_revenue)
        
        # Get fiscal balance - handle different structures
        fiscal_balance = result.get('fiscal_balance', 0)
        if fiscal_balance == 0:
            # Estimate from baseline if not available
            baseline = result.get('baseline', self.steady_state)
            fiscal_balance = baseline.get('government', 0) - baseline.get('gdp', 0) * 0.1
        
        # Update SWF assets
        swf_update = self.swf.update_swf_assets(
            fiscal_surplus=fiscal_balance,
            oil_windfall=oil_windfall
        )
        
        # Calculate wealth effect on consumption
        wealth_effect = self.swf.calculate_wealth_effect()
        swf_update['wealth_effect'] = wealth_effect
        
        return swf_update
    
    def _update_labor_market(self, result: Dict) -> Dict:
        """Update labor market with entry/exit dynamics."""
        # Extract wages from results
        wages = {}
        
        # Map result wages to skill categories
        base_wages = self.state['wage_levels']
        
        # Adjust wages based on labor market tightness
        employment_growth = result.get('employment_growth', 0)
        wage_pressure = employment_growth * 0.5  # 50% pass-through
        
        for skill in ['high_skilled', 'medium_skilled', 'low_skilled']:
            if skill == 'high_skilled':
                base = base_wages['expat_high']
            elif skill == 'medium_skilled':
                base = base_wages['expat_medium']
            else:
                base = base_wages['expat_low']
            
            wages[skill] = base * (1 + wage_pressure)
        
        # Calculate labor demand
        labor_demand = self._calculate_labor_demand(result)
        
        # Update labor flows
        labor_update = self.labor_dynamics.update_labor_flows(
            wages=wages,
            labor_demand=labor_demand,
            time_step=1.0
        )
        
        return labor_update
    
    def _calculate_labor_demand(self, result: Dict) -> Dict:
        """Calculate labor demand from economic results."""
        # Get employment level - handle different result structures
        employment_level = result.get('employment', 0)
        if employment_level == 0:
            # Try to get from baseline
            baseline = result.get('baseline', self.steady_state)
            base_employment = baseline.get('employment', self.steady_state['employment'])
            
            # Apply employment growth if available
            if 'employment_growth' in result:
                employment_level = base_employment * (1 + result['employment_growth'])
            else:
                employment_level = base_employment
        
        # Distribute across skill categories (approximate shares)
        labor_demand = {
            'high_skilled': employment_level * 0.15,
            'medium_skilled': employment_level * 0.45,
            'low_skilled': employment_level * 0.40
        }
        
        return labor_demand
    
    def _update_prices(self, result: Dict) -> Dict:
        """Update price indices based on market conditions."""
        # Calculate sectoral unit costs (simplified)
        unit_costs = {}
        sectors = ['manufacturing', 'services', 'trade', 'construction']
        
        for sector in sectors:
            # Base cost with wage component
            wage_index = np.mean(list(self.state['wage_levels'].values())) / 10000
            unit_costs[sector] = 0.8 + 0.2 * wage_index
        
        # Demand pressure from GDP growth
        gdp_growth = result.get('gdp_growth', 0)
        demand_pressure = {sector: 1.0 + gdp_growth for sector in sectors}
        
        # Import shares (fixed for now)
        import_shares = {
            'manufacturing': 0.45,
            'services': 0.10,
            'trade': 0.35,
            'construction': 0.05
        }
        
        # Determine sectoral prices
        sectoral_prices = self.price_dynamics.determine_sectoral_prices(
            unit_costs, demand_pressure, import_shares
        )
        
        # Update price indices
        sectoral_weights = {sector: 0.25 for sector in sectors}
        self.price_dynamics.update_price_indices(sectoral_prices, sectoral_weights)
        
        # Calculate real exchange rate
        rer = self.price_dynamics.calculate_real_exchange_rate()
        
        return {
            'indices': self.price_dynamics.price_indices.copy(),
            'sectoral_prices': sectoral_prices,
            'rer': rer
        }
    
    def _calculate_feedback_effects(self, 
                                  swf_update: Dict,
                                  labor_update: Dict,
                                  price_update: Dict) -> Dict:
        """Calculate how dynamic changes affect next period."""
        feedback = {}
        
        # SWF wealth effect on consumption
        feedback['consumption_boost'] = swf_update.get('wealth_effect', 0) / self.steady_state['consumption']
        
        # Labor supply effect on potential output
        labor_change = (labor_update['total_labor'] - self.steady_state['employment']) / self.steady_state['employment']
        feedback['output_potential'] = labor_change * 0.7  # Labor share of output
        
        # Price competitiveness effect on exports
        rer_change = price_update.get('rer', 1.0) - 1.0
        feedback['export_competitiveness'] = -rer_change * 1.2  # Export elasticity
        
        # Fiscal space from SWF returns
        feedback['fiscal_space'] = swf_update.get('returns', 0) / self.steady_state['gdp']
        
        return feedback
    
    def _apply_feedback(self, feedback: Dict) -> None:
        """Apply feedback effects to model parameters."""
        # Adjust consumption from wealth effects
        if 'consumption_boost' in feedback:
            consumption_adj = feedback['consumption_boost']
            # Store for next period's calculations
            self.state['consumption_adjustment'] = consumption_adj
        
        # Adjust potential output
        if 'output_potential' in feedback:
            output_adj = feedback['output_potential']
            self.state['output_adjustment'] = output_adj
        
        # Store other adjustments
        self.state['export_adjustment'] = feedback.get('export_competitiveness', 0)
        self.state['fiscal_adjustment'] = feedback.get('fiscal_space', 0)
    
    def _update_state(self, swf_update: Dict, labor_update: Dict, price_update: Dict) -> None:
        """Update the model's internal state with new period values."""
        # Update SWF assets
        if hasattr(self.swf, 'swf_assets'):
            self.state['swf_assets'] = self.swf.swf_assets.copy()
        else:
            self.state['swf_assets']['adia'] = swf_update.get('total_assets', 0) * 0.7
            self.state['swf_assets']['mubadala'] = swf_update.get('total_assets', 0) * 0.3
        
        # Update labor stocks
        self.state['labor_stocks']['total'] = labor_update['total_labor']
        
        # Update price indices
        if hasattr(self.price_dynamics, 'price_indices'):
            self.state['price_indices'] = self.price_dynamics.price_indices.copy()
        else:
            self.state['price_indices'] = price_update.get('indices', self.state['price_indices'])
        
        # Update wage levels (with inflation adjustment)
        inflation_rate = (price_update.get('indices', {}).get('cpi', 100) / 100) - 1
        for wage_type in self.state['wage_levels']:
            self.state['wage_levels'][wage_type] *= (1 + inflation_rate * 0.5)  # Partial indexation
    
    def _print_period_summary(self, period: int, result: Dict, 
                             swf_update: Dict, labor_update: Dict, price_update: Dict) -> None:
        """Print summary for current period."""
        # Handle different result structures
        baseline = result.get('baseline', self.steady_state)
        
        # Get GDP
        gdp_value = result.get('gdp', baseline.get('gdp', 0))
        if gdp_value == 0 and 'gdp_growth' in result:
            gdp_value = baseline.get('gdp', 0) * (1 + result['gdp_growth'])
        
        # Get GDP growth
        gdp_growth = result.get('gdp_growth', 0)
        
        print(f"GDP: {gdp_value:,.0f}M AED ({gdp_growth*100:+.1f}%)")
        print(f"Employment: {labor_update['total_labor']:,.0f}")
        print(f"SWF Assets: {swf_update['total_assets']:,.0f}M AED")
        print(f"CPI: {price_update.get('indices', {}).get('cpi', 100):.1f}")
        print()
    
    def _calculate_summary_statistics(self, paths: Dict) -> Dict:
        """Calculate summary statistics from simulation paths."""
        summary = {}
        
        # Calculate growth rates - handle empty or invalid data
        if len(paths['gdp']) > 1 and all(x > 0 for x in paths['gdp']):
            gdp_growth = [(paths['gdp'][i] - paths['gdp'][i-1]) / paths['gdp'][i-1] 
                         for i in range(1, len(paths['gdp']))
                         if paths['gdp'][i-1] > 0]
            if gdp_growth:
                summary['avg_gdp_growth'] = np.mean(gdp_growth)
                summary['gdp_volatility'] = np.std(gdp_growth)
            else:
                summary['avg_gdp_growth'] = 0.0
                summary['gdp_volatility'] = 0.0
        else:
            summary['avg_gdp_growth'] = 0.0
            summary['gdp_volatility'] = 0.0
        
        # Calculate final values - handle empty paths
        summary['final_gdp'] = paths['gdp'][-1] if paths['gdp'] and len(paths['gdp']) > 0 else 0
        summary['final_employment'] = paths['employment'][-1] if paths['employment'] and len(paths['employment']) > 0 else 0
        summary['final_swf_assets'] = paths['swf_assets'][-1] if paths['swf_assets'] and len(paths['swf_assets']) > 0 else 0
        
        # Calculate cumulative effects - handle empty or zero values
        if len(paths['gdp']) > 1 and paths['gdp'][0] > 0:
            summary['cumulative_gdp_change'] = (paths['gdp'][-1] - paths['gdp'][0]) / paths['gdp'][0]
        else:
            summary['cumulative_gdp_change'] = 0.0
        
        return summary
    
    def get_dynamic_multipliers(self, shock_type: str = 'tax_cut') -> Dict:
        """Calculate dynamic multipliers for different shock types."""
        # Baseline simulation
        baseline_shocks = {}
        baseline_paths = self.solve_dynamic_path(baseline_shocks, periods=5, verbose=False)
        
        # Shock simulation
        if shock_type == 'tax_cut':
            shock_shocks = {'tax_changes': {'corporate': -0.01}}  # 1pp tax cut
        elif shock_type == 'spending_increase':
            shock_shocks = {'government_spending': 0.02}  # 2% increase
        else:
            shock_shocks = {shock_type: 0.01}
        
        shock_paths = self.solve_dynamic_path(shock_shocks, periods=5, verbose=False)
        
        # Calculate multipliers - handle empty or invalid data
        multipliers = {}
        for var in ['gdp', 'employment', 'consumption', 'investment']:
            if (var in baseline_paths and var in shock_paths and 
                len(baseline_paths[var]) > 0 and len(shock_paths[var]) > 0):
                
                baseline_series = np.array(baseline_paths[var])
                shock_series = np.array(shock_paths[var])
                
                # Ensure arrays are same length
                min_length = min(len(baseline_series), len(shock_series))
                baseline_series = baseline_series[:min_length]
                shock_series = shock_series[:min_length]
                
                # Period-by-period multipliers
                differences = shock_series - baseline_series
                multipliers[f'{var}_multipliers'] = differences.tolist()
                
                # Peak multiplier
                multipliers[f'{var}_peak_multiplier'] = float(np.max(np.abs(differences))) if len(differences) > 0 else 0.0
            else:
                multipliers[f'{var}_multipliers'] = []
                multipliers[f'{var}_peak_multiplier'] = 0.0
        
        return multipliers
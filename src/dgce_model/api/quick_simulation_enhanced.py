"""
Enhanced Quick Policy Simulation Using DGCE Model Only (FIXED)
=============================================================

FIXES:
- Proper integration with fixed DGCE model
- Better error handling and validation
- Realistic parameter bounds and calculations
- Consistent use of corporate tax simulation functions
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List

# Import the enhanced DGCE model
from dgce_model.model.dgce_model_enhanced_fixed import SimplifiedDGCEModel
from dgce_model.openfisca_runner import run_corporate_tax_simulation, validate_simulation_results
from dgce_model.data_loader.new_data_loader import RealUAEDataLoader


class EnhancedQuickSimulation:
    """Enhanced quick simulation using only the DGCE model (FIXED)."""
    
    def __init__(self):
        """Initialize with DGCE model."""
        print("🚀 Initializing Enhanced Quick Simulation...")
        
        try:
            # Initialize DGCE model
            self.dgce_model = SimplifiedDGCEModel()
            # Load company and sectoral data for microsimulation
            self.data_loader = RealUAEDataLoader()
            
            print("✅ Enhanced Quick Simulation initialized with DGCE model")
        except Exception as e:
            print(f"⚠️ Error initializing Enhanced Quick Simulation: {e}")
            raise
        
    def run_simulation(self, params: Dict) -> Dict:
        """
        Run quick policy simulation using the DGCE model exclusively.
        FIXED: Better validation and realistic calculations.
        
        Parameters:
        -----------
        params : dict
            Policy parameters including:
            - standard_rate: Corporate tax rate (0-1)
            - threshold: Tax-free threshold
            - small_biz_threshold: Small business relief threshold
            - oil_gas_rate: Oil & gas sector rate
            - fz_qualifying_rate: Free zone qualifying income rate
            - sme_election_rate: SME election rate
            - compliance_rate: Expected compliance rate
        """
        
        try:
            # Validate and extract parameters
            standard_rate = self._validate_rate(params.get('standard_rate'), 'standard_rate')
            threshold = self._validate_amount(params.get('threshold'), 'threshold', default=375000)
            small_biz_threshold = self._validate_amount(params.get('small_biz_threshold'), 'small_biz_threshold', default=3000000)
            oil_gas_rate = self._validate_rate(params.get('oil_gas_rate'), 'oil_gas_rate', default=0.55)
            fz_rate = self._validate_rate(params.get('fz_qualifying_rate', 0.0), 'fz_qualifying_rate', default=0.0)
            sme_rate = self._validate_rate(params.get('sme_election_rate', 0.80), 'sme_election_rate', default=0.80)
            compliance = self._validate_rate(params.get('compliance_rate'), 'compliance_rate')

            # New levers
            vat_rate = self._validate_rate(params.get('vat_rate', 0.05), 'vat_rate', default=0.05)
            g_change = float(params.get('government_spending_rel_change', 0.0))
            years = int(params.get('years', 5))
            
            print(f"Running simulation: Tax rate {standard_rate*100:.1f}%, Compliance {compliance*100:.1f}%")
            
            # Build policy parameters for DGCE model
            policy_params = {
                'standard_rate': standard_rate,
                'oil_gas_rate': oil_gas_rate,
                'vat_rate': vat_rate,
                'compliance': compliance,
                'threshold': threshold,
                'small_biz_threshold': small_biz_threshold,
                'government_spending_rel_change': g_change,
                'incentives': {}
            }
            
            # Run DGCE simulation – obtain both static impacts (apply_scenario)
            # *and* dynamic path via the new `simulate` interface.

            # 1. Static snapshot for backward-compatibility
            dgce_results = self.dgce_model.apply_scenario(policy_params)

            # 2. Dynamic 5-year path (uses elasticity-based engine)
            try:
                path_df = self.dgce_model.simulate(
                    {
                        "corporate_tax_rate": standard_rate,
                        "vat_rate": vat_rate,
                        "government_spending_rel_change": g_change,
                    },
                    years=years,
                )
            except Exception as e:
                print(f"⚠️ Dynamic simulation failed: {e}")
                path_df = None
            
            # Run microsimulation for detailed tax calculation
            micro_params = {
                "standard_rate": standard_rate,
                "free_zone_rate": fz_rate,
                "oil_gas_rate": oil_gas_rate,
                "small_business_threshold": small_biz_threshold,
                "profit_allowance": threshold
            }

            micro_results = run_corporate_tax_simulation(
                companies=self.data_loader.commerce_registry,
                sectoral_panel=self.data_loader.sectoral_panel,
                params=micro_params
            )
            
            # Validate microsimulation results
            validation = validate_simulation_results(micro_results)
            
            # Extract revenue analysis from DGCE results
            revenue_analysis = dgce_results.get('revenue_analysis', {})
            
            # Calculate key metrics
            # Prefer the DGCE-calculated corporate component (already compliance-adjusted)
            if 'corporate_component' in revenue_analysis:
                total_tax_revenue = float(revenue_analysis['corporate_component'])
            else:
                # Fallback: discount microsim potential tax by assumed compliance
                total_tax_revenue_aed = float(micro_results['corporate_tax'].sum()) * float(compliance)
                total_tax_revenue = total_tax_revenue_aed / 1_000_000  # millions AED (post-compliance)
            avg_effective_rate = float(micro_results['effective_tax_rate'].mean())
            total_companies = len(micro_results)
            taxable_companies = len(micro_results[micro_results['corporate_tax'] > 0])
            
            # Calculate revenue changes
            baseline_revenue = revenue_analysis.get('baseline_revenue', 0)
            revenue_change = revenue_analysis.get('revenue_change', 0)
            revenue_change_pct = revenue_analysis.get('revenue_change_pct', 0)
            
            # Prepare sector analysis
            # sector_analysis = self._analyze_sectors(micro_results, dgce_results.get('sectoral_impacts', {}))
            
            # Calculate compliance metrics
            compliance_metrics = self._calculate_compliance_metrics(
                micro_results, compliance, standard_rate
            )
            
            # Prepare final results
            results = {
                # Core metrics
                'corporate_tax_rate_pct': standard_rate * 100,
                'micro_tax_revenue': total_tax_revenue,
                'revenue_change_m': revenue_change,
                'revenue_change_pct': revenue_change_pct,
                'avg_effective_rate_pct': avg_effective_rate * 100,
                
                # Economic impacts from DGCE
                'gdp_impact_pct': dgce_results.get('gdp_impact', 0),
                'employment_impact_pct': dgce_results.get('employment_impact', 0),
                'investment_impact_pct': dgce_results.get('investment_impact', 0),
                'consumption_impact_pct': dgce_results.get('consumption_impact', 0),
                
                # Company statistics
                'total_companies': total_companies,
                'taxable_companies': taxable_companies,
                'tax_base_companies_pct': (taxable_companies / total_companies * 100) if total_companies > 0 else 0,
                
                # Detailed analysis
                # 'sector_analysis': sector_analysis,
                'compliance_metrics': compliance_metrics,
                'revenue_analysis': revenue_analysis,
                
                # Validation
                'simulation_valid': validation['validation_passed'],
                'validation_details': validation,
                
                # Metadata
                'simulation_timestamp': datetime.now().isoformat(),
                'parameters_used': policy_params,
                'dynamic_path': None if path_df is None else path_df.to_dict(orient='list')
            }
            
            print(f"✅ Simulation complete: Tax revenue = AED {total_tax_revenue:.1f}M, "
                  f"GDP impact = {dgce_results.get('gdp_impact', 0):.2f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in simulation: {str(e)}")
            return self._create_error_response(str(e), params)
    
    def _validate_rate(self, value, param_name: str, default: float = None) -> float:
        """Validate tax rate parameter."""
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"{param_name} is required")
        
        try:
            rate = float(value)
            if not (0 <= rate <= 1):
                raise ValueError(f"{param_name} must be between 0 and 1")
            return rate
        except (TypeError, ValueError):
            raise ValueError(f"{param_name} must be a valid number between 0 and 1")
    
    def _validate_amount(self, value, param_name: str, default: float = None) -> float:
        """Validate monetary amount parameter."""
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"{param_name} is required")
        
        try:
            amount = float(value)
            if amount < 0:
                raise ValueError(f"{param_name} must be non-negative")
            return amount
        except (TypeError, ValueError):
            raise ValueError(f"{param_name} must be a valid non-negative number")
    
    def _analyze_sectors(self, micro_results: Dict, sectoral_impacts: Dict) -> Dict:
        """Analyze sector-specific results."""
        if micro_results.empty:
            return {}
        
        # Group by sector
        sector_groups = micro_results.groupby('ISIC_level_1').agg({
            'corporate_tax': ['sum', 'mean', 'count'],
            'effective_tax_rate': 'mean',
            'revenue': 'sum'
        }).round(2)
        
        sector_analysis = {}
        for sector in sector_groups.index:
            tax_sum = sector_groups.loc[sector, ('corporate_tax', 'sum')]
            tax_mean = sector_groups.loc[sector, ('corporate_tax', 'mean')]
            company_count = sector_groups.loc[sector, ('corporate_tax', 'count')]
            avg_rate = sector_groups.loc[sector, ('effective_tax_rate', 'mean')]
            revenue_sum = sector_groups.loc[sector, ('revenue', 'sum')]
            
            sector_analysis[sector] = {
                'total_tax': float(tax_sum),
                'avg_tax_per_company': float(tax_mean),
                'company_count': int(company_count),
                'avg_effective_rate': float(avg_rate),
                'total_revenue': float(revenue_sum),
                'dgce_impact': sectoral_impacts.get(sector, 0)
            }
        
        return sector_analysis
    
    def _calculate_compliance_metrics(self, micro_results: Dict, compliance_rate: float, tax_rate: float) -> Dict:
        """Calculate compliance-related metrics.

        Definitions (consistent with tax logic):
        - total_potential_tax: sum of pre-compliance corporate tax liabilities
          at statutory rules (i.e., Σ corporate_tax_i from microsim, which is
          based on taxable_profit and firm-specific applicable rates).
        - actual_collections: potential tax discounted by the assumed
          compliance rate (uniform in this quick view).
        """
        if micro_results.empty:
            return {}

        # Prefer pre-compliance corporate tax from microsim as the potential tax
        if 'corporate_tax' in micro_results.columns:
            total_potential_tax = float(micro_results['corporate_tax'].sum())
        else:
            # Fallback: derive from taxable_profit if present, else last-resort from revenue
            if 'taxable_profit' in micro_results.columns:
                total_potential_tax = float(micro_results['taxable_profit'].sum()) * float(tax_rate)
            else:
                total_potential_tax = float(micro_results.get('revenue', 0).sum()) * float(tax_rate)

        # Apply uniform compliance to estimate actual collections
        actual_tax = total_potential_tax * float(compliance_rate)

        # convert to millions for readability
        total_potential_tax_m = total_potential_tax / 1_000_000
        actual_tax_m = actual_tax / 1_000_000
        compliance_gap_m = max(0.0, total_potential_tax - actual_tax) / 1_000_000
        
        return {
            'expected_compliance_rate': compliance_rate * 100,
            'implied_compliance_rate': (actual_tax / total_potential_tax * 100) if total_potential_tax > 0 else 0,
            'compliance_gap_m': compliance_gap_m,
            'total_potential_tax_m': total_potential_tax_m,
            'actual_collections_m': actual_tax_m
        }
    
    def _create_error_response(self, error_message: str, params: Dict) -> Dict:
        """Create error response with safe defaults."""
        return {
            'error': True,
            'error_message': error_message,
            'corporate_tax_rate_pct': params.get('standard_rate', 0) * 100,
            'micro_tax_revenue': 0,
            'revenue_change_m': 0,
            'revenue_change_pct': 0,
            'avg_effective_rate_pct': 0,
            'gdp_impact_pct': 0,
            'employment_impact_pct': 0,
            'investment_impact_pct': 0,
            'consumption_impact_pct': 0,
            'total_companies': 0,
            'taxable_companies': 0,
            'tax_base_companies_pct': 0,
            'simulation_valid': False,
            'simulation_timestamp': datetime.now().isoformat(),
            'parameters_used': params
        }
    
    def get_simulation_bounds(self) -> Dict:
        """Get realistic bounds for simulation parameters."""
        return {
            'standard_rate': {'min': 0.00, 'max': 0.30, 'default': 0.09},
            'threshold': {'min': 0, 'max': 1_000_000, 'default': 375_000},
            'small_biz_threshold': {'min': 1_000_000, 'max': 10_000_000, 'default': 3_000_000},
            'oil_gas_rate': {'min': 0.00, 'max': 0.80, 'default': 0.55},
            'fz_qualifying_rate': {'min': 0.00, 'max': 1.00, 'default': 0.80},
            'compliance_rate': {'min': 0.30, 'max': 1.00, 'default': 0.75}
        }

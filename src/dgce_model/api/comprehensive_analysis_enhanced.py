"""
Comprehensive Policy Analysis Using DGCE Model (FIXED)
=====================================================

FIXES:
- Proper integration with fixed DGCE model and data loader
- Better error handling and validation
- Realistic calculations and parameter bounds
- Consistent use of microsimulation functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import the enhanced DGCE model
from dgce_model.model.dgce_model_enhanced_fixed import SimplifiedDGCEModel
from dgce_model.openfisca_runner import run_corporate_tax_simulation, validate_simulation_results
from dgce_model.data_loader.new_data_loader import RealUAEDataLoader

# Import dynamic capabilities if available
try:
    from dgce_model.model.dgce_model_enhanced_fixed import SimplifiedDGCEModel
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False
    print("⚠️ Dynamic DGCE not available")

# Import sector mapping
try:
    from dgce_model.sector_mapping import ISIC_SECTORS, get_policy_rate_for_sector
except ImportError:
    ISIC_SECTORS = [
        'Financial and insurance activities',
        'Manufacturing', 
        'Mining and quarrying',
        'Real estate activities',
        'Wholesale and retail trade, repair of motor vehicles and motorcycles',
        'Construction',
        'Information and communication',
        'Professional, scientific and technical activities'
    ]


class ComprehensivePolicyAnalyzer:
    """
    Comprehensive policy analysis system that uses only the DGCE model (FIXED).
    All calculations go through the core DGCE framework.
    """
    
    def __init__(self):
        """Initialize with DGCE model components."""
        print("🚀 Initializing Comprehensive Policy Analyzer...")
        
        try:
            # Initialize core DGCE model
            self.dgce_model = SimplifiedDGCEModel()
            # Load data for microsimulation
            self.data_loader = RealUAEDataLoader()
            
            # Initialize dynamic model if available
            if DYNAMICS_AVAILABLE:
                try:
                    self.dgce_dynamic = SimplifiedDGCEModel()
                    self.has_dynamics = True
                    print("✅ Dynamic DGCE model available")
                except Exception as e:
                    print(f"⚠️ Dynamic DGCE failed to initialize: {e}")
                    self.has_dynamics = False
            else:
                self.has_dynamics = False
            
            print("✅ Comprehensive Policy Analyzer initialized")
        except Exception as e:
            print(f"❌ Error initializing Comprehensive Policy Analyzer: {e}")
            raise
    
    def analyze_comprehensive_policy(self, policy_params: Dict) -> Dict:
        """
        Run comprehensive policy analysis using only the DGCE model.
        FIXED: Better validation and realistic calculations.
        """
        
        try:
            print("🔍 Starting comprehensive policy analysis...")
            
            # Validate parameters
            validated_params = self._validate_policy_params(policy_params)
            
            print(f"Analyzing policy: Tax rate {validated_params['standard_rate']*100:.1f}%, "
                  f"Compliance {validated_params['compliance']*100:.1f}%")
            
            # Run static analysis with DGCE model
            static_result = self._run_static_analysis(validated_params)
            
            # Run microsimulation analysis
            micro_result = self._run_microsimulation_analysis(validated_params)
            
            # Run sector analysis
            sector_result = self._run_sector_analysis(validated_params)
            
            # Calculate revenue analysis
            revenue_analysis = self._calculate_comprehensive_revenue_analysis(
                validated_params, static_result, micro_result
            )
            
            # Run risk assessment
            risk_assessment = self._comprehensive_risk_assessment(
                static_result, validated_params['compliance']
            )
            
            # Dynamic analysis via new PolicyShockEngine interface
            dynamic_result = {}
            try:
                years = int(validated_params.get('time_horizon', 5))
                dynamic_path = self.dgce_model.simulate(
                    {
                        "corporate_tax_rate": validated_params['standard_rate'],
                        "vat_rate": validated_params['vat_rate'],
                        "government_spending_rel_change": validated_params['government_spending_rel_change'],
                    },
                    years=years,
                )
                dynamic_result = {
                    'years': years,
                    'path': dynamic_path.to_dict(orient='list')
                }
            except Exception as e:
                print(f"⚠️ Dynamic path generation failed: {e}")
            
            # Compile comprehensive results
            comprehensive_results = {
                # Core economic impacts
                'gdp_impact': static_result.get('gdp_impact', 0),
                'employment_impact': static_result.get('employment_impact', 0),
                'investment_impact': static_result.get('investment_impact', 0),
                'consumption_impact': static_result.get('consumption_impact', 0),
                
                # Revenue analysis
                'revenue_analysis': revenue_analysis,
                
                # Microsimulation results
                'microsimulation': {
                    'total_tax_revenue': micro_result.get('total_tax_revenue', 0),
                    'average_effective_rate': micro_result.get('average_effective_rate', 0),
                    'companies_affected': micro_result.get('companies_affected', 0),
                    'validation_passed': micro_result.get('validation_passed', False)
                },
                
                # Sector analysis
                'sectoral_impacts': sector_result.get('sectoral_impacts', {}),
                'sector_distribution': sector_result.get('sector_distribution', {}),
                
                # Risk assessment
                'risk_assessment': risk_assessment,
                
                # Dynamic results
                'dynamic_impacts': dynamic_result,
                
                # Oil sector (important for UAE)
                'oil_sector': static_result.get('oil_sector', {}),
                
                # Policy parameters used
                'policy_parameters': validated_params,
                
                # Metadata
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': 'comprehensive',
                'model_version': 'enhanced_fixed'
            }
            
            print(f"✅ Comprehensive analysis complete: "
                  f"GDP impact {static_result.get('gdp_impact', 0):.2f}%, "
                  f"Revenue change AED {revenue_analysis.get('revenue_change', 0):.1f}M")
            
            return comprehensive_results
            
        except Exception as e:
            print(f"❌ Error in comprehensive policy analysis: {str(e)}")
            return self._create_error_response(str(e), policy_params)
    
    def _validate_policy_params(self, params: Dict) -> Dict:
        """Validate and normalize policy parameters."""
        validated = {}
        
        # Required parameters with validation
        validated['standard_rate'] = self._validate_rate(
            params.get('standard_rate'), 'standard_rate', min_val=0.0, max_val=0.5
        )
        validated['compliance'] = self._validate_rate(
            params.get('compliance'), 'compliance', min_val=0.1, max_val=1.0
        )
        
        # Optional parameters with defaults
        validated['threshold'] = max(0, float(params.get('threshold', 375_000)))
        validated['small_biz_threshold'] = max(1000000, float(params.get('small_biz_threshold', 3_000_000)))
        validated['oil_gas_rate'] = self._validate_rate(
            params.get('oil_gas_rate', 0.55), 'oil_gas_rate', min_val=0.0, max_val=0.8
        )
        validated['vat_rate'] = self._validate_rate(
            params.get('vat_rate', 0.05), 'vat_rate', min_val=0.0, max_val=0.3
        )
        validated['government_spending_rel_change'] = float(
            params.get('government_spending_rel_change', 0.0)
        )
        validated['time_horizon'] = max(1, int(params.get('time_horizon', 10)))
        
        # Incentives
        validated['incentives'] = params.get('incentives', {})
        
        return validated
    
    def _validate_rate(self, value, param_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate rate parameter."""
        if value is None:
            raise ValueError(f"{param_name} is required")
        
        try:
            rate = float(value)
            if not (min_val <= rate <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            return rate
        except (TypeError, ValueError):
            raise ValueError(f"{param_name} must be a valid number between {min_val} and {max_val}")
    
    def _run_static_analysis(self, policy_params: Dict) -> Dict:
        """Run static DGCE analysis."""
        try:
            return self.dgce_model.apply_scenario(policy_params)
        except Exception as e:
            print(f"⚠️ Error in static analysis: {e}")
            return {'error': str(e)}
    
    def _run_microsimulation_analysis(self, policy_params: Dict) -> Dict:
        """Run detailed microsimulation analysis."""
        try:
            # Prepare microsimulation parameters
            micro_params = {
                "standard_rate": policy_params['standard_rate'],
                "free_zone_rate": 0.0,
                "oil_gas_rate": policy_params['oil_gas_rate'],
                "small_business_threshold": policy_params['small_biz_threshold'],
                "profit_allowance": policy_params['threshold']
            }

            # Run microsimulation
            micro_results = run_corporate_tax_simulation(
                companies=self.data_loader.commerce_registry,
                sectoral_panel=self.data_loader.sectoral_panel,
                params=micro_params
            )
            
            # Validate results
            validation = validate_simulation_results(micro_results)
            
            # Calculate summary statistics
            total_tax_revenue = float(micro_results['corporate_tax'].sum())
            average_effective_rate = float(micro_results['effective_tax_rate'].mean())
            companies_affected = len(micro_results[micro_results['corporate_tax'] > 0])
            
            return {
                'total_tax_revenue': total_tax_revenue,
                'average_effective_rate': average_effective_rate,
                'companies_affected': companies_affected,
                'total_companies': len(micro_results),
                'validation_passed': validation['validation_passed'],
                'validation_details': validation,
                'results_df': micro_results  # Keep for further analysis
            }
            
        except Exception as e:
            print(f"⚠️ Error in microsimulation: {e}")
            return {'error': str(e), 'validation_passed': False}
    
    def _run_sector_analysis(self, policy_params: Dict) -> Dict:
        """Run sector-specific analysis."""
        try:
            # Get sectoral impacts from DGCE
            dgce_results = self.dgce_model.solve_policy_impact({
                'tax_changes': {
                    'corporate': policy_params['standard_rate'],
                    'oil': policy_params['oil_gas_rate'],
                    'vat': policy_params['vat_rate']
                },
                'compliance_rate': policy_params['compliance']
            })
            
            sectoral_impacts = dgce_results.get('sectoral_impacts', {})
            
            # Calculate sector distribution from data
            sector_distribution = {}
            if hasattr(self.data_loader, 'commerce_registry'):
                registry = self.data_loader.commerce_registry
                sector_counts = registry['ISIC_level_1'].value_counts()
                total_companies = len(registry)
                
                for sector, count in sector_counts.items():
                    sector_distribution[sector] = {
                        'company_count': int(count),
                        'percentage': float(count / total_companies * 100),
                        'dgce_impact': sectoral_impacts.get(sector, 0)
                    }
            
            return {
                'sectoral_impacts': sectoral_impacts,
                'sector_distribution': sector_distribution
            }
            
        except Exception as e:
            print(f"⚠️ Error in sector analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_comprehensive_revenue_analysis(
        self, policy_params: Dict, static_result: Dict, micro_result: Dict
    ) -> Dict:
        """Calculate comprehensive revenue analysis."""
        try:
            # Extract revenue analysis from DGCE results
            revenue_analysis = static_result.get('revenue_analysis', {})
            
            # If not available, calculate from microsimulation
            if not revenue_analysis and micro_result.get('validation_passed'):
                micro_tax = micro_result.get('total_tax_revenue', 0)
                
                # Estimate baseline (9% rate at 75% compliance)
                baseline_rate = 0.09
                baseline_compliance = 0.75
                current_rate = policy_params['standard_rate']
                current_compliance = policy_params['compliance']
                
                # Simple scaling for baseline estimation
                baseline_tax = micro_tax * (baseline_rate * baseline_compliance) / (current_rate * current_compliance)
                revenue_change = micro_tax - baseline_tax
                revenue_change_pct = (revenue_change / baseline_tax * 100) if baseline_tax > 0 else 0
                
                revenue_analysis = {
                    'baseline_revenue': baseline_tax,
                    'projected_revenue': micro_tax,
                    'revenue_change': revenue_change / 1_000_000,  # Convert to millions
                    'revenue_change_pct': revenue_change_pct,
                    'microsim_revenue': micro_tax,
                    'avg_effective_rate': micro_result.get('average_effective_rate', 0)
                }
            
            # Add additional analysis
            revenue_analysis.update({
                'revenue_efficiency': revenue_analysis.get('avg_effective_rate', 0) / policy_params['standard_rate'] if policy_params['standard_rate'] > 0 else 0,
                'laffer_position': min(policy_params['standard_rate'] / 0.25, 1.0),  # Theoretical peak at 25%
                'compliance_gap': (1 - policy_params['compliance']) * 100
            })
            
            return revenue_analysis
            
        except Exception as e:
            print(f"⚠️ Error in revenue analysis: {e}")
            return {'error': str(e)}
    
    def _comprehensive_risk_assessment(self, static_result: Dict, compliance: float) -> Dict:
        """Comprehensive risk assessment based on DGCE results."""
        try:
            # Economic risks
            gdp_impact = static_result.get('gdp_impact', 0)
            employment_impact = static_result.get('employment_impact', 0)
            investment_impact = static_result.get('investment_impact', 0)
            
            # Risk categories
            economic_risk = 'High' if gdp_impact < -5 else 'Medium' if gdp_impact < -2 else 'Low'
            employment_risk = 'High' if employment_impact < -3 else 'Medium' if employment_impact < -1 else 'Low'
            compliance_risk = 'High' if compliance < 0.6 else 'Medium' if compliance < 0.8 else 'Low'
            
            # Overall risk score (0-100)
            risk_score = 0
            risk_score += max(0, -gdp_impact * 10)  # GDP impact component
            risk_score += max(0, -employment_impact * 10)  # Employment component
            risk_score += (1 - compliance) * 30  # Compliance component
            risk_score = min(100, risk_score)
            
            return {
                'overall_risk_score': risk_score,
                'risk_level': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low',
                'economic_risk': economic_risk,
                'employment_risk': employment_risk,
                'compliance_risk': compliance_risk,
                'risk_factors': {
                    'gdp_decline': gdp_impact < -2,
                    'employment_decline': employment_impact < -1,
                    'low_compliance': compliance < 0.7,
                    'investment_decline': investment_impact < -5
                },
                'mitigation_recommendations': self._generate_risk_mitigation(
                    economic_risk, employment_risk, compliance_risk
                )
            }
            
        except Exception as e:
            print(f"⚠️ Error in risk assessment: {e}")
            return {'error': str(e)}
    
    def _generate_risk_mitigation(self, economic_risk: str, employment_risk: str, compliance_risk: str) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        if economic_risk == 'High':
            recommendations.append("Consider phased implementation of tax rate increases")
            recommendations.append("Implement pro-business incentives to offset negative impacts")
        
        if employment_risk == 'High':
            recommendations.append("Introduce employment protection measures")
            recommendations.append("Consider lower rates for labor-intensive sectors")
        
        if compliance_risk == 'High':
            recommendations.append("Strengthen tax administration and enforcement")
            recommendations.append("Implement digital tax filing and payment systems")
            recommendations.append("Provide taxpayer education and support")
        
        if not recommendations:
            recommendations.append("Current policy parameters appear to have manageable risk levels")
        
        return recommendations
    
    def _run_dynamic_analysis(self, policy_params: Dict) -> Dict:
        """Run dynamic analysis if available."""
        if not self.has_dynamics:
            return {}
        
        try:
            # Run dynamic analysis over time horizon
            time_horizon = policy_params.get('time_horizon', 10)
            return self.dgce_dynamic.calculate_dynamic_impacts(policy_params, time_horizon)
        except Exception as e:
            print(f"⚠️ Error in dynamic analysis: {e}")
            return {'error': str(e)}
    
    def _create_error_response(self, error_message: str, params: Dict) -> Dict:
        """Create error response with safe defaults."""
        return {
            'error': True,
            'error_message': error_message,
            'gdp_impact': 0,
            'employment_impact': 0,
            'investment_impact': 0,
            'consumption_impact': 0,
            'revenue_analysis': {'revenue_change': 0, 'revenue_change_pct': 0},
            'microsimulation': {'total_tax_revenue': 0, 'validation_passed': False},
            'sectoral_impacts': {},
            'risk_assessment': {'overall_risk_score': 100, 'risk_level': 'High'},
            'analysis_timestamp': datetime.now().isoformat(),
            'policy_parameters': params
        }

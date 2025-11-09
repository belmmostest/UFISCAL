"""
Enhanced Sector-Specific Policy Analysis (FIXED)
================================================

FIXES:
- Proper integration with fixed DGCE model
- Better sector validation and mapping
- Realistic calculations and bounds
- Improved error handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# Import the enhanced DGCE model
from dgce_model.model.dgce_model_enhanced_fixed import SimplifiedDGCEModel
from dgce_model.openfisca_runner import run_corporate_tax_simulation, validate_simulation_results
from dgce_model.data_loader.new_data_loader import RealUAEDataLoader

# Import sector mapping
try:
    from dgce_model.sector_mapping import ISIC_SECTORS, SHORT_TO_ISIC, get_policy_rate_for_sector
except ImportError:
    ISIC_SECTORS = [
        'Financial and insurance activities',
        'Manufacturing', 
        'Mining and quarrying',
        'Real estate activities',
        'Wholesale and retail trade, repair of motor vehicles and motorcycles',
        'Construction',
        'Information and communication',
        'Professional, scientific and technical activities',
        'Transportation and storage',
        'Accommodation and food service activities'
    ]
    SHORT_TO_ISIC = {
        'financial': 'Financial and insurance activities',
        'manufacturing': 'Manufacturing',
        'oil_gas': 'Mining and quarrying', 
        'real_estate': 'Real estate activities',
        'trade': 'Wholesale and retail trade, repair of motor vehicles and motorcycles',
        'construction': 'Construction',
        'technology': 'Information and communication',
        'professional': 'Professional, scientific and technical activities',
        'transport': 'Transportation and storage',
        'hospitality': 'Accommodation and food service activities'
    }
    
    def get_policy_rate_for_sector(sector, policy):
        return policy.get('standard_rate', 0.09)


class SectorPolicyAnalyzer:
    """
    Enhanced sector-specific policy analysis using DGCE model (FIXED).
    """
    
    def __init__(self):
        """Initialize with DGCE model and data loader."""
        print("🎯 Initializing Sector Policy Analyzer...")
        
        try:
            # Initialize core components
            self.dgce_model = SimplifiedDGCEModel()
            self.data_loader = RealUAEDataLoader()
            
            # Sector characteristics for analysis
            self.sector_characteristics = self._initialize_sector_characteristics()
            
            print("✅ Sector Policy Analyzer initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Sector Policy Analyzer: {e}")
            raise
    
    def analyze_sector_policy(self, sector_name: str, policy_params: Dict) -> Dict:
        """
        Analyze policy impact on a specific sector.
        FIXED: Better validation and realistic calculations.
        """
        
        try:
            print(f"🎯 Analyzing sector policy for: {sector_name}")
            
            # Validate and normalize sector name
            validated_sector = self._validate_and_normalize_sector(sector_name)
            
            # Validate policy parameters
            validated_params = self._validate_sector_policy_params(policy_params)
            
            # Get sector-specific data
            sector_data = self._get_sector_data(validated_sector)
            
            # Run DGCE simulation with sector focus
            dgce_results = self._run_sector_dgce_simulation(validated_sector, validated_params)
            
            # Run sector-specific microsimulation
            micro_results = self._run_sector_microsimulation(validated_sector, validated_params)
            
            # Calculate sector-specific impacts
            sector_impacts = self._extract_sector_impacts(validated_sector, dgce_results)
            
            # Analyze competitive effects
            competitive_analysis = self._analyze_competitive_effects(validated_sector, validated_params)
            
            # Generate sector recommendations
            recommendations = self._generate_sector_recommendations(
                validated_sector, sector_impacts, validated_params
            )
            
            # Compile results
            results = {
                # Sector identification
                'sector_name': validated_sector,
                'sector_code': self._get_sector_code(validated_sector),
                
                # Sector data
                'sector_profile': sector_data,
                
                # Economic impacts
                'sector_gdp_impact': sector_impacts.get('gdp_impact', 0),
                'sector_employment_impact': sector_impacts.get('employment_impact', 0),
                'sector_investment_impact': sector_impacts.get('investment_impact', 0),
                'sector_competitiveness_impact': sector_impacts.get('competitiveness_impact', 0),
                
                # Tax impacts
                'sector_tax_burden': micro_results.get('sector_tax_burden', 0),
                'sector_effective_rate': micro_results.get('sector_effective_rate', 0),
                'sector_companies_affected': micro_results.get('companies_affected', 0),
                
                # Detailed analysis
                'microsimulation_results': micro_results,
                'competitive_analysis': competitive_analysis,
                'sector_characteristics': self.sector_characteristics.get(validated_sector, {}),
                
                # Recommendations
                'policy_recommendations': recommendations,
                
                # Cross-sector comparison
                'relative_impact': self._calculate_relative_impact(validated_sector, dgce_results),
                
                # Validation
                'analysis_valid': micro_results.get('validation_passed', False),
                
                # Metadata
                'analysis_timestamp': datetime.now().isoformat(),
                'policy_parameters': validated_params,
                'analysis_type': 'sector_specific'
            }
            
            print(f"✅ Sector analysis complete for {validated_sector}: "
                  f"GDP impact {sector_impacts.get('gdp_impact', 0):.2f}%, "
                  f"Tax burden AED {micro_results.get('sector_tax_burden', 0)/1e6:.1f}M")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in sector policy analysis: {str(e)}")
            return self._create_sector_error_response(str(e), sector_name, policy_params)
    
    def _validate_and_normalize_sector(self, sector_input: str) -> str:
        """Validate and normalize sector name."""
        if not sector_input:
            raise ValueError("Sector name is required")
        
        # Check if already a valid ISIC sector
        if sector_input in ISIC_SECTORS:
            return sector_input
        
        # Check short names mapping
        sector_lower = sector_input.lower().strip()
        if sector_lower in SHORT_TO_ISIC:
            return SHORT_TO_ISIC[sector_lower]
        
        # Partial matching
        for isic_sector in ISIC_SECTORS:
            if sector_lower in isic_sector.lower():
                return isic_sector
        
        # If no match found, suggest alternatives
        suggestions = [sector for sector in ISIC_SECTORS if any(
            word in sector.lower() for word in sector_lower.split()
        )]
        
        error_msg = f"Sector '{sector_input}' not recognized."
        if suggestions:
            error_msg += f" Did you mean: {', '.join(suggestions[:3])}?"
        error_msg += f" Available sectors: {', '.join(ISIC_SECTORS)}"
        
        raise ValueError(error_msg)
    
    def _validate_sector_policy_params(self, params: Dict) -> Dict:
        """Validate sector policy parameters."""
        validated = {}
        
        # Required parameters
        if 'standard_rate' not in params:
            raise ValueError("standard_rate is required")
        
        try:
            validated['standard_rate'] = float(params['standard_rate'])
            if not (0.0 <= validated['standard_rate'] <= 0.5):
                raise ValueError("standard_rate must be between 0% and 50%")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid standard_rate: {e}")
        
        if 'compliance_rate' not in params:
            raise ValueError("compliance_rate is required")
        
        try:
            validated['compliance_rate'] = float(params['compliance_rate'])
            if not (0.1 <= validated['compliance_rate'] <= 1.0):
                raise ValueError("compliance_rate must be between 10% and 100%")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid compliance_rate: {e}")
        
        # Optional parameters with defaults
        validated['threshold'] = max(0, float(params.get('threshold', 375_000)))
        validated['small_biz_threshold'] = max(1000000, float(params.get('small_biz_threshold', 3_000_000)))
        validated['oil_gas_rate'] = max(0, min(0.8, float(params.get('oil_gas_rate', 0.55))))
        validated['incentives'] = params.get('incentives', {})
        
        return validated
    
    def _get_sector_data(self, sector_name: str) -> Dict:
        """Get sector-specific data from registry."""
        try:
            registry = self.data_loader.commerce_registry
            sector_companies = registry[registry['ISIC_level_1'] == sector_name]
            
            # Calculate sector statistics
            total_companies = len(sector_companies)
            total_revenue = sector_companies['annual_revenue'].sum()
            avg_revenue = sector_companies['annual_revenue'].mean()
            total_employees = sector_companies['employee_count'].sum()
            avg_employees = sector_companies['employee_count'].mean()
            
            # Get sector characteristics
            characteristics = self.sector_characteristics.get(sector_name, {})
            
            return {
                'total_companies': int(total_companies),
                'total_revenue': float(total_revenue),
                'average_revenue': float(avg_revenue),
                'total_employees': int(total_employees),
                'average_employees': float(avg_employees),
                'market_share': float(total_revenue / registry['annual_revenue'].sum() * 100) if len(registry) > 0 else 0,
                'characteristics': characteristics
            }
            
        except Exception as e:
            print(f"⚠️ Error getting sector data: {e}")
            return {
                'total_companies': 0,
                'total_revenue': 0,
                'average_revenue': 0,
                'total_employees': 0,
                'average_employees': 0,
                'market_share': 0,
                'characteristics': {}
            }
    
    def _run_sector_dgce_simulation(self, sector_name: str, policy_params: Dict) -> Dict:
        """Run DGCE simulation with sector-specific focus."""
        try:
            # Build shocks for DGCE model
            shocks = {
                'tax_changes': {
                    'corporate': policy_params['standard_rate']
                },
                'compliance_rate': policy_params['compliance_rate'],
                'incentives': policy_params.get('incentives', {}),
                'sector_focus': sector_name
            }
            
            # Run DGCE simulation
            results = self.dgce_model.solve_policy_impact(shocks)
            
            print(f"📈 DGCE simulation completed for {sector_name}: "
                  f"GDP impact {results.get('gdp_growth', 0)*100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"⚠️ Error in DGCE simulation: {e}")
            return {'error': str(e)}
    
    def _run_sector_microsimulation(self, sector_name: str, policy_params: Dict) -> Dict:
        """Run microsimulation focused on specific sector."""
        try:
            # Filter companies for the sector
            registry = self.data_loader.commerce_registry
            sector_companies = registry[registry['ISIC_level_1'] == sector_name].copy()
            
            if sector_companies.empty:
                return {
                    'sector_tax_burden': 0,
                    'sector_effective_rate': 0,
                    'companies_affected': 0,
                    'validation_passed': False,
                    'error': 'No companies found in sector'
                }
            
            # Prepare microsimulation parameters
            micro_params = {
                "standard_rate": policy_params['standard_rate'],
                "free_zone_rate": 0.0,
                "oil_gas_rate": policy_params.get('oil_gas_rate', 0.55),
                "small_business_threshold": policy_params['small_biz_threshold'],
                "profit_allowance": policy_params['threshold']
            }
            
            # Run microsimulation on sector companies
            sector_results = run_corporate_tax_simulation(
                companies=sector_companies,
                sectoral_panel=self.data_loader.sectoral_panel,
                params=micro_params
            )
            
            # Validate results
            validation = validate_simulation_results(sector_results)
            
            # Calculate sector metrics
            sector_tax_burden = float(sector_results['corporate_tax'].sum())
            sector_effective_rate = float(sector_results['effective_tax_rate'].mean())
            companies_affected = len(sector_results[sector_results['corporate_tax'] > 0])
            
            return {
                'sector_tax_burden': sector_tax_burden,
                'sector_effective_rate': sector_effective_rate,
                'companies_affected': companies_affected,
                'total_sector_companies': len(sector_results),
                'validation_passed': validation['validation_passed'],
                'validation_details': validation,
                # 'detailed_results': sector_results
            }
            
        except Exception as e:
            print(f"⚠️ Error in sector microsimulation: {e}")
            return {
                'sector_tax_burden': 0,
                'sector_effective_rate': 0,
                'companies_affected': 0,
                'validation_passed': False,
                'error': str(e)
            }
    
    def _extract_sector_impacts(self, sector_name: str, dgce_results: Dict) -> Dict:
        """Extract sector-specific impacts from DGCE results."""
        try:
            # Get overall impacts
            gdp_growth = dgce_results.get('gdp_growth', 0)
            employment_growth = dgce_results.get('employment_growth', 0)
            investment_growth = dgce_results.get('investment_growth', 0)
            
            # Get sector-specific multipliers
            sector_characteristics = self.sector_characteristics.get(sector_name, {})
            sector_sensitivity = sector_characteristics.get('tax_sensitivity', 1.0)
            
            # Calculate sector-specific impacts
            sector_gdp_impact = gdp_growth * sector_sensitivity
            sector_employment_impact = employment_growth * sector_sensitivity
            sector_investment_impact = investment_growth * sector_sensitivity
            
            # Calculate competitiveness impact
            competitiveness_impact = self._calculate_competitiveness_impact(
                sector_name, sector_gdp_impact
            )
            
            return {
                'gdp_impact': sector_gdp_impact * 100,  # Convert to percentage
                'employment_impact': sector_employment_impact * 100,
                'investment_impact': sector_investment_impact * 100,
                'competitiveness_impact': competitiveness_impact
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting sector impacts: {e}")
            return {
                'gdp_impact': 0,
                'employment_impact': 0,
                'investment_impact': 0,
                'competitiveness_impact': 0
            }
    
    def _analyze_competitive_effects(self, sector_name: str, policy_params: Dict) -> Dict:
        """Analyze competitive effects of policy on sector."""
        try:
            characteristics = self.sector_characteristics.get(sector_name, {})
            
            # Factors affecting competitiveness
            tax_rate_impact = policy_params['standard_rate'] * -100  # Higher tax = lower competitiveness
            compliance_burden = (1 - policy_params['compliance_rate']) * 50  # Lower compliance = higher burden
            
            # Sector-specific factors
            international_exposure = characteristics.get('international_exposure', 0.5)
            regulatory_intensity = characteristics.get('regulatory_intensity', 0.5)
            
            # Calculate overall competitiveness score
            competitiveness_score = (
                tax_rate_impact * international_exposure +
                compliance_burden * regulatory_intensity
            )
            
            # Normalize to -100 to +100 scale
            competitiveness_score = max(-100, min(100, competitiveness_score))
            
            return {
                'competitiveness_score': competitiveness_score,
                'tax_rate_impact': tax_rate_impact,
                'compliance_burden': compliance_burden,
                'international_exposure': international_exposure,
                'regulatory_intensity': regulatory_intensity,
                'competitive_position': (
                    'Strong' if competitiveness_score > 20 else
                    'Moderate' if competitiveness_score > -20 else
                    'Weak'
                )
            }
            
        except Exception as e:
            print(f"⚠️ Error in competitive analysis: {e}")
            return {'error': str(e)}
    
    def _generate_sector_recommendations(self, sector_name: str, impacts: Dict, policy_params: Dict) -> List[str]:
        """Generate sector-specific policy recommendations."""
        recommendations = []
        
        try:
            gdp_impact = impacts.get('gdp_impact', 0)
            employment_impact = impacts.get('employment_impact', 0)
            tax_rate = policy_params['standard_rate']
            
            # General recommendations based on impacts
            if gdp_impact < -5:
                recommendations.append(f"Consider phased implementation for {sector_name} to minimize economic disruption")
                recommendations.append("Implement sector-specific incentives to offset negative impacts")
            
            if employment_impact < -3:
                recommendations.append("Monitor employment effects and consider job protection measures")
            
            if tax_rate > 0.15:  # 15%
                recommendations.append("High tax rate may reduce sector competitiveness")
            
            # Sector-specific recommendations
            characteristics = self.sector_characteristics.get(sector_name, {})
            
            if characteristics.get('international_exposure', 0) > 0.7:
                recommendations.append("Consider international tax competitiveness implications")
            
            if characteristics.get('capital_intensity', 0) > 0.7:
                recommendations.append("Evaluate impact on capital investment decisions")
                recommendations.append("Consider enhanced depreciation allowances")
            
            if characteristics.get('innovation_intensity', 0) > 0.6:
                recommendations.append("Implement R&D tax credits to maintain innovation incentives")
            
            # Default recommendation if no specific issues
            if not recommendations:
                recommendations.append("Policy parameters appear well-suited for this sector")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_relative_impact(self, sector_name: str, dgce_results: Dict) -> Dict:
        """Calculate sector impact relative to economy-wide effects."""
        try:
            # Get economy-wide impacts
            economy_gdp = dgce_results.get('gdp_growth', 0)
            economy_employment = dgce_results.get('employment_growth', 0)
            
            # Get sector characteristics
            characteristics = self.sector_characteristics.get(sector_name, {})
            sector_sensitivity = characteristics.get('tax_sensitivity', 1.0)
            
            # Calculate relative impacts
            relative_gdp = economy_gdp * sector_sensitivity / economy_gdp if economy_gdp != 0 else 1.0
            relative_employment = economy_employment * sector_sensitivity / economy_employment if economy_employment != 0 else 1.0
            
            return {
                'relative_gdp_impact': relative_gdp,
                'relative_employment_impact': relative_employment,
                'sector_sensitivity': sector_sensitivity,
                'impact_ranking': self._rank_sector_impact(sector_name, relative_gdp)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_competitiveness_impact(self, sector_name: str, gdp_impact: float) -> float:
        """Calculate competitiveness impact score."""
        try:
            characteristics = self.sector_characteristics.get(sector_name, {})
            
            # Base competitiveness impact from GDP
            base_impact = gdp_impact * 2  # Amplify for competitiveness
            
            # Adjust for sector characteristics
            international_factor = characteristics.get('international_exposure', 0.5)
            base_impact *= (1 + international_factor)
            
            return max(-100, min(100, base_impact))
            
        except Exception:
            return 0.0
    
    def _rank_sector_impact(self, sector_name: str, relative_impact: float) -> str:
        """Rank sector impact severity."""
        if relative_impact > 1.5:
            return "High positive impact"
        elif relative_impact > 1.1:
            return "Moderate positive impact"
        elif relative_impact > 0.9:
            return "Neutral impact"
        elif relative_impact > 0.7:
            return "Moderate negative impact"
        else:
            return "High negative impact"
    
    def _get_sector_code(self, sector_name: str) -> str:
        """Get sector code for the given sector name."""
        sector_codes = {
            'Financial and insurance activities': 'K',
            'Manufacturing': 'C',
            'Mining and quarrying': 'B',
            'Real estate activities': 'L',
            'Wholesale and retail trade, repair of motor vehicles and motorcycles': 'G',
            'Construction': 'F',
            'Information and communication': 'J',
            'Professional, scientific and technical activities': 'M',
            'Transportation and storage': 'H',
            'Accommodation and food service activities': 'I'
        }
        return sector_codes.get(sector_name, 'Unknown')
    
    def _initialize_sector_characteristics(self) -> Dict[str, Dict]:
        """Initialize sector characteristics for analysis."""
        return {
            'Financial and insurance activities': {
                'tax_sensitivity': 1.2,
                'international_exposure': 0.8,
                'capital_intensity': 0.6,
                'innovation_intensity': 0.7,
                'regulatory_intensity': 0.9
            },
            'Manufacturing': {
                'tax_sensitivity': 1.0,
                'international_exposure': 0.7,
                'capital_intensity': 0.8,
                'innovation_intensity': 0.6,
                'regulatory_intensity': 0.6
            },
            'Mining and quarrying': {
                'tax_sensitivity': 0.5,  # Less sensitive due to resource rents
                'international_exposure': 0.9,
                'capital_intensity': 0.9,
                'innovation_intensity': 0.4,
                'regulatory_intensity': 0.8
            },
            'Real estate activities': {
                'tax_sensitivity': 0.8,
                'international_exposure': 0.3,
                'capital_intensity': 0.9,
                'innovation_intensity': 0.3,
                'regulatory_intensity': 0.7
            },
            'Wholesale and retail trade, repair of motor vehicles and motorcycles': {
                'tax_sensitivity': 1.1,
                'international_exposure': 0.5,
                'capital_intensity': 0.4,
                'innovation_intensity': 0.4,
                'regulatory_intensity': 0.5
            },
            'Construction': {
                'tax_sensitivity': 0.9,
                'international_exposure': 0.4,
                'capital_intensity': 0.6,
                'innovation_intensity': 0.3,
                'regulatory_intensity': 0.6
            },
            'Information and communication': {
                'tax_sensitivity': 1.3,
                'international_exposure': 0.9,
                'capital_intensity': 0.5,
                'innovation_intensity': 0.9,
                'regulatory_intensity': 0.4
            },
            'Professional, scientific and technical activities': {
                'tax_sensitivity': 1.2,
                'international_exposure': 0.6,
                'capital_intensity': 0.4,
                'innovation_intensity': 0.8,
                'regulatory_intensity': 0.5
            },
            'Transportation and storage': {
                'tax_sensitivity': 0.8,
                'international_exposure': 0.6,
                'capital_intensity': 0.7,
                'innovation_intensity': 0.4,
                'regulatory_intensity': 0.7
            },
            'Accommodation and food service activities': {
                'tax_sensitivity': 1.0,
                'international_exposure': 0.5,
                'capital_intensity': 0.5,
                'innovation_intensity': 0.3,
                'regulatory_intensity': 0.6
            }
        }
    
    def _create_sector_error_response(self, error_message: str, sector_name: str, policy_params: Dict) -> Dict:
        """Create error response for sector analysis."""
        return {
            'error': True,
            'error_message': error_message,
            'sector_name': sector_name,
            'sector_gdp_impact': 0,
            'sector_employment_impact': 0,
            'sector_investment_impact': 0,
            'sector_tax_burden': 0,
            'sector_effective_rate': 0,
            'analysis_valid': False,
            'analysis_timestamp': datetime.now().isoformat(),
            'policy_parameters': policy_params
        }
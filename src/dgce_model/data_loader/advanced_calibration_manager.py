"""
Advanced Calibration Manager for OpenFisca UAE
==============================================

This enhanced calibration manager integrates real UAE data including:
- Input-output matrix
- Advanced economic parameters
- Real sectoral data
- Sophisticated elasticity calculations

Replaces the unified calibration manager with real data-driven parameters.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dgce_model.model.dgce_model_enhanced_fixed import CalibrationLoader, UAECalibration2024
    DGCE_AVAILABLE = True
except ImportError as e:
    DGCE_AVAILABLE = False
    logging.warning(f"DGCE model not available: {e}")

from .real_data_loader import RealUAEDataLoader

logger = logging.getLogger(__name__)


class AdvancedCalibrationManager:
    """
    Advanced calibration manager using real UAE data and sophisticated economic parameters.
    
    Integrates:
    - Real business registry data
    - Input-output matrix
    - Advanced macroeconomic parameters
    - Sectoral elasticities derived from real data
    - Dynamic parameter updating
    """
    
    def __init__(self, data_path: str = None, real_data_loader: RealUAEDataLoader = None):
        """
        Initialize with real UAE data.
        
        Args:
            data_path: Path to data directory
            real_data_loader: Pre-initialized RealUAEDataLoader (optional)
        """
        self.calibration_date = datetime.now().isoformat()
        self.data_source = "Real UAE Data + DGCE Calibration"
        
        # Load real data
        if real_data_loader is not None:
            self.real_data = real_data_loader
        else:
            self.real_data = RealUAEDataLoader(data_path)
        
        # Load DGCE calibration for comparison/validation
        self.dgce_available = DGCE_AVAILABLE
        if self.dgce_available:
            try:
                loader = CalibrationLoader(data_path)
                self.dgce_calibration = loader.load_calibration()
            except Exception as e:
                logger.warning(f"Could not load DGCE calibration: {e}")
                self.dgce_calibration = UAECalibration2024()
        else:
            self.dgce_calibration = UAECalibration2024()
        
        # Initialize advanced parameters
        self._initialize_advanced_calibration()
        
        logger.info(f"🚀 AdvancedCalibrationManager initialized with real UAE data")
        logger.info(f"📊 Data source: {self.data_source}")
    
    def _initialize_advanced_calibration(self):
        """Initialize calibration using real data and advanced parameters."""
        
        # Get advanced economic parameters
        self.advanced_params = self.real_data.get_advanced_parameters()
        
        # Get input-output matrix
        self.input_output_matrix = self.real_data.get_input_output_matrix()
        
        # Get latest sectoral data
        self.sectoral_data = self.real_data.get_sectoral_parameters()
        
        # Calculate elasticities from real data
        self._calculate_real_elasticities()
        
        # Initialize sector-specific parameters
        self._initialize_sector_parameters()
        
        # Calculate production function parameters
        self._calculate_production_parameters()
        
        # Set up labor market parameters
        self._initialize_labor_parameters()
        
        # Initialize innovation and FDI parameters
        self._initialize_innovation_fdi_parameters()
        
        logger.info("✅ Advanced calibration initialized from real data")
    
    def _calculate_real_elasticities(self):
        """Calculate elasticities from real UAE sectoral data."""
        
        # Base elasticities from DGCE (validated)
        self.core_elasticities = {
            'output_tax_elasticity': self.dgce_calibration.output_tax_elasticity,
            'investment_tax_elasticity': self.dgce_calibration.investment_tax_elasticity,
            'consumption_tax_elasticity': self.dgce_calibration.consumption_tax_elasticity,
            'labor_supply_elasticity': self.dgce_calibration.labor_supply_elasticity,
        }
        
        # Advanced elasticities from real parameters
        self.advanced_elasticities = {
            'frisch_elasticity': self.advanced_params['frisch_elasticity_labor_supply'],
            'capital_labor_substitution': self.advanced_params['elasticity_substitution_capital_labor'],
            'intersectoral_substitution': self._calculate_intersectoral_elasticity(),
        }
        
        # Sector-specific elasticities calculated from real data
        self.sectoral_elasticities = self._calculate_sectoral_elasticities()
        
        logger.info(f"📈 Calculated elasticities from real data:")
        logger.info(f"   Frisch elasticity: {self.advanced_elasticities['frisch_elasticity']}")
        logger.info(f"   Capital-labor substitution: {self.advanced_elasticities['capital_labor_substitution']}")
    
    def _calculate_intersectoral_elasticity(self) -> float:
        """Calculate intersectoral substitution elasticity from I-O matrix."""
        # Calculate elasticity based on I-O matrix structure
        # Higher off-diagonal elements suggest more substitutability
        
        io_matrix = self.input_output_matrix.values
        diagonal_sum = np.trace(io_matrix)
        off_diagonal_sum = np.sum(io_matrix) - diagonal_sum
        
        # Elasticity proxy: ratio of off-diagonal to diagonal
        elasticity = off_diagonal_sum / diagonal_sum if diagonal_sum > 0 else 0.5
        
        # Bound between reasonable values (0.3 to 2.0)
        return np.clip(elasticity, 0.3, 2.0)
    
    def _calculate_sectoral_elasticities(self) -> Dict[str, Dict[str, float]]:
        """Calculate sector-specific elasticities from real data."""
        
        sectoral_elasticities = {}
        
        for sector in self.sectoral_data['economic_activity'].unique():
            sector_data = self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].iloc[0] if not self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].empty else None
            
            if sector_data is None:
                continue
            
            # Calculate elasticities based on sector characteristics
            labor_share = sector_data.get('labor_share', 0.5)
            capital_intensity = sector_data.get('capital_intensity', 100000)
            
            # Employment elasticity: function of labor share and capital intensity
            # Higher labor share = more elastic employment
            # Higher capital intensity = less elastic employment
            employment_elasticity = -0.2 - (labor_share * 0.3) + (np.log(capital_intensity + 1) * 0.05)
            employment_elasticity = np.clip(employment_elasticity, -1.0, -0.1)
            
            # Investment elasticity: function of capital intensity
            # More capital-intensive sectors more responsive to tax changes
            investment_elasticity = -0.5 - (np.log(capital_intensity + 1) * 0.1)
            investment_elasticity = np.clip(investment_elasticity, -1.5, -0.3)
            
            # Output elasticity: weighted average
            output_elasticity = (labor_share * employment_elasticity + 
                               (1 - labor_share) * investment_elasticity * 0.5)
            
            sectoral_elasticities[sector] = {
                'employment_elasticity': employment_elasticity,
                'investment_elasticity': investment_elasticity,
                'output_elasticity': output_elasticity,
                'labor_share': labor_share,
                'capital_intensity': capital_intensity
            }
        
        return sectoral_elasticities
    
    def _initialize_sector_parameters(self):
        """Initialize sector-specific parameters from real data."""
        
        # Calculate sector shares from real data
        latest_data = self.sectoral_data.copy()
        total_value_added = latest_data['value_added_in_aed'].sum()
        
        self.real_sector_shares = {}
        self.sector_characteristics = {}
        
        for _, row in latest_data.iterrows():
            sector = row['economic_activity']
            
            # Real sector share in economy
            self.real_sector_shares[sector] = row['value_added_in_aed'] / total_value_added
            
            # Comprehensive sector characteristics
            self.sector_characteristics[sector] = {
                'value_added': row['value_added_in_aed'],
                'output': row['output_in_aed'],
                'employment': row['number_of_employees'],
                'labor_productivity': row.get('labor_productivity', 0),
                'capital_intensity': row.get('capital_intensity', 0),
                'profit_margin': row.get('profit_margin', 0),
                'labor_share': row.get('labor_share', 0),
                'intermediate_intensity': row.get('intermediate_intensity', 0)
            }
        
        logger.info(f"🏭 Initialized {len(self.real_sector_shares)} sectors from real data")
    
    def _calculate_production_parameters(self):
        """Calculate production function parameters from real data."""
        
        # Capital share from advanced parameters
        self.capital_share = self.advanced_params['capital_share_output']
        
        # Labor share (complement)
        self.labor_share = 1 - self.capital_share
        
        # Total factor productivity from advanced parameters
        self.tfp = self.advanced_params['total_factor_productivity']
        
        # Depreciation rate
        self.depreciation_rate = self.advanced_params['capital_depreciation_rate']
        
        # Calculate sector-specific TFP adjustments
        self.sector_tfp_adjustments = {}
        
        # Use labor productivity to adjust TFP by sector
        mean_productivity = self.sectoral_data['labor_productivity'].mean()
        
        for sector in self.sectoral_data['economic_activity'].unique():
            sector_data = self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].iloc[0] if not self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].empty else None
            
            if sector_data is not None:
                # TFP adjustment based on relative productivity
                productivity_ratio = sector_data['labor_productivity'] / mean_productivity
                tfp_adjustment = 0.8 + 0.4 * productivity_ratio  # Range: 0.8 to 1.2
                self.sector_tfp_adjustments[sector] = tfp_adjustment
        
        self.production_parameters = {
            'capital_share': self.capital_share,
            'labor_share': self.labor_share,
            'tfp': self.tfp,
            'depreciation_rate': self.depreciation_rate,
            'sector_tfp_adjustments': self.sector_tfp_adjustments
        }
        
        logger.info(f"⚙️ Production parameters: α={self.capital_share:.3f}, δ={self.depreciation_rate:.3f}")
    
    def _initialize_labor_parameters(self):
        """Initialize labor market parameters from real data."""
        
        # Get labor productivity type fractions
        self.labor_productivity_fractions = self.advanced_params['labor_productivity_type_fractions']
        
        # Calculate employment structure from real data
        registry_summary = self.real_data.get_employment_distribution()
        
        # Total employment from real data
        total_employment = self.sectoral_data['number_of_employees'].sum()
        
        # Calculate sector employment shares
        sector_employment_shares = {}
        for sector in self.sectoral_data['economic_activity'].unique():
            sector_employment = self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ]['number_of_employees'].iloc[0] if not self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].empty else 0
            
            sector_employment_shares[sector] = sector_employment / total_employment
        
        # Labor market parameters
        self.labor_parameters = {
            'total_employment': total_employment,
            'sector_employment_shares': sector_employment_shares,
            'productivity_type_fractions': self.labor_productivity_fractions,
            'frisch_elasticity': self.advanced_elasticities['frisch_elasticity'],
            'labor_augmenting_growth': self.advanced_params['labor_augmenting_tech_growth_rate']
        }
        
        # Calculate wage structure from real data
        self._calculate_wage_structure()
        
        logger.info(f"👥 Labor parameters: {total_employment:,} total employment")
    
    def _calculate_wage_structure(self):
        """Calculate wage structure from sectoral compensation data."""
        
        self.wage_structure = {}
        
        for sector in self.sectoral_data['economic_activity'].unique():
            sector_data = self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].iloc[0] if not self.sectoral_data[
                self.sectoral_data['economic_activity'] == sector
            ].empty else None
            
            if sector_data is not None and sector_data['number_of_employees'] > 0:
                # Average wage in sector
                avg_wage = (sector_data['compensation_of_employees_in_aed'] / 
                           sector_data['number_of_employees'] / 12)  # Monthly wage
                
                self.wage_structure[sector] = {
                    'average_monthly_wage': avg_wage,
                    'annual_compensation': sector_data['compensation_of_employees_in_aed'],
                    'employment': sector_data['number_of_employees']
                }
    
    def _initialize_innovation_fdi_parameters(self):
        """Initialize innovation and FDI parameters."""
        
        # Innovation parameters - combine real data with research-based estimates
        self.innovation_parameters = {
            'rd_tax_elasticity': 1.2,  # From innovation literature
            'startup_tax_elasticity': 1.5,
            'patent_elasticity': 0.8,
            'productivity_multiplier': 0.3,
            'rd_intensity_by_sector': self._calculate_rd_intensity()
        }
        
        # FDI parameters - based on sectoral characteristics
        self.fdi_parameters = {
            'base_fdi_elasticity': self.core_elasticities['investment_tax_elasticity'] * 1.5,
            'competitiveness_elasticity': -1.0,
            'headquarters_elasticity': -1.8,
            'sector_fdi_sensitivity': self._calculate_fdi_sensitivity()
        }
        
        logger.info("💡 Innovation and FDI parameters initialized")
    
    def _calculate_rd_intensity(self) -> Dict[str, float]:
        """Calculate R&D intensity by sector from real data."""
        
        # Default R&D intensities (can be enhanced with real R&D data if available)
        base_intensities = {
            'Manufacturing': 0.020,
            'Information and communication': 0.073,
            'Financial and insurance activities': 0.016,
            'Professional, scientific and technical activities': 0.044,
            'Mining and quarrying': 0.006
        }
        
        # Adjust based on sector productivity and capital intensity
        rd_intensities = {}
        
        for sector in self.sectoral_data['economic_activity'].unique():
            base_intensity = base_intensities.get(sector, 0.005)
            
            if sector in self.sector_characteristics:
                # Adjust based on labor productivity (proxy for innovation)
                characteristics = self.sector_characteristics[sector]
                productivity_factor = min(2.0, characteristics['labor_productivity'] / 100000)
                adjusted_intensity = base_intensity * productivity_factor
                rd_intensities[sector] = adjusted_intensity
            else:
                rd_intensities[sector] = base_intensity
        
        return rd_intensities
    
    def _calculate_fdi_sensitivity(self) -> Dict[str, float]:
        """Calculate FDI sensitivity by sector from real data."""
        
        fdi_sensitivities = {}
        
        for sector in self.sectoral_data['economic_activity'].unique():
            if sector in self.sector_characteristics:
                characteristics = self.sector_characteristics[sector]
                
                # Higher capital intensity = higher FDI sensitivity
                capital_factor = min(2.0, np.log(characteristics['capital_intensity'] + 1) / 10)
                
                # Higher profit margins = higher sensitivity
                profit_factor = 1 + characteristics['profit_margin']
                
                # Combine factors
                sensitivity = capital_factor * profit_factor
                sensitivity = np.clip(sensitivity, 0.5, 2.0)  # Reasonable bounds
                
                fdi_sensitivities[sector] = sensitivity
            else:
                fdi_sensitivities[sector] = 1.0  # Default
        
        return fdi_sensitivities
    
    # Public Interface Methods
    
    def get_elasticities(self, model_type: str = 'general', sector: str = None) -> Dict[str, float]:
        """
        Get elasticities for specific model type and sector.
        
        Args:
            model_type: 'general', 'labor', 'fdi', 'innovation', 'dgce', or 'sector'
            sector: Specific sector for sector-specific elasticities
            
        Returns:
            Dictionary of relevant elasticities
        """
        base_elasticities = self.core_elasticities.copy()
        base_elasticities.update(self.advanced_elasticities)
        
        if model_type == 'labor':
            base_elasticities.update({
                'employment_elasticity': -0.33,  # Default
                'wage_elasticity': -0.2,
                'emiratization_elasticity': 0.1,
                'rd_tax_elasticity': 1.2,  # From innovation literature
                'startup_tax_elasticity': 1.5,
                'patent_elasticity': 0.8,
                'productivity_multiplier': 0.3,
            })
            
            # Add sector-specific if requested
            if sector and sector in self.sectoral_elasticities:
                base_elasticities['employment_elasticity'] = (
                    self.sectoral_elasticities[sector]['employment_elasticity']
                )
        
        elif model_type == 'fdi':
            base_elasticities.update({
                'fdi_elasticity': self.fdi_parameters['base_fdi_elasticity'],
                'competitiveness_elasticity': self.fdi_parameters['competitiveness_elasticity'],
                'headquarters_elasticity': self.fdi_parameters['headquarters_elasticity']
            })
            
            # Add sector sensitivity
            if sector and sector in self.fdi_parameters['sector_fdi_sensitivity']:
                sensitivity = self.fdi_parameters['sector_fdi_sensitivity'][sector]
                base_elasticities['fdi_elasticity'] *= sensitivity
        
        elif model_type == 'innovation':
            base_elasticities.update(self.innovation_parameters)
            
            # Add sector R&D intensity
            if sector and sector in self.innovation_parameters['rd_intensity_by_sector']:
                base_elasticities['sector_rd_intensity'] = (
                    self.innovation_parameters['rd_intensity_by_sector'][sector]
                )
        
        elif model_type == 'sector' and sector:
            # Return sector-specific elasticities
            if sector in self.sectoral_elasticities:
                base_elasticities.update(self.sectoral_elasticities[sector])
        
        return base_elasticities
    
    def get_sector_characteristics(self, sector: str = None) -> Dict[str, Any]:
        """Get real sector characteristics."""
        if sector:
            return self.sector_characteristics.get(sector, {})
        return self.sector_characteristics
    
    def get_sector_shares(self) -> Dict[str, float]:
        """Get real sector shares from data."""
        return self.real_sector_shares.copy()
    
    def get_production_parameters(self) -> Dict[str, Any]:
        """Get production function parameters."""
        return self.production_parameters.copy()
    
    def get_labor_parameters(self) -> Dict[str, Any]:
        """Get labor market parameters."""
        return self.labor_parameters.copy()
    
    def get_input_output_matrix(self) -> pd.DataFrame:
        """Get the input-output matrix."""
        return self.input_output_matrix.copy()
    
    def get_advanced_parameters(self) -> Dict[str, Any]:
        """Get all advanced economic parameters."""
        return self.advanced_params.copy()
    
    def get_wage_structure(self, sector: str = None) -> Dict[str, Any]:
        """Get wage structure by sector."""
        if sector:
            return self.wage_structure.get(sector, {})
        return self.wage_structure.copy()
    
    def calculate_fallback_impact(self, 
                                tax_changes: Dict[str, float], 
                                impact_type: str = 'gdp',
                                sector: str = None) -> float:
        """
        Calculate economic impact using real data elasticities when DGCE model fails.
        
        Args:
            tax_changes: Dictionary of tax rate changes
            impact_type: 'gdp', 'employment', 'investment', 'consumption'
            sector: Specific sector for sectoral analysis
            
        Returns:
            Estimated percentage impact
        """
        if 'corporate_rate' in tax_changes:
            current_rate = 0.09  # Current corporate rate
            new_rate = tax_changes['corporate_rate']
            tax_change_percent = ((new_rate - current_rate) / current_rate) * 100
            
            # Get appropriate elasticity
            if sector and sector in self.sectoral_elasticities:
                elasticities = self.sectoral_elasticities[sector]
                if impact_type == 'gdp':
                    elasticity = elasticities['output_elasticity']
                elif impact_type == 'employment':
                    elasticity = elasticities['employment_elasticity']
                elif impact_type == 'investment':
                    elasticity = elasticities['investment_elasticity']
                else:
                    elasticity = self.core_elasticities.get(f'{impact_type}_tax_elasticity', -0.25)
            else:
                # Use core elasticities
                elasticity_map = {
                    'gdp': 'output_tax_elasticity',
                    'employment': 'labor_supply_elasticity',
                    'investment': 'investment_tax_elasticity',
                    'consumption': 'consumption_tax_elasticity'
                }
                elasticity = self.core_elasticities.get(
                    elasticity_map.get(impact_type, 'output_tax_elasticity'),
                    -0.25
                )
            
            return tax_change_percent * elasticity
        
        return 0.0
    
    def get_calibration_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about current calibration."""
        return {
            'data_source': self.data_source,
            'calibration_date': self.calibration_date,
            'real_data_available': True,
            'dgce_available': self.dgce_available,
            'total_businesses': len(self.real_data.commerce_registry),
            'sectors_covered': len(self.real_sector_shares),
            'input_output_dimensions': list(self.input_output_matrix.shape),
            'advanced_parameters_count': len(self.advanced_params),
            'elasticities': {
                'core': self.core_elasticities,
                'advanced': self.advanced_elasticities,
                'sectoral_count': len(self.sectoral_elasticities)
            },
            'production_parameters': {
                'capital_share': self.capital_share,
                'tfp': self.tfp,
                'depreciation_rate': self.depreciation_rate
            },
            'labor_parameters': {
                'total_employment': self.labor_parameters['total_employment'],
                'frisch_elasticity': self.advanced_elasticities['frisch_elasticity']
            },
            'version': '2.0 - Real Data Enhanced'
        }
    
    def export_calibration_summary(self, filepath: str = None) -> Dict[str, Any]:
        """Export comprehensive calibration summary."""
        summary = {
            'metadata': self.get_calibration_metadata(),
            'sector_characteristics': self.sector_characteristics,
            'sectoral_elasticities': self.sectoral_elasticities,
            'real_sector_shares': self.real_sector_shares,
            'production_parameters': self.production_parameters,
            'labor_parameters': self.labor_parameters,
            'innovation_parameters': self.innovation_parameters,
            'fdi_parameters': self.fdi_parameters,
            'wage_structure': self.wage_structure,
            'input_output_matrix_info': {
                'shape': list(self.input_output_matrix.shape),
                'sectors': list(self.input_output_matrix.index),
                'row_sums_check': self.input_output_matrix.sum(axis=1).tolist()
            }
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"📄 Advanced calibration summary exported to {filepath}")
        
        return summary


# Global instance for easy import
_global_advanced_calibration_manager = None

def get_advanced_calibration_manager(data_path: str = None, 
                                   real_data_loader: RealUAEDataLoader = None,
                                   force_reload: bool = False) -> AdvancedCalibrationManager:
    """
    Get global advanced calibration manager instance.
    
    Args:
        data_path: Path to data directory
        real_data_loader: Pre-initialized RealUAEDataLoader
        force_reload: Force reload of calibration data
        
    Returns:
        AdvancedCalibrationManager instance
    """
    global _global_advanced_calibration_manager
    
    if _global_advanced_calibration_manager is None or force_reload:
        _global_advanced_calibration_manager = AdvancedCalibrationManager(
            data_path, real_data_loader
        )
    
    return _global_advanced_calibration_manager


# Convenience functions for quick access
def get_real_elasticity(elasticity_name: str, 
                       model_type: str = 'general', 
                       sector: str = None) -> float:
    """Quick access to real data elasticity."""
    manager = get_advanced_calibration_manager()
    elasticities = manager.get_elasticities(model_type, sector)
    return elasticities.get(elasticity_name, 0.0)

def get_real_sector_characteristics(sector: str) -> Dict[str, Any]:
    """Quick access to real sector characteristics."""
    manager = get_advanced_calibration_manager()
    return manager.get_sector_characteristics(sector)

def get_real_fallback_impact(tax_changes: Dict[str, float], 
                           impact_type: str = 'gdp',
                           sector: str = None) -> float:
    """Quick access to real data fallback impact calculation."""
    manager = get_advanced_calibration_manager()
    return manager.calculate_fallback_impact(tax_changes, impact_type, sector)
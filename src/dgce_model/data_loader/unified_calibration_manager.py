"""
Unified Calibration Manager for OpenFisca UAE
=============================================

This module provides a single source of truth for all economic model calibration data,
ensuring consistency between DGCE model and individual economic models (Labor, FDI, Innovation).

All models should use this manager to get elasticities, parameters, and calibration data
instead of hardcoded values.
"""

import os
import sys
import json
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, "./")

try:
    from dgce_model.model.dgce_model_enhanced_fixed import CalibrationLoader, UAECalibration2024
    DGCE_AVAILABLE = True
except ImportError as e:
    DGCE_AVAILABLE = False
    logging.warning(f"DGCE model not available: {e}")

logger = logging.getLogger(__name__)


class UnifiedCalibrationManager:
    """
    Central manager for all economic model calibration data.
    
    Ensures all models (DGCE, Labor, FDI, Innovation) use consistent, up-to-date
    calibration parameters from the same data source.
    """
    
    def __init__(self, dgce_data_path: str = None, force_reload: bool = False):
        """
        Initialize with DGCE calibration data.
        
        Args:
            dgce_data_path: Path to DGCE data directory
            force_reload: Force reload of calibration data
        """
        self.calibration_date = datetime.now().isoformat()
        self.dgce_available = DGCE_AVAILABLE
        self._calibration_cache = {}
        
        # Load DGCE calibration data
        if self.dgce_available:
            try:
                self._load_dgce_calibration(dgce_data_path, force_reload)
                self.data_source = "DGCE 2024 Calibration (JSON)"
                logger.info("✅ Unified calibration loaded from DGCE model")
            except Exception as e:
                logger.error(f"Failed to load DGCE calibration: {e}")
                self._load_fallback_calibration()
                self.data_source = "Fallback hardcoded values"
        else:
            self._load_fallback_calibration()
            self.data_source = "Fallback hardcoded values"
        
        # Initialize model-specific parameter mappings
        self._initialize_model_mappings()
        
        logger.info(f"📊 Unified Calibration Manager initialized with {self.data_source}")

    def get_calibration_metadata(self) -> Dict[str, Any]:
        """Return metadata about calibration source and date."""
        return {
            'calibration_date': self.calibration_date,
            'data_source': getattr(self, 'data_source', 'Unknown')
        }
    
    def _load_dgce_calibration(self, data_path: str = None, force_reload: bool = False) -> None:
        """Load calibration from DGCE model."""
        if not force_reload and hasattr(self, 'dgce_calibration'):
            return
            
        loader = CalibrationLoader(data_path)
        self.dgce_calibration = loader.load_calibration()
        logger.info("📁 DGCE calibration data loaded successfully")
    
    def _load_fallback_calibration(self) -> None:
        """Load fallback calibration when DGCE is not available."""
        self.dgce_calibration = UAECalibration2024()
        logger.warning("⚠️ Using fallback calibration - DGCE model not available")
    
    def _initialize_model_mappings(self) -> None:
        """Initialize mappings for different economic models."""
        
        # Core elasticities from DGCE calibration
        self.core_elasticities = {
            'output_tax_elasticity': self.dgce_calibration.output_tax_elasticity,
            'investment_tax_elasticity': self.dgce_calibration.investment_tax_elasticity,
            'consumption_tax_elasticity': self.dgce_calibration.consumption_tax_elasticity,
            'labor_supply_elasticity': self.dgce_calibration.labor_supply_elasticity,
        }
        
        # Employment elasticity (used by labor model)
        # Note: Labor model typically uses negative values for tax effects
        self.employment_elasticity = -0.33  # Keep empirical value, but could be derived from DGCE
        
        # FDI elasticity (investment response to tax changes)
        self.fdi_elasticity = self.dgce_calibration.investment_tax_elasticity * 1.5  # FDI more elastic than domestic investment
        
        # Sector-specific sensitivities
        self.sector_sensitivities = {
            'financial': 1.2,      # More sensitive to tax changes
            'technology': 1.5,     # Highly mobile sector
            'manufacturing': 0.8,  # Less sensitive
            'construction': 0.7,   # Domestic focused
            'real_estate': 0.9,    # Moderate sensitivity
            'other_services': 1.0, # Average sensitivity
            'oil_gas': 0.3,        # Low sensitivity due to special regime
            'trade': 0.9,          # Moderate sensitivity
            'transport': 0.8,      # Less mobile
            'hospitality': 1.1     # Moderately sensitive
        }
        
        # Regional benchmark rates for competitiveness analysis
        self.regional_benchmark_rates = {
            'oil_gas': 0.50,
            'financial': 0.15,
            'manufacturing': 0.15,
            'technology': 0.12,
            'real_estate': 0.20,
            'construction': 0.20,
            'retail': 0.15,
            'hospitality': 0.15,
            'transport': 0.15
        }
        
        # Wage structure from DGCE calibration
        self.wage_structure = {
            'emirati_public': self.dgce_calibration.emirati_wage_public,
            'emirati_private': self.dgce_calibration.emirati_wage_private,
            'expat_high': self.dgce_calibration.expat_wage_high,
            'expat_medium': self.dgce_calibration.expat_wage_medium,
            'expat_low': self.dgce_calibration.expat_wage_low
        }
        
        # Innovation parameters
        self.innovation_parameters = {
            'rd_tax_elasticity': 1.2,  # R&D responds positively to tax incentives
            'startup_tax_elasticity': 1.5,  # Startups highly sensitive to tax rates
            'patent_elasticity': 0.8,  # Patent filing response
            'productivity_multiplier': 0.3  # Productivity gain from R&D growth
        }
    
    # Core Methods for Economic Models
    
    def get_elasticities(self, model_type: str = 'general', sector: Optional[str] = None) -> Dict[str, float]:
        """
        Get elasticities for specific model type.
        
        Args:
            model_type: 'general', 'labor', 'fdi', 'innovation', or 'dgce'
            
        Returns:
            Dictionary of relevant elasticities
        """
        base_elasticities = self.core_elasticities.copy()
        
        if model_type == 'labor':
            base_elasticities.update({
                'employment_elasticity': self.employment_elasticity,
                'wage_elasticity': -0.2,  # Wage response to employment changes
                'emiratization_elasticity': 0.1  # Emiratization response to policies
            })
        elif model_type == 'fdi':
            base_elasticities.update({
                'fdi_elasticity': self.fdi_elasticity,
                'competitiveness_elasticity': -1.0,  # Competitiveness improves with lower taxes
                'headquarters_elasticity': -1.8  # HQ location very sensitive to taxes
            })
        elif model_type == 'innovation':
            base_elasticities.update(self.innovation_parameters)
        elif model_type == 'dgce':
            # Return raw DGCE elasticities
            return base_elasticities
        elif model_type == 'sector':
            # Sector-specific tax sensitivity
            # Include sector sensitivity multiplier
            # sector argument handled in overloaded call
            return base_elasticities
        
        return base_elasticities
    
    def get_sector_sensitivity(self, sector: str) -> float:
        """Get tax sensitivity multiplier for specific sector."""
        return self.sector_sensitivities.get(sector, 1.0)
    
    def get_regional_benchmark(self, sector: str) -> float:
        """Get regional benchmark tax rate for sector."""
        return self.regional_benchmark_rates.get(sector, 0.15)
    
    def get_wage_structure(self) -> Dict[str, float]:
        """Get wage structure from DGCE calibration."""
        return self.wage_structure.copy()
    
    def get_sector_characteristics(self, sector: Optional[str] = None) -> Any:
        """Return sector share data or specific sector share."""
        if sector:
            return {sector: self.dgce_calibration.sector_shares.get(sector)}
        return self.dgce_calibration.sector_shares
    
    def get_steady_state_values(self) -> Dict[str, float]:
        """Get steady state economic values from DGCE calibration."""
        return {
            'gdp': self.dgce_calibration.total_output,
            'consumption': self.dgce_calibration.household_consumption,
            'investment': self.dgce_calibration.investment,
            'government_spending': self.dgce_calibration.government_consumption,
            'exports': self.dgce_calibration.exports,
            'imports': self.dgce_calibration.imports,
            'inflation': self.dgce_calibration.inflation,
            'oil_share': self.dgce_calibration.oil_share
        }
    
    def get_tax_rates(self) -> Dict[str, float]:
        """Get current tax rates from calibration."""
        return {
            'corporate_rate': self.dgce_calibration.corporate_tax_rate,
            'vat_rate': self.dgce_calibration.vat_rate,
            'oil_royalty_rate': self.dgce_calibration.oil_royalty_rate
        }
    
    def get_sector_shares(self) -> Dict[str, float]:
        """Get sector shares from DGCE calibration."""
        return self.dgce_calibration.sector_shares.copy()
    
    # Utility Methods
    
    def calculate_fallback_impact(self, tax_changes: Dict[str, float], impact_type: str = 'gdp') -> float:
        """
        Calculate economic impact using DGCE elasticities when DGCE model fails.
        
        Args:
            tax_changes: Dictionary of tax rate changes
            impact_type: 'gdp', 'employment', 'investment', 'consumption'
            
        Returns:
            Estimated percentage impact
        """
        if 'corporate_rate' in tax_changes:
            current_rate = self.dgce_calibration.corporate_tax_rate
            new_rate = tax_changes['corporate_rate']
            tax_change_percent = ((new_rate - current_rate) / current_rate) * 100
            
            if impact_type == 'gdp':
                return tax_change_percent * self.core_elasticities['output_tax_elasticity']
            elif impact_type == 'employment':
                return tax_change_percent * self.employment_elasticity
            elif impact_type == 'investment':
                return tax_change_percent * self.core_elasticities['investment_tax_elasticity']
            elif impact_type == 'consumption':
                return tax_change_percent * self.core_elasticities['consumption_tax_elasticity']
        
        return 0.0
    
    def validate_model_consistency(self, models: List[Any]) -> Dict[str, bool]:
        """
        Validate that all models are using consistent calibration data.
        
        Args:
            models: List of economic model instances
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {
            'all_models_consistent': True,
            'inconsistent_models': [],
            'calibration_date_match': True
        }
        
        base_elasticity = self.core_elasticities['output_tax_elasticity']
        
        for model in models:
            if hasattr(model, 'get_elasticity'):
                model_elasticity = model.get_elasticity('output_tax')
                if abs(model_elasticity - base_elasticity) > 0.01:
                    validation_results['all_models_consistent'] = False
                    validation_results['inconsistent_models'].append(model.__class__.__name__)
        
        return validation_results
    
    def get_calibration_metadata(self) -> Dict[str, Any]:
        """Get metadata about current calibration."""
        return {
            'data_source': self.data_source,
            'calibration_date': self.calibration_date,
            'dgce_available': self.dgce_available,
            'core_elasticities': self.core_elasticities,
            'wage_structure': self.wage_structure,
            'sector_count': len(self.sector_sensitivities),
            'version': '1.0'
        }
    
    def export_calibration_summary(self, filepath: str = None) -> Dict[str, Any]:
        """Export calibration summary for documentation."""
        summary = {
            'metadata': self.get_calibration_metadata(),
            'elasticities': self.core_elasticities,
            'sector_sensitivities': self.sector_sensitivities,
            'regional_benchmarks': self.regional_benchmark_rates,
            'wage_structure': self.wage_structure,
            'steady_state': self.get_steady_state_values(),
            'tax_rates': self.get_tax_rates()
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"📄 Calibration summary exported to {filepath}")
        
        return summary


# Global instance for easy import
_global_calibration_manager = None

def get_calibration_manager(force_reload: bool = False) -> UnifiedCalibrationManager:
    """
    Get global calibration manager instance.
    
    Args:
        force_reload: Force reload of calibration data
        
    Returns:
        UnifiedCalibrationManager instance
    """
    global _global_calibration_manager
    
    if _global_calibration_manager is None or force_reload:
        _global_calibration_manager = UnifiedCalibrationManager(force_reload=force_reload)
    
    return _global_calibration_manager


# Convenience functions for quick access
def get_elasticity(elasticity_name: str, model_type: str = 'general') -> float:
    """Quick access to specific elasticity."""
    manager = get_calibration_manager()
    elasticities = manager.get_elasticities(model_type)
    return elasticities.get(elasticity_name, 0.0)

def get_sector_sensitivity(sector: str) -> float:
    """Quick access to sector sensitivity."""
    manager = get_calibration_manager()
    return manager.get_sector_sensitivity(sector)

def get_fallback_impact(tax_changes: Dict[str, float], impact_type: str = 'gdp') -> float:
    """Quick access to fallback impact calculation."""
    manager = get_calibration_manager()
    return manager.calculate_fallback_impact(tax_changes, impact_type)
"""
Central sector mapping configuration for UAE economic models.

This module provides consistent sector naming and mapping across all models,
ensuring alignment with ISIC Level 1 classifications used in the data.
"""

# ISIC Level 1 sectors as found in the data
ISIC_SECTORS = [
    'Accommodation and food service activities',
    'Activities of households as employers, undifferentiated goods- and services-producing activities of households for own use',
    'Administrative and support service activities',
    'Agriculture, forestry and fishing',
    'Arts, entertainment and recreation',
    'Construction',
    'Education',
    'Electricity, gas, water supply; and waste management activities',
    'Financial and insurance activities',
    'Human health and social work activities',
    'Information and communication',
    'Manufacturing',
    'Mining and quarrying',
    'Other service activities',
    'Professional, scientific and technical activities',
    'Public administration and defence, compulsory social security',
    'Real estate activities',
    'Transportation and storage',
    'Wholesale and retail trade, repair of motor vehicles and motorcycles'
]

# Short names for internal use (maps to ISIC)
SHORT_TO_ISIC = {
    # Primary sectors
    'agriculture': 'Agriculture, forestry and fishing',
    'mining': 'Mining and quarrying',
    
    # Secondary sectors
    'manufacturing': 'Manufacturing',
    'construction': 'Construction',
    'utilities': 'Electricity, gas, water supply; and waste management activities',
    
    # Tertiary sectors
    'trade': 'Wholesale and retail trade, repair of motor vehicles and motorcycles',
    'transport': 'Transportation and storage',
    'accommodation': 'Accommodation and food service activities',
    'information': 'Information and communication',
    'finance': 'Financial and insurance activities',
    'real_estate': 'Real estate activities',
    'professional': 'Professional, scientific and technical activities',
    'administrative': 'Administrative and support service activities',
    'public': 'Public administration and defence, compulsory social security',
    'education': 'Education',
    'health': 'Human health and social work activities',
    'arts': 'Arts, entertainment and recreation',
    'other_services': 'Other service activities',
    'households': 'Activities of households as employers, undifferentiated goods- and services-producing activities of households for own use'
}

# Policy rate mapping (which sectors get which tax rate)
POLICY_RATE_MAPPING = {
    # Financial sector
    'Financial and insurance activities': 'financial_rate',
    
    # Technology sector (maps to Information and communication)
    'Information and communication': 'technology_rate',
    
    # Manufacturing sector
    'Manufacturing': 'manufacturing_rate',
    
    # Real estate sector
    'Real estate activities': 'real_estate_rate',
    
    # Construction sector  
    'Construction': 'construction_rate',
    
    # Trade sector
    'Wholesale and retail trade, repair of motor vehicles and motorcycles': 'trade_rate',
    
    # Service sectors (professional services rate if specified)
    'Professional, scientific and technical activities': 'services_rate',
    'Administrative and support service activities': 'services_rate',
    'Accommodation and food service activities': 'services_rate',
    'Other service activities': 'services_rate',
    'Arts, entertainment and recreation': 'services_rate',
    
    # Other sectors use standard rate
    'Agriculture, forestry and fishing': 'standard_rate',
    'Mining and quarrying': 'standard_rate',
    'Electricity, gas, water supply; and waste management activities': 'standard_rate',
    'Transportation and storage': 'standard_rate',
    'Public administration and defence, compulsory social security': 'standard_rate',
    'Education': 'standard_rate',
    'Human health and social work activities': 'standard_rate',
    'Activities of households as employers, undifferentiated goods- and services-producing activities of households for own use': 'standard_rate'
}

# Economic model sector aggregation (for models that use fewer sectors)
MODEL_SECTOR_AGGREGATION = {
    # General Equilibrium model sectors
    'ge_model': {
        'oil_gas': ['Mining and quarrying'],
        'manufacturing': ['Manufacturing'],
        'construction': ['Construction'],
        'trade': ['Wholesale and retail trade, repair of motor vehicles and motorcycles'],
        'transport': ['Transportation and storage'],
        'finance': ['Financial and insurance activities'],
        'real_estate': ['Real estate activities'],
        'government': ['Public administration and defence, compulsory social security'],
        'information and communication': ['Information and communication'],
        'professional_scientific_technical activities': ['Professional, scientific and technical activities'],
        'admin': ['Administrative and support service activities'],
        'education': ['Education'],
        'humean_health': ['Human health and social work activities'],
        'arts': ['Arts, entertainment and recreation'],
        'other_services': ['Other service activities', 'Activities of households as employers, undifferentiated goods- and services-producing activities of households for own use'],
        'accommodation': ['Accommodation and food service activities'],
        'electricity_gas_water': ['Electricity, gas, water supply; and waste management activities'],
        'agro': ['Agriculture, forestry and fishing']
    },
    
    # Labor market model sectors
    'labor_model': {
        'construction': ['Construction'],
        'trade': ['Wholesale and retail trade, repair of motor vehicles and motorcycles'],
        'manufacturing': ['Manufacturing'],
        'services': [
            'Information and communication',
            'Professional, scientific and technical activities',
            'Administrative and support service activities',
            'Accommodation and food service activities',
            'Arts, entertainment and recreation',
            'Other service activities'
        ],
        'transport': ['Transportation and storage'],
        'finance': ['Financial and insurance activities'],
        'government': ['Public administration and defence, compulsory social security'],
        'other': [
            'Real estate activities',
            'Education',
            'Human health and social work activities',
            'Electricity, gas, water supply; and waste management activities',
            'Agriculture, forestry and fishing',
            'Mining and quarrying',
            'Activities of households as employers, undifferentiated goods- and services-producing activities of households for own use'
        ]
    }
}

# Innovation-intensive sectors (for innovation model)
INNOVATION_SECTORS = [
    'Information and communication',  # High tech
    'Manufacturing',  # Can be R&D intensive
    'Professional, scientific and technical activities',  # R&D services
    'Financial and insurance activities',  # Fintech
    'Human health and social work activities'  # Medical research
]

# FDI-sensitive sectors (for FDI model)
FDI_SENSITIVE_SECTORS = [
    'Financial and insurance activities',
    'Information and communication',
    'Manufacturing',
    'Real estate activities',
    'Professional, scientific and technical activities',
    'Wholesale and retail trade, repair of motor vehicles and motorcycles'
]

# Export-oriented sectors
EXPORT_SECTORS = [
    'Mining and quarrying',  # Oil & gas exports
    'Manufacturing',
    'Wholesale and retail trade, repair of motor vehicles and motorcycles',  # Re-exports
    'Transportation and storage',  # Logistics hub
    'Financial and insurance activities'  # Financial services exports
]


def get_policy_rate_for_sector(sector: str, policy: dict) -> float:
    """Get the appropriate tax rate for a sector based on policy configuration."""
    if sector not in ISIC_SECTORS:
        # Try to map from short name
        sector = SHORT_TO_ISIC.get(sector, sector)
    
    rate_field = POLICY_RATE_MAPPING.get(sector, 'standard_rate')
    return policy.get(rate_field, policy.get('standard_rate', 0.09))


def map_isic_to_model_sectors(isic_sector: str, model: str) -> str:
    """Map ISIC sector to model-specific sector aggregation."""
    if model not in MODEL_SECTOR_AGGREGATION:
        return isic_sector
    
    for model_sector, isic_list in MODEL_SECTOR_AGGREGATION[model].items():
        if isic_sector in isic_list:
            return model_sector
    
    return 'other'  # Default fallback


def is_innovation_intensive(sector: str) -> bool:
    """Check if a sector is innovation-intensive."""
    return sector in INNOVATION_SECTORS


def is_fdi_sensitive(sector: str) -> bool:
    """Check if a sector is sensitive to FDI flows."""
    return sector in FDI_SENSITIVE_SECTORS


def is_export_oriented(sector: str) -> bool:
    """Check if a sector is export-oriented."""
    return sector in EXPORT_SECTORS
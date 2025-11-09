"""
Business characteristics and entity-specific variables.
"""

from dgce_model.openfisca_core.model_api import *
from dgce_model.entities import Business


class is_resident(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business is a UAE tax resident"
    reference = "Federal Decree-Law No. 47 of 2022, Article 11"
    documentation = """
    A business is considered UAE tax resident if:
    - It is incorporated in the UAE, or
    - It is effectively managed and controlled in the UAE
    """
    default_value = True


class is_free_zone_person(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business is a free zone person"
    documentation = "Business is registered and operates in a UAE free zone"
    default_value = False

    def formula(business, period, parameters):
        """Return ``True`` if the firm is recorded as free-zone resident."""
        return bool(business("is_free_zone", period))


# is_qualifying_free_zone_person is defined in free_zones.py


class has_adequate_substance(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business has adequate substance in UAE"
    documentation = "Business conducts core income-generating activities in UAE"
    default_value = True


class derives_qualifying_income(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business derives qualifying income"
    documentation = "Business income is from qualifying activities per regulations"
    default_value = True


class maintains_adequate_accounts(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business maintains adequate accounting records"
    default_value = True


class complies_with_transfer_pricing(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business complies with transfer pricing regulations"
    default_value = True


class business_sector(Variable):
    value_type = Enum
    possible_values = Enum('BusinessSector', [
        'manufacturing',
        'trading',
        'services',
        'banking',
        'insurance',
        'real_estate',
        'oil_gas',
        'technology',
        'other'
    ])
    default_value = possible_values.other
    entity = Business
    definition_period = YEAR
    label = "Primary business sector"


class number_of_employees(Variable):
    value_type = int
    entity = Business
    definition_period = YEAR
    label = "Number of employees"
    documentation = "Total number of employees in the business"
    default_value = 0


class emirate_of_operation(Variable):
    value_type = Enum
    possible_values = Enum('Emirate', [
        'abu_dhabi',
        'dubai',
        'sharjah',
        'ajman',
        'umm_al_quwain',
        'ras_al_khaimah',
        'fujairah'
    ])
    default_value = possible_values.dubai
    entity = Business
    definition_period = YEAR
    label = "Primary emirate of operation"
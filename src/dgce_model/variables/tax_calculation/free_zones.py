"""
Enhanced Free Zone taxation rules for UAE corporate tax.

Free zones offer special tax treatment for qualifying businesses,
with 0% tax on qualifying income and standard rates on non-qualifying income.
"""

from dgce_model.openfisca_core.model_api import *
from dgce_model.entities import Business


# Qualifying Activities
class derives_income_from_qualifying_activity(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Derives income from qualifying activities"
    reference = "Cabinet Decision No. 55 of 2023"
    documentation = """
    Qualifying activities include:
    - Manufacturing of goods
    - Processing of goods or materials
    - Holding of shares and other securities
    - Ownership, management and operation of ships
    - Reinsurance services
    - Fund management services
    - Wealth and investment management services
    - Headquarter services to related parties
    - Treasury and financing services to related parties
    - Distribution of goods or materials in designated zones
    - Logistics services
    """
    
    def formula(business, period, parameters):
        sector = business('business_sector', period)
        
        # Map sectors to qualifying status
        qualifying_sectors = [
            b'manufacturing',
            b'trading',  # If in designated zone
            b'services'  # Specific services only
        ]
        
        # Simplified - would need more detailed activity classification
        return business('is_free_zone_person', period) * \
               (sector == sector.possible_values.manufacturing)


class has_excluded_activities(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Has excluded activities"
    reference = "Ministerial Decision No. 139 of 2023"
    documentation = """
    Excluded activities that prevent QFZP status:
    - Regulated banking activities
    - Insurance activities
    - Finance and leasing activities
    - Ownership or exploitation of immovable property (except specific cases)
    - Activities specifically listed as excluded
    """
    
    def formula(business, period, parameters):
        sector = business('business_sector', period)
        
        excluded_sectors = [
            b'banking',
            b'insurance',
            b'real_estate'  # With exceptions
        ]
        
        # Check if in excluded sectors
        is_excluded = select(
            [sector == s for s in excluded_sectors],
            [True] * len(excluded_sectors),
            default=False
        )
        
        return is_excluded


# Income categorization
class qualifying_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Qualifying income for free zone"
    reference = "Cabinet Decision No. 100 of 2023"
    documentation = """
    Income that qualifies for 0% tax rate:
    - Income from transactions with other Free Zone Persons
    - Income from export of goods outside UAE
    - Income from qualifying activities
    - Certain other specified income types
    """
    
    def formula(business, period, parameters):
        is_qfzp = business('is_qualifying_free_zone_person', period)
        total_revenue = business('revenue', period)
        
        # Percentage of qualifying income (would come from data)
        qualifying_percentage = business('qualifying_income_percentage', period)
        
        return where(is_qfzp, total_revenue * qualifying_percentage, 0)


class qualifying_income_percentage(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Percentage of income that is qualifying"
    documentation = "Proportion of total income from qualifying activities/transactions"
    default_value = 0.9  # 90% qualifying by default for QFZP


class non_qualifying_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Non-qualifying income for free zone"
    
    def formula(business, period, parameters):
        is_qfzp = business('is_qualifying_free_zone_person', period)
        total_revenue = business('revenue', period)
        qualifying_income = business('qualifying_income', period)
        
        return where(is_qfzp, total_revenue - qualifying_income, 0)


# Enhanced QFZP determination
class is_qualifying_free_zone_person(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business is a Qualifying Free Zone Person"
    reference = "Federal Decree-Law No. 47 of 2022, Article 18"
    
    def formula(business, period, parameters):
        # Basic requirements
        is_fz = business('is_free_zone_person', period)
        has_substance = business('has_adequate_substance', period)
        maintains_accounts = business('maintains_adequate_accounts', period)
        complies_tp = business('complies_with_transfer_pricing', period)
        
        # Activity requirements
        has_qualifying_activities = business('derives_income_from_qualifying_activity', period)
        no_excluded_activities = not_(business('has_excluded_activities', period))
        
        # Additional requirements
        meets_de_minimis = business('meets_de_minimis_requirement', period)
        
        return (is_fz * has_substance * maintains_accounts * complies_tp * 
                has_qualifying_activities * no_excluded_activities * meets_de_minimis)


class meets_de_minimis_requirement(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Meets de minimis requirement for non-qualifying income"
    
    def formula(business, period, parameters):
        total_revenue = business('revenue', period)
        non_qualifying = business('non_qualifying_income', period)
        
        absolute_limit = parameters(period).free_zones.de_minimis_absolute
        percentage_limit = parameters(period).free_zones.de_minimis_percentage
        
        # Must be under both limits
        under_absolute = non_qualifying <= absolute_limit
        under_percentage = non_qualifying <= (total_revenue * percentage_limit)
        
        return under_absolute * under_percentage


# Free zone tax calculation
class free_zone_taxable_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Taxable income for free zone person"
    
    def formula(business, period, parameters):
        is_qfzp = business('is_qualifying_free_zone_person', period)
        is_fz = business('is_free_zone_person', period)
        
        # For QFZP: only non-qualifying income is taxable
        if is_qfzp.any():
            non_qualifying = business('non_qualifying_income', period)
            
            # Apply same deductions calculation as standard
            revenue_ratio = non_qualifying / business('revenue', period)
            total_expenses = (
                business('cost_of_goods_sold', period) +
                business('deductible_expenses', period)
            )
            
            # Pro-rate expenses to non-qualifying income
            allocated_expenses = total_expenses * revenue_ratio
            
            return where(is_qfzp, max_(non_qualifying - allocated_expenses, 0), 0)
        
        # For non-QFZP free zone persons: standard calculation
        return where(is_fz * not_(is_qfzp), business('taxable_income', period), 0)


class free_zone_corporate_tax(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Corporate tax for free zone person"
    reference = "Federal Decree-Law No. 47 of 2022, Article 18"
    
    def formula(business, period, parameters):
        is_qfzp = business('is_qualifying_free_zone_person', period)
        is_fz = business('is_free_zone_person', period)
        
        # QFZP with de minimis: 0% tax
        meets_de_minimis = business('meets_de_minimis_requirement', period)
        
        if (is_qfzp * meets_de_minimis).any():
            return business.empty_array()
        
        # QFZP exceeding de minimis: tax on non-qualifying income
        if is_qfzp.any():
            taxable_income = business('free_zone_taxable_income', period)
            threshold = parameters(period).corporate_tax.taxable_income_threshold
            rate = parameters(period).corporate_tax.standard_rate
            
            taxable_above_threshold = max_(taxable_income - threshold, 0)
            return where(is_qfzp, taxable_above_threshold * rate, 0)
        
        # Non-QFZP free zone: standard tax
        return where(is_fz * not_(is_qfzp), business('corporate_tax', period), 0)


# Transitional rules
class free_zone_transitional_relief(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Eligible for free zone transitional relief"
    documentation = """
    Businesses registered before December 2022 may have transitional relief
    maintaining 0% rate until specified dates depending on their jurisdiction.
    """
    
    def formula(business, period, parameters):
        # This would check registration date and specific free zone rules
        # Simplified for now
        return False


# Mainland branch income
class mainland_branch_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Income from mainland branch"
    documentation = """
    Income derived by a Free Zone Person from a mainland branch
    is generally subject to standard corporate tax.
    """
    default_value = 0


class free_zone_effective_tax_rate(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Effective tax rate for free zone person"
    
    def formula(business, period, parameters):
        is_fz = business('is_free_zone_person', period)
        
        if not is_fz.any():
            return business.empty_array()
        
        total_revenue = business('revenue', period)
        
        # Get appropriate tax amount
        is_qfzp = business('is_qualifying_free_zone_person', period)
        tax = where(
            is_qfzp,
            business('free_zone_corporate_tax', period),
            business('corporate_tax', period)
        )
        
        return where(total_revenue > 0, tax / total_revenue, 0)
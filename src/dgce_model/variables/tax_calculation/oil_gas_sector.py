"""
Oil and gas sector special taxation rules for UAE corporate tax.

This module implements the 55% tax rate for oil and gas companies as per
Federal Decree-Law No. 47 of 2022, Article 8.
"""

from dgce_model.openfisca_core.model_api import *
from dgce_model.entities import Business
import numpy as np


class is_oil_gas_company(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business engaged in oil and gas extraction/production"
    reference = "Federal Decree-Law No. 47 of 2022, Article 8"
    
    def formula(business, period, parameters):
        # Primary identification through sector
        sector = business('business_sector', period)
        is_extractive = sector == b'mining'
        
        # Additional checks for oil/gas activities
        has_petroleum_license = business('has_petroleum_license', period)
        extracts_hydrocarbons = business('extracts_hydrocarbons', period)
        
        # Can be identified by any of these criteria
        return is_extractive * (has_petroleum_license + extracts_hydrocarbons)


class has_petroleum_license(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Has petroleum concession or license"
    reference = "Petroleum agreements with UAE government"
    default_value = False


class extracts_hydrocarbons(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Engaged in extraction of oil, gas, or other hydrocarbons"
    default_value = False


class oil_gas_revenue(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Revenue from oil and gas extraction activities"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        total_revenue = business('revenue', period)
        
        # For oil/gas companies, segregate O&G revenue
        oil_gas_percentage = business('oil_gas_revenue_percentage', period)
        
        return where(
            is_oil_gas,
            total_revenue * oil_gas_percentage,
            0
        )


class oil_gas_revenue_percentage(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Percentage of revenue from oil/gas activities"
    default_value = 1.0  # Assume 100% for pure O&G companies


class non_oil_gas_revenue(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Revenue from non-oil/gas activities"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        total_revenue = business('revenue', period)
        oil_gas_rev = business('oil_gas_revenue', period)
        
        return where(
            is_oil_gas,
            total_revenue - oil_gas_rev,
            0
        )


class oil_gas_taxable_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Taxable income from oil and gas activities"
    reference = "Federal Decree-Law No. 47 of 2022, Article 8"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # Calculate O&G specific taxable income
        oil_gas_rev = business('oil_gas_revenue', period)
        
        # Deductible expenses for O&G operations
        oil_gas_expenses = business('oil_gas_deductible_expenses', period)
        
        # Special deductions for O&G sector
        depletion_allowance = business('depletion_allowance', period)
        exploration_costs = business('exploration_cost_deduction', period)
        
        taxable = oil_gas_rev - oil_gas_expenses - depletion_allowance - exploration_costs
        
        return where(is_oil_gas, max_(taxable, 0), 0)


class oil_gas_deductible_expenses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Deductible expenses for oil/gas operations"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # Operating expenses specific to O&G
        operating_expenses = business('operating_expenses', period)
        oil_gas_percentage = business('oil_gas_revenue_percentage', period)
        
        # Allocate expenses proportionally
        oil_gas_opex = operating_expenses * oil_gas_percentage
        
        # Add O&G specific costs
        production_costs = business('oil_gas_production_costs', period)
        
        return where(is_oil_gas, oil_gas_opex + production_costs, 0)


class oil_gas_production_costs(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Direct costs of oil/gas production"
    default_value = 0


class depletion_allowance(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Depletion allowance for oil/gas reserves"
    reference = "Standard depletion calculation methods"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # Depletion rate (typically percentage of revenue or unit of production)
        depletion_rate = parameters(period).corporate_tax.oil_gas.depletion_rate
        oil_gas_rev = business('oil_gas_revenue', period)
        
        # Simple percentage depletion (actual would be more complex)
        depletion = oil_gas_rev * depletion_rate
        
        return where(is_oil_gas, depletion, 0)


class exploration_cost_deduction(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Deduction for oil/gas exploration costs"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # Exploration costs can typically be expensed immediately
        exploration_costs = business('current_year_exploration_costs', period)
        
        return where(is_oil_gas, exploration_costs, 0)


class current_year_exploration_costs(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Exploration costs incurred in current year"
    default_value = 0


class oil_gas_tax(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Corporate tax on oil and gas activities at 55%"
    reference = "Federal Decree-Law No. 47 of 2022, Article 8"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # Taxable income from O&G activities
        oil_gas_taxable = business('oil_gas_taxable_income', period)
        
        # Apply 55% rate
        oil_gas_rate = parameters(period).corporate_tax.oil_gas.tax_rate
        
        oil_gas_tax_amount = oil_gas_taxable * oil_gas_rate
        
        return where(is_oil_gas, oil_gas_tax_amount, 0)


class standard_corporate_tax_on_non_oil_gas(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Standard 9% tax on non-oil/gas income"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # Non-O&G revenue
        non_og_revenue = business('non_oil_gas_revenue', period)
        
        # Apply standard calculation to non-O&G income
        if not is_oil_gas.any():
            return business.empty_array()
        
        # Calculate expenses attributable to non-O&G
        total_expenses = business('total_expenses', period)
        oil_gas_expenses = business('oil_gas_deductible_expenses', period)
        non_og_expenses = total_expenses - oil_gas_expenses
        
        # Non-O&G taxable income
        non_og_taxable = max_(non_og_revenue - non_og_expenses, 0)
        
        # Apply standard rate with threshold
        threshold = parameters(period).corporate_tax.taxable_income_threshold
        standard_rate = parameters(period).corporate_tax.standard_rate
        
        taxable_above_threshold = max_(non_og_taxable - threshold, 0)
        standard_tax = taxable_above_threshold * standard_rate
        
        return where(is_oil_gas, standard_tax, 0)


class total_corporate_tax_with_oil_gas(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Total corporate tax including oil/gas sector rules"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        # For O&G companies: 55% on O&G + 9% on other
        oil_gas_tax_amount = business('oil_gas_tax', period)
        standard_tax_non_og = business('standard_corporate_tax_on_non_oil_gas', period)
        
        # For non-O&G companies: standard calculation
        standard_tax = business('corporate_tax', period)
        
        return where(
            is_oil_gas,
            oil_gas_tax_amount + standard_tax_non_og,
            standard_tax
        )


# Special provisions for National Oil Companies
class is_national_oil_company(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Is a National Oil Company (NOC)"
    reference = "ADNOC and subsidiaries"
    default_value = False


class noc_special_provisions(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Special provisions/adjustments for NOCs"
    
    def formula(business, period, parameters):
        is_noc = business('is_national_oil_company', period)
        
        # NOCs may have special agreements
        # This is a placeholder for specific provisions
        return where(is_noc, 0, 0)


# Service companies in O&G sector
class is_oil_gas_service_company(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Provides services to oil/gas sector"
    default_value = False


class oil_gas_service_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Income from providing services to O&G sector"
    
    def formula(business, period, parameters):
        is_service = business('is_oil_gas_service_company', period)
        total_revenue = business('revenue', period)
        
        # Service companies are taxed at standard 9% rate
        # This variable helps track O&G related income
        service_percentage = business('oil_gas_service_percentage', period)
        
        return where(is_service, total_revenue * service_percentage, 0)


class oil_gas_service_percentage(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Percentage of revenue from O&G sector services"
    default_value = 0


# Ring-fencing provisions
class oil_gas_ring_fenced_losses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Ring-fenced losses from O&G activities"
    reference = "O&G losses cannot offset non-O&G income"
    
    def formula(business, period, parameters):
        is_oil_gas = business('is_oil_gas_company', period)
        
        oil_gas_taxable = business('oil_gas_taxable_income', period)
        
        # If O&G operations have losses, they are ring-fenced
        ring_fenced = where(oil_gas_taxable < 0, -oil_gas_taxable, 0)
        
        return where(is_oil_gas, ring_fenced, 0)
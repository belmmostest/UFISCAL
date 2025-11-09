"""
Core corporate tax calculation variables for UAE.
"""

from dgce_model.openfisca_core.model_api import *
from dgce_model.entities import Business


class taxable_income_before_losses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Taxable income before loss offset"
    reference = "Federal Decree-Law No. 47 of 2022, Article 20"
    
    def formula(business, period, parameters):
        accounting_profit = business('accounting_profit', period)
        
        # Add back non-deductible expenses
        interest_expense = business('interest_expense', period)
        deductible_interest = business('deductible_interest', period)
        non_deductible_interest = interest_expense - deductible_interest
        
        entertainment_clients = business('entertainment_expenses_clients', period)
        deductible_entertainment = business('deductible_entertainment_expenses', period)
        entertainment_employees = business('entertainment_expenses_employees', period)
        non_deductible_entertainment = (entertainment_clients + entertainment_employees) - deductible_entertainment
        
        # Taxable income = accounting profit + non-deductible expenses
        return accounting_profit + non_deductible_interest + non_deductible_entertainment


class taxable_income(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Taxable income"
    reference = "Federal Decree-Law No. 47 of 2022, Article 20"
    
    def formula(business, period, parameters):
        income_before_losses = business('taxable_income_before_losses', period)
        deductible_losses = business('deductible_tax_losses', period)
        
        return max_(income_before_losses - deductible_losses, 0)


class small_business_relief_eligible(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Eligible for small business relief"
    reference = "Federal Decree-Law No. 47 of 2022, Article 21"
    
    def formula(business, period, parameters):
        revenue = business('revenue', period)
        threshold = parameters(period).small_business.revenue_threshold
        has_elected = business('small_business_relief_election', period)
        
        # Check if threshold exists (relief expires in 2027)
        if threshold is None:
            return False
            
        return (revenue <= threshold) * has_elected


class small_business_relief_election(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business has elected for small business relief"
    default_value = False


class corporate_tax_before_reliefs(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Corporate tax before reliefs"
    
    def formula(business, period, parameters):
        taxable_income = business('taxable_income', period)
        threshold = parameters(period).corporate_tax.taxable_income_threshold
        rate = parameters(period).corporate_tax.standard_rate
        
        # 0% on first 375,000 AED, 9% on excess
        taxable_above_threshold = max_(taxable_income - threshold, 0)
        
        return taxable_above_threshold * rate


class corporate_tax(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Corporate income tax"
    reference = "Federal Decree-Law No. 47 of 2022, Article 12"
    
    def formula(business, period, parameters):
        # Check if it's an oil & gas company first
        is_oil_gas = business('is_oil_gas_company', period)
        
        if is_oil_gas.any():
            # Use special O&G calculation
            return business('total_corporate_tax_with_oil_gas', period)
        
        tax_before_reliefs = business('corporate_tax_before_reliefs', period)
        small_business_eligible = business('small_business_relief_eligible', period)
        
        # If eligible for small business relief, tax is 0
        if small_business_eligible:
            return 0
            
        return tax_before_reliefs


class effective_tax_rate(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Effective corporate tax rate"
    
    def formula(business, period, parameters):
        corporate_tax = business('corporate_tax', period)
        accounting_profit = business('accounting_profit', period)
        
        # Avoid division by zero
        if accounting_profit <= 0:
            return 0
            
        return corporate_tax / accounting_profit


class marginal_tax_rate(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Marginal corporate tax rate"
    
    def formula(business, period, parameters):
        taxable_income = business('taxable_income', period)
        threshold = parameters(period).corporate_tax.taxable_income_threshold
        rate = parameters(period).corporate_tax.standard_rate
        small_business_eligible = business('small_business_relief_eligible', period)
        
        if small_business_eligible:
            return 0
        elif taxable_income <= threshold:
            return 0
        else:
            return rate
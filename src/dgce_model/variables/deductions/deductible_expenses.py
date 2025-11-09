"""
Deductible expenses and limitations for UAE corporate tax.
"""

from dgce_model.openfisca_core.model_api import *
from dgce_model.entities import Business
import numpy as np


class deductible_expenses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Total deductible expenses"
    reference = "Federal Decree-Law No. 47 of 2022, Articles 28-33"
    
    def formula(business, period, parameters):
        operating_expenses = business('operating_expenses', period)
        deductible_interest = business('deductible_interest', period)
        depreciation = business('depreciation_expense', period)
        deductible_entertainment = business('deductible_entertainment_expenses', period)
        other_deductible = business('other_deductible_expenses', period)
        
        return operating_expenses + deductible_interest + depreciation + deductible_entertainment + other_deductible


class net_interest_expense(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Net interest expense"
    reference = "Federal Decree-Law No. 47 of 2022, Article 30"
    
    def formula(business, period, parameters):
        interest_expense = business('interest_expense', period)
        interest_income = business('interest_income', period)
        return max(interest_expense - interest_income, 0)


class ebitda(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Earnings Before Interest, Tax, Depreciation and Amortization"
    documentation = "Tax EBITDA for interest limitation calculation"
    
    def formula(business, period, parameters):
        accounting_profit = business('accounting_profit', period)
        interest_expense = business('interest_expense', period)
        depreciation = business('depreciation_expense', period)
        return accounting_profit + interest_expense + depreciation


class deductible_interest(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Deductible interest expense"
    reference = "Federal Decree-Law No. 47 of 2022, Article 30"
    
    def formula(business, period, parameters):
        net_interest = business('net_interest_expense', period)
        ebitda = business('ebitda', period)
        interest_limit = parameters(period).deductions.interest_expense_limit
        
        # Interest deduction limited to 30% of EBITDA
        maximum_deductible = ebitda * interest_limit
        
        return min(net_interest, maximum_deductible)


class entertainment_expenses_clients(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Entertainment expenses for clients"
    documentation = "Entertainment expenses incurred for business promotion with clients"


class entertainment_expenses_employees(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Entertainment expenses for employees"
    documentation = "Entertainment expenses for employee events and welfare"


class deductible_entertainment_expenses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Deductible entertainment expenses"
    reference = "Federal Decree-Law No. 47 of 2022, Article 28"
    
    def formula(business, period, parameters):
        client_entertainment = business('entertainment_expenses_clients', period)
        employee_entertainment = business('entertainment_expenses_employees', period)
        
        client_limit = parameters(period).deductions.entertainment_client_limit
        
        # 50% of client entertainment, 100% of employee entertainment
        deductible_client = client_entertainment * client_limit
        deductible_employee = employee_entertainment
        
        return deductible_client + deductible_employee


class other_deductible_expenses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Other deductible expenses"
    documentation = "Other business expenses that are fully deductible"


class tax_losses_brought_forward(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Tax losses brought forward from previous periods"
    documentation = "Accumulated tax losses from previous periods available for offset"


class deductible_tax_losses(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Deductible tax losses"
    reference = "Federal Decree-Law No. 47 of 2022, Article 37"
    
    def formula(business, period, parameters):
        taxable_income_before_losses = business('taxable_income_before_losses', period)
        losses_brought_forward = business('tax_losses_brought_forward', period)
        loss_offset_limit = parameters(period).deductions.tax_loss_offset_limit
        
        # Losses can offset up to 75% of taxable income
        maximum_offset = taxable_income_before_losses * loss_offset_limit
        
        return min(losses_brought_forward, maximum_offset)
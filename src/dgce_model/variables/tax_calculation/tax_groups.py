"""
Tax Group consolidation variables for UAE corporate tax.

Tax groups allow companies with 95%+ ownership to file consolidated returns,
eliminating intra-group transactions and pooling profits/losses.
"""

from dgce_model.openfisca_core.model_api import *
from dgce_model.entities import Business
from dgce_model.openfisca_core.entities import GroupEntity as TaxGroup


# Business-level variables for group membership

class is_tax_group_parent(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business is a tax group parent"
    reference = "Federal Decree-Law No. 47 of 2022, Article 40"
    default_value = False


class is_tax_group_member(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Business is a member of a tax group"
    reference = "Federal Decree-Law No. 47 of 2022, Article 40"
    default_value = False


class parent_ownership_percentage(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Percentage owned by parent company"
    documentation = "For subsidiaries, the percentage of ownership by parent"
    default_value = 0


class same_financial_year_end(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Has same financial year end as parent"
    default_value = True


class tax_group_eligible(Variable):
    value_type = bool
    entity = Business
    definition_period = YEAR
    label = "Eligible to be in a tax group"
    reference = "Federal Decree-Law No. 47 of 2022, Article 40"
    
    def formula(business, period, parameters):
        ownership = business('parent_ownership_percentage', period)
        same_year = business('same_financial_year_end', period)
        is_resident = business('is_resident', period)
        
        ownership_threshold = parameters(period).tax_group.ownership_threshold
        
        # Must be 95%+ owned, same year end, and resident
        return (ownership >= ownership_threshold) * same_year * is_resident


# Tax Group level variables

class tax_group_members_count(Variable):
    value_type = int
    entity = TaxGroup
    definition_period = YEAR
    label = "Number of members in the tax group"
    
    def formula(tax_group, period, parameters):
        # Count subsidiaries plus parent
        return tax_group.nb_persons(TaxGroup.SUBSIDIARY) + 1


class tax_group_total_revenue(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Total revenue of all group members"
    
    def formula(tax_group, period, parameters):
        parent_revenue = tax_group.parent('revenue', period)
        subsidiaries_revenue = tax_group.sum(
            tax_group.members('revenue', period),
            role=TaxGroup.SUBSIDIARY
        )
        
        return parent_revenue + subsidiaries_revenue


class tax_group_intra_transactions(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Intra-group transactions to be eliminated"
    documentation = """
    Transactions between group members that must be eliminated
    in consolidation. This would typically come from accounting data.
    """
    default_value = 0


class tax_group_consolidated_revenue(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Consolidated revenue after eliminations"
    
    def formula(tax_group, period, parameters):
        total_revenue = tax_group('tax_group_total_revenue', period)
        eliminations = tax_group('tax_group_intra_transactions', period)
        
        return total_revenue - eliminations


class tax_group_total_expenses(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Total expenses of all group members"
    
    def formula(tax_group, period, parameters):
        # Get all expense components for parent
        parent_cogs = tax_group.parent('cost_of_goods_sold', period)
        parent_operating = tax_group.parent('operating_expenses', period)
        parent_interest = tax_group.parent('interest_expense', period)
        parent_depreciation = tax_group.parent('depreciation_expense', period)
        parent_other = tax_group.parent('other_expenses', period)
        
        # Get all expense components for subsidiaries
        sub_cogs = tax_group.sum(
            tax_group.members('cost_of_goods_sold', period),
            role=TaxGroup.SUBSIDIARY
        )
        sub_operating = tax_group.sum(
            tax_group.members('operating_expenses', period),
            role=TaxGroup.SUBSIDIARY
        )
        sub_interest = tax_group.sum(
            tax_group.members('interest_expense', period),
            role=TaxGroup.SUBSIDIARY
        )
        sub_depreciation = tax_group.sum(
            tax_group.members('depreciation_expense', period),
            role=TaxGroup.SUBSIDIARY
        )
        sub_other = tax_group.sum(
            tax_group.members('other_expenses', period),
            role=TaxGroup.SUBSIDIARY
        )
        
        total_expenses = (
            parent_cogs + parent_operating + parent_interest + 
            parent_depreciation + parent_other +
            sub_cogs + sub_operating + sub_interest + 
            sub_depreciation + sub_other
        )
        
        return total_expenses


class tax_group_consolidated_profit(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Consolidated accounting profit"
    
    def formula(tax_group, period, parameters):
        revenue = tax_group('tax_group_consolidated_revenue', period)
        expenses = tax_group('tax_group_total_expenses', period)
        
        return revenue - expenses


class tax_group_total_losses(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Total tax losses from all members"
    
    def formula(tax_group, period, parameters):
        parent_losses = tax_group.parent('tax_losses_brought_forward', period)
        sub_losses = tax_group.sum(
            tax_group.members('tax_losses_brought_forward', period),
            role=TaxGroup.SUBSIDIARY
        )
        
        return parent_losses + sub_losses


class tax_group_consolidated_taxable_income(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Consolidated taxable income"
    reference = "Federal Decree-Law No. 47 of 2022, Article 40"
    
    def formula(tax_group, period, parameters):
        profit = tax_group('tax_group_consolidated_profit', period)
        
        # Add back non-deductible expenses (simplified)
        # In reality, this would need to aggregate all adjustments
        
        # Apply loss offset at group level
        losses = tax_group('tax_group_total_losses', period)
        loss_limit = parameters(period).deductions.tax_loss_offset_limit
        
        taxable_before_losses = max_(profit, 0)
        allowable_loss_offset = min_(losses, taxable_before_losses * loss_limit)
        
        return max_(taxable_before_losses - allowable_loss_offset, 0)


class tax_group_corporate_tax(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Corporate tax for the tax group"
    reference = "Federal Decree-Law No. 47 of 2022, Article 40"
    
    def formula(tax_group, period, parameters):
        taxable_income = tax_group('tax_group_consolidated_taxable_income', period)
        threshold = parameters(period).corporate_tax.taxable_income_threshold
        rate = parameters(period).corporate_tax.standard_rate
        
        # Check if any member has small business relief
        # (Note: In practice, large groups unlikely to qualify)
        parent_small_biz = tax_group.parent('small_business_relief_eligible', period)
        any_small_biz = parent_small_biz  # Simplified - would check all members
        
        if any_small_biz:
            return 0
            
        # Standard calculation
        taxable_above_threshold = max_(taxable_income - threshold, 0)
        
        return taxable_above_threshold * rate


class tax_group_effective_rate(Variable):
    value_type = float
    entity = TaxGroup
    definition_period = YEAR
    label = "Effective tax rate for the group"
    
    def formula(tax_group, period, parameters):
        tax = tax_group('tax_group_corporate_tax', period)
        profit = tax_group('tax_group_consolidated_profit', period)
        
        return where(profit > 0, tax / profit, 0)


class tax_group_member_allocation(Variable):
    value_type = float
    entity = Business
    definition_period = YEAR
    label = "Allocated share of group tax liability"
    documentation = """
    For group members, their allocated share of the group's tax liability.
    Allocation methods can vary - this uses profit contribution.
    """
    
    def formula(business, period, parameters):
        is_member = business('is_tax_group_member', period)
        
        # If not a group member, no allocation
        if not is_member.any():
            return business.empty_array()
            
        # This is simplified - would need proper group linkage
        # In practice, would allocate based on profit contribution
        individual_profit = business('accounting_profit', period)
        
        # Placeholder for actual group tax allocation
        return where(is_member, individual_profit * 0.09, 0)
import pandas as pd
import pytest

from dgce_model.openfisca_runner import _compute_corporate_tax, _default_tax_params


def test_compute_corporate_tax_handles_policy_toggles():
    companies = pd.DataFrame(
        {
            "profit": [1_000_000.0, 5_000_000.0, 500_000.0],
            "annual_revenue": [5_000_000.0, 10_000_000.0, 2_000_000.0],
            "ISIC_level_1": [
                "Manufacturing",
                "Mining and quarrying",
                "Information and communication",
            ],
            "is_free_zone": [False, False, True],
        }
    )

    params = _default_tax_params()
    params.update(
        {
            "sme_exemption_rate": 0.0,  # include SMEs
            "free_zone_taxable_share": 0.0,  # free-zone firm fully exempt
            "random_seed": 0,
        }
    )

    taxed = _compute_corporate_tax(companies, params)

    # Manufacturing firm: taxable profit = 1,000,000 - 375,000 = 625,000 → tax = 56,250
    assert pytest.approx(taxed.loc[0, "corporate_tax"], rel=1e-6) == 56_250.0

    # Mining firm taxed at 55% on taxable profit 4,625,000
    assert pytest.approx(taxed.loc[1, "corporate_tax"], rel=1e-6) == 4_625_000.0 * 0.55

    # Free zone firm should pay zero tax
    assert taxed.loc[2, "corporate_tax"] == 0.0

    # Verify effective tax rates bounded between 0 and 1
    assert taxed["effective_tax_rate"].between(0.0, 1.0).all()


import numpy as np
import pandas as pd
import pytest

from dgce_model.ogcore_firm import prepare_and_evaluate


def make_sectoral_panel():
    return pd.DataFrame(
        {
            "economic_activity": [
                "Manufacturing",
                "Manufacturing",
                "Mining and quarrying",
                "Mining and quarrying",
            ],
            "year": [2022, 2023, 2022, 2023],
            "output_in_aed": [8_000_000.0, 8_400_000.0, 12_000_000.0, 12_600_000.0],
            "intermediate_consumption_in_aed": [4_000_000.0, 4_200_000.0, 6_000_000.0, 6_300_000.0],
            "value_added_in_aed": [4_000_000.0, 4_200_000.0, 6_000_000.0, 6_300_000.0],
            "compensation_of_employees_in_aed": [2_000_000.0, 2_100_000.0, 1_500_000.0, 1_575_000.0],
            "gross_fixed_capital_formation_in_aed": [600_000.0, 630_000.0, 900_000.0, 945_000.0],
            "number_of_employees": [80, 82, 60, 61],
        }
    )


def make_companies():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "ISIC_level_1": [
                "Manufacturing",
                "Manufacturing",
                "Mining and quarrying",
                "Mining and quarrying",
            ],
            "status": ["Active"] * 4,
            "annual_revenue": [2_500_000.0, 2_750_000.0, 5_000_000.0, 5_250_000.0],
            "employee_count": [40, 38, 55, 57],
            "is_free_zone": [False, False, False, False],
        }
    )


def test_prepare_and_evaluate_respects_target_margin():
    results = prepare_and_evaluate(
        companies=make_companies(),
        sectoral_panel=make_sectoral_panel(),
        target_profit_margin=0.12,
        target_year=2023,
    )

    assert {"profit", "gross_income"}.issubset(results.columns)

    gross = results["gross_income"].sum()
    profit = results["profit"].sum()
    margin = profit / gross

    assert pytest.approx(margin, rel=0.05) == 0.12

    # Sector-level margins should not exceed the imposed ceiling
    sector_margins = (
        results.groupby("ISIC_level_1")["profit"].sum()
        / results.groupby("ISIC_level_1")["gross_income"].sum()
    )
    assert (sector_margins <= 0.18 + 1e-6).all()

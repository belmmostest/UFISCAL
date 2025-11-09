import pandas as pd

from dgce_model.model.enhancements.policy_shock_engine import PolicyShockEngine


def test_policy_shock_engine_applies_growth_trend():
    baseline = {
        "gdp": 1_000.0,
        "consumption": 600.0,
        "investment": 200.0,
        "government": 150.0,
        "exports": 80.0,
        "imports": 60.0,
        "employment": 5.0,
    }
    engine = PolicyShockEngine(
        baseline,
        growth_trends={"gdp": 0.02, "consumption": 0.02, "investment": 0.03, "employment": 0.01},
    )

    df = engine.simulate(delta_tax=0.0, years=3)

    # Growth trend should produce non-flat projections
    assert isinstance(df, pd.DataFrame)
    assert df.loc[2, "gdp"] > df.loc[1, "gdp"] > df.loc[0, "gdp"]
    assert df.loc[2, "employment"] > df.loc[1, "employment"]

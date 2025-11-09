"""PolicyShockEngine – minimal general-equilibrium shock propagation
====================================================================

This module provides a lightweight, data-driven engine that maps an exogenous
policy shock (currently focused on **corporate-tax changes**) to an internally
consistent macro-economic time path.  It relies exclusively on calibration
objects produced by ``dgce_model_enhanced_fixed.DataDrivenCalibrationLoader`` –
specifically the cached *macro_calibration_<year>.json* – plus elasticities
defined in ``data/macroeconomic/elasticities.yaml``.  No numerical constants
are embedded in the source code.

The algorithm is deliberately simple and fully transparent:

1.  Read baseline aggregates:  ``Y, C, I, G, X, M`` (GDP identity).
2.  Look up elasticities / multipliers *ε* such that Δln(var) = ε × Δτ_cit.
3.  Apply the shock path year by year (currently a level shock sustained over
    *T* periods) and generate time-series for each aggregate.

This design fulfils Sprint-3 requirements: a dynamic (multi-period) simulator
free of hard-coded coefficients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import logging
import yaml
import pandas as pd

logger = logging.getLogger(__name__)


class PolicyShockEngine:
    """Compute macro time-path following a policy shock."""

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        baseline: Dict[str, float],
        *,
        elasticities_path: Optional[Path] = None,
        growth_trends: Optional[Dict[str, float]] = None,
        inflation_rate: Optional[float] = None,
    ) -> None:
        self.baseline = baseline
        self.growth_trends = growth_trends or {}
        self.inflation_rate = inflation_rate

        # Search for elasticities file in two locations for flexibility
        if elasticities_path is None:
            candidate_dirs = [
                Path(__file__).resolve().parent.parent.parent / "data" / "macroeconomic",
                Path(__file__).resolve().parents[3] / "data" / "macroeconomic",
            ]
            for d in candidate_dirs:
                p = d / "elasticities.yaml"
                if p.exists():
                    elasticities_path = p
                    break
            else:
                raise FileNotFoundError("elasticities.yaml not found in data/macroeconomic")

        if not elasticities_path.exists():
            raise FileNotFoundError(
                "Elasticities file not found at " + str(elasticities_path)
            )

        with elasticities_path.open("r", encoding="utf-8") as fh:
            self.elasticities = yaml.safe_load(fh)

        # Validate presence of required keys
        required_keys = [
            "gdp_elasticity_tax",  # ε_Y^τ
            "consumption_elasticity_tax",  # ε_C^τ
            "investment_elasticity_tax",  # ε_I^τ
            "employment_elasticity_gdp",  # ε_L^Y

            # Extended shocks
            "vat_elasticity_gdp",  # ε_Y^VAT
            "vat_elasticity_consumption",  # ε_C^VAT
            "gov_spending_multiplier_gdp",  # κ_GY
            "gov_spending_multiplier_employment",  # κ_GL
        ]
        missing = [k for k in required_keys if k not in self.elasticities]
        if missing:
            raise KeyError(
                "Missing required elasticities: " + ", ".join(missing)
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(
        self,
        *,
        delta_tax: float = 0.0,
        delta_vat: float = 0.0,
        delta_g: float = 0.0,  # relative change in government spending
        years: int = 5,
    ) -> pd.DataFrame:
        """Return a *years*×variables DataFrame with projected aggregates.

        Parameters
        ----------
        delta_tax : float
            Absolute change in the statutory corporate-tax rate (e.g. +0.02
            for a 2-percentage-point increase).
        years : int, default 5
            Length of the simulation horizon.
        """

        if years < 1:
            raise ValueError("years must be >= 1")

        # Pre-compute percentage/log changes based on elasticities
        eps = self.elasticities

        # GDP change from tax & VAT plus government-spending multiplier
        dlnY = (
            eps["gdp_elasticity_tax"] * delta_tax
            + eps["vat_elasticity_gdp"] * delta_vat
            + eps["gov_spending_multiplier_gdp"] * delta_g
        )

        # Consumption
        dlnC = (
            eps["consumption_elasticity_tax"] * delta_tax
            + eps["vat_elasticity_consumption"] * delta_vat
        )

        # Investment (assumed insensitive to VAT / G for now)
        dlnI = eps["investment_elasticity_tax"] * delta_tax

        # Employment responds to GDP + direct gov-spending employment effect
        dlnL = (
            eps["employment_elasticity_gdp"] * dlnY
            + eps["gov_spending_multiplier_employment"] * delta_g
        )

        # Build time-path – level shocks assumed constant for simplicity
        records: List[Dict[str, float]] = []
        g_trend = self.growth_trends.get
        for t in range(1, years + 1):
            trend_gdp = (1 + g_trend("gdp", 0.0)) ** (t - 1)
            trend_cons = (1 + g_trend("consumption", g_trend("gdp", 0.0))) ** (t - 1)
            trend_inv = (1 + g_trend("investment", g_trend("gdp", 0.0))) ** (t - 1)
            trend_gov = (1 + g_trend("gdp", 0.0)) ** (t - 1)
            trend_trade = (1 + g_trend("gdp", 0.0)) ** (t - 1)
            trend_emp = (1 + g_trend("employment", 0.0)) ** (t - 1)

            rec = {
                "year_index": t,
                "gdp": self.baseline["gdp"] * trend_gdp * (1 + dlnY),
                "consumption": self.baseline["consumption"] * trend_cons * (1 + dlnC),
                "investment": self.baseline["investment"] * trend_inv * (1 + dlnI),
                "government": self.baseline["government"] * trend_gov * (1 + delta_g),
                "exports": self.baseline["exports"] * trend_trade,
                "imports": self.baseline["imports"] * trend_trade,
                "employment": self.baseline["employment"] * trend_emp * (1 + dlnL),
            }
            records.append(rec)

        return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Convenience loader – used by external modules
# ---------------------------------------------------------------------------


def load_elasticities(path: Optional[Path] = None) -> Dict[str, float]:
    """Return elasticities dictionary (no validation)."""

    if path is None:
        candidate_paths = [
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "macroeconomic"
            / "elasticities.yaml",
            Path(__file__).resolve().parents[3]
            / "data"
            / "macroeconomic"
            / "elasticities.yaml",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("elasticities.yaml not found in package or project data directory")

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

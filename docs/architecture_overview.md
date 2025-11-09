# UFISCAL Modeling Stack – Architecture Overview

## Purpose
- Dynamic General Corporate Equilibrium (UFISCAL) stack built for UAE tax policy analysis.
- Integrates micro-level company data, sectoral national accounts, and macro elasticities to simulate corporate, VAT, and oil-revenue impacts.
- Primary entry points: `src/dgce_model/api/quick_simulation_enhanced.py`, `src/dgce_model/api/comprehensive_analysis_enhanced.py`, dashboard via `scripts/run_dashboard.py`.

## Data Sources & Calibration
- Raw inputs under `data/`:
  - `commerce_registry/full_registry_business.csv` – active firm register; ~600k rows.
  - `macroeconomic/` – sectoral panel, elasticities (`elasticities.yaml`), parameter JSON (`advanced_parameters.json`).
  - `distributions/` – firm size/activity priors used when registry fields are missing.
- `src/dgce_model/data_loader/new_data_loader.py`
  - Uses `_safe_read_csv` to avoid pandas parser crashes; cleans column names.
  - `RealUAEDataLoader` loads registry (optional), sectoral panel, IO table, parameter files; simulates missing revenue and employment counts (`_simulate_revenue_from_sector_output`).
  - `compute_corporate_tax_base` samples firms, runs OG-Core profit pipeline, applies SME relief and clipping to estimate aggregate taxable profit. Caches calibration snapshots to `src/dgce_model/parameters/macro_calibration_<year>.json`.

## Firm & Tax Micro Simulation
- `src/dgce_model/ogcore_firm.py`
  - `build_sector_parameters` derives capital/labor stats from sector panel (latest year).
  - `allocate_capital_to_firms` maps sector aggregates to firms using revenue/employment weights, enforces realistic capital-revenue ratios.
  - `evaluate_firm_profits_vectorized` applies CES production, calibrates to observed revenue, clips margins; outputs revenue, costs, profit.
  - `prepare_and_evaluate` orchestrates the above; default parameters in `DEFAULT_PARAMS`.
- `src/dgce_model/openfisca_runner.py`
  - `_default_tax_params` sets realistic UAE tax knobs (CIT 9%, allowances, free zone rate etc.).
  - `run_corporate_tax_simulation` validates firm & sector inputs, invokes OG-Core pipeline, then calculates taxable profit and corporate tax by firm, returning `corporate_tax`, `effective_tax_rate`, etc.
  - `validate_simulation_results` performs aggregate sanity checks on simulation output.

## Macro Model Layer
- `src/dgce_model/model/dgce_model_enhanced_fixed.py`
  - `DataDrivenCalibrationLoader` builds `UAEEconomicIndicators` using `RealUAEDataLoader`: GDP, consumption, investment, government spending, net exports, corporate tax base, sector shares. Loads cached macro calibration when available.
  - `OilDynamicsBlock` derives oil revenue path from sector shares or provided production.
  - `DataDrivenDGCEModel`
    - Renders steady-state dictionary from indicators (`gdp`, `employment`, `consumption`, `corporate_tax_base`, `oil_revenue`, sector shares).
    - `_initialize_sector_sensitivities` computes tax elasticities per sector using capital intensity & profit volatility.
    - `solve_policy_impact` combines baseline vs policy scenarios; `_apply_tax_policy_with_data` uses microsimulation + VAT/oil adjustments; `_calculate_*_impact` functions convert tax changes to macro responses.
    - `simulate` delegates to `PolicyShockEngine` for multi-year macro paths (CIT/VAT/spending shocks).
    - `apply_scenario` provides legacy API surface returning GDP/employment impacts, revenue analysis, sector results.
  - `SimplifiedDGCEModel` is an alias to `DataDrivenDGCEModel` for backward compatibility.
- `src/dgce_model/model/enhancements/policy_shock_engine.py`
  - Loads elasticities from `data/macroeconomic/elasticities.yaml` and produces time paths for GDP, consumption, investment, employment given `delta_tax`, `delta_vat`, `delta_g`.
  - Additional enhancement blocks (labor, SWF, price dynamics) reside in `src/dgce_model/model/enhancements/` and power `src/dgce_model/model/dgce_with_dynamics.py` when richer dynamics are needed.

## API Layer & User-Facing Modules
- `src/dgce_model/api/quick_simulation_enhanced.py`
  - Instantiates `SimplifiedDGCEModel` and `RealUAEDataLoader`.
  - Runs `apply_scenario` for static impacts, `simulate` for dynamics, `run_corporate_tax_simulation` for microsim; assembles quick-result payload with macro impacts, tax metrics, sector analysis.
- `src/dgce_model/api/comprehensive_analysis_enhanced.py`
  - Similar stack with additional steps: compliance validation, sectoral decomposition, risk assessment, dynamic path output, uses microsim and UFISCAL model for full report.
- Additional helpers:
  - `src/dgce_model/api/sector_analyzer_enhanced.py` – sector-focused wrapper.
  - `src/dgce_model/api/api_app_final.py` – validates API inputs (used by dashboard/service layer).
  - `src/dgce_model/api/policy_simulator.py`, `strategic_policy_scenarios.py`, `subpolicy_analyzer.py` – specialty analyses reusing the same core components.

## Dashboards & Runners
- `scripts/run_dashboard.py` launches a Flask/Bootstrap/Chart.js UI defined under `dashboard/` for interactive simulations.
- `config_and_startup.py` configures application defaults.

## Calibration Artifacts & Logs
- Macro snapshots stored under `src/dgce_model/parameters/macro_calibration_<year>.json`.
- Elasticities and auxiliary priors under `data/macroeconomic/`.
- Operational logs written to `dgce_system.log` when running full stack.

## Current Status & Follow-Up Items
- Microsimulation now calibrates sector-level profit margins to macro data and applies policy toggles (SME relief, free-zone selection, sector exclusions) consistent with calibration routines.
- Macro multipliers are sourced from `data/macroeconomic/elasticities.yaml` and combined with observed revenue deltas, eliminating hard-coded coefficients (`dgce_model_enhanced_fixed` now derives GDP/employment/investment impacts from data).
- PolicyShockEngine consumes the same elasticities and additionally receives trend growth and inflation extracted from the sectoral panel, so multi-year projections follow a realistic trajectory instead of remaining flat.
- `dgce_model.data_loader.new_data_loader.RealUAEDataLoader` caches instances keyed by data path/seed, preventing redundant reads of the 600k-firm registry during composite analyses (e.g., `scripts/final_run.py`, MCP endpoint).
- Debug noise from OG-Core vectorised pipeline removed to keep CLI output clean during large simulations.
- New pytest coverage exercises OG-Core profit calibration, corporate-tax policy toggles, the policy shock engine growth logic, the MCP orchestration layer, and the comprehensive analysis API with dependency stubs.
- Remaining enhancements: consider persisting regenerated macro calibration snapshots (`macro_calibration_<year>.json`) after future data refreshes and expose deterministic sampling controls through the public API surface if scenario reproducibility becomes critical.

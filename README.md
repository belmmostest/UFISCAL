# UFISCAL – UAE Fiscal Policy Simulation & Corporate Analysis Lab

UFISCAL is a production-grade modelling stack that fuses the OG-Core firm block
with an OpenFisca-style microsimulation and a calibrated Policy Shock Engine to
evaluate corporate-tax, VAT, and spending reforms in the UAE. All calibration
inputs are data-driven, and the repository ships with public mock data so the
full stack can be exercised without disclosing confidential datasets.

## Why UFISCAL?
- **Micro-founded economics.** Sector productivity, capital intensity, and
  profit margins are derived from national accounts and fed into the OG-Core
  production block (`src/dgce_model/ogcore_firm.py`).
- **Household-level microsimulation.** OpenFisca-inspired logic
  (`src/dgce_model/openfisca_runner.py`) applies statutory schedules, SME relief,
  free-zone rules, and compliance toggles to real firm registries.
- **Dynamic policy propagation.** `PolicyShockEngine` blends literature-backed
  elasticities with observed trend growth to trace GDP, consumption, investment,
  and employment paths after shocks.
- **Auditable results.** Validation expectations are captured in
  `docs/validation_log.md`, tests cover profit calibration, microsimulation
  toggles, and the policy package orchestration, and the dashboard / MCP server
  expose the same core model.

## Economic Theory in Brief
- **OG-Core foundation.** Firms maximise profits under CES production. Capital
  shares, depreciation, and productivity growth come from the sectoral panel in
  `data/macroeconomic/sectoral_panel.csv`. Margin caps keep each sector aligned
  with historical averages. Read more about [OG-Core](https://pslmodels.github.io/OG-Core/content/intro/intro.html)
- **OpenFisca microsimulation.** The registry sample flows through the
  OG-Core block, then `_compute_corporate_tax` applies statutory brackets,
  free-zone qualifiers, SME elections, and sector-specific carve-outs. Read more about [Openfisca](https://openfisca.org/en/)
- **Policy Shock Engine.** Elasticities in
  `data/macroeconomic/elasticities.yaml` translate statutory changes into
  macro deltas while respecting baseline growth trends inferred from the data.
- **Oil & VAT integration.** VAT shocks and oil revenue adjustments are treated
  symmetrically within the macro loader, so fiscal balances reflect both direct
  and indirect effects.

## Data & Privacy
- The repository contains **only mock data** under `data/`. Each file mirrors
  the schema of the private datasets defined in `docs/data_requirements.md`.
- Replace those files with jurisdiction-specific inputs (keeping the same column
  names) to obtain realistic outputs, or point `RealUAEDataLoader` at an
  alternate directory via its `data_path` argument.
- No calibration parameters are hard-coded; snapshots get written to
  `src/dgce_model/parameters/macro_calibration_<year>.json` when regenerated.

## Installation
UFISCAL targets **Python 3.10+** on macOS, Linux, or WSL.

```bash
git clone https://github.com/your-org/ufiscal.git
cd ufiscal
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # runtime deps
pip install -e .[dev]                    # editable install + pytest
pytest                                   # run the regression suite
```

Need the FastMCP server? Install the optional extra:

```bash
pip install -e .[mcp]
```

## Everyday Commands
| Goal | Command |
| --- | --- |
| Full batch run (quick + comprehensive + strategic) | `python scripts/final_run.py` |
| Launch dashboard UI | `python scripts/run_dashboard.py --port 8080` |
| Start FastMCP server | `python scripts/fastmcp_server.py` |
| Quick API smoke test | `python -m dgce_model.api.quick_simulation_enhanced` |
| Comprehensive API smoke test | `python -m dgce_model.api.comprehensive_analysis_enhanced` |

All commands operate on the mock data by default. Overwrite the CSV/JSON/YAML
files in `data/` or point `RealUAEDataLoader(data_path="path/to/your/data")`
to simulate with real inputs.

## Repository Layout
```
src/dgce_model/        # Core package (data loaders, OG-Core, OpenFisca, APIs)
scripts/               # CLI entry points (batch runner, dashboard, MCP server)
dashboard/             # Bootstrap + Chart.js front-end assets
docs/                  # Architecture notes, data contract, validation log
data/                  # 10-row synthetic calibration files (safe to publish)
tests/                 # Pytest suite exercising loaders, microsim, orchestration
```

## Validation & Testing
- `pytest` covers OG-Core profit calibration, corporate tax toggles, the policy
  shock engine, and the orchestration layer (see `tests/`).
- `docs/validation_log.md` records macro sanity checks (tax base ≈ AED 268 bn,
  GDP path monotonicity, compliance-adjusted rates) used to greenlight releases.
- The dashboard reproduces the same JSON payload that the CLI emits, ensuring
  UI parity with scripted runs.

## Documentation Set
- `docs/architecture_overview.md` – deep dive into each module and data flow.
- `docs/data_requirements.md` – schema-level contract for every input file.
- `docs/validation_log.md` – recorded benchmarks and sanity checks.

## Contributing
Please read `CONTRIBUTING.md` for workflow details (branching, testing, data
handling) and `CODE_OF_CONDUCT.md` for expected behaviour. Issues and pull
requests are welcome—high-impact areas include enriching the Policy Shock
Engine with new elasticities and extending the dashboard with additional
scenario levers.

## Citation
If you use Decreon or reference its work, please cite:
**Decreon (2025).** Available at [https://decreon.ai](https://decreon.ai)

## License
UFISCAL is released under the CC BY-NC License.
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

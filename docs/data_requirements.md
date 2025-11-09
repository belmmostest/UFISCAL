# Data Requirements & Mock Datasets

The repository ships with a fully mocked `data/` directory. Every file mirrors
the structure expected by the private `data_secret/` folder so that the codebase
can be executed end-to-end without exposing confidential inputs. Replace the
sample content with jurisdiction-specific data before running production
analysis.

## Directory Layout

```
data/
├── Penn_table_UAE_2019.csv
├── commerce_registry/
│   └── full_registry_business.csv
├── corporate_tax_levers.json
├── distributions/
│   ├── company_size_workforce.csv
│   └── economic_activity_distribution.csv
└── macroeconomic/
    ├── advanced_parameters.json
    ├── economic_activity_distribution.csv
    ├── elasticities.yaml
    ├── input_output_matrix.csv
    └── sectoral_panel.csv
```

> **Tip:** Copy `data/` to `data_secret/` and then overwrite each file with the
> real dataset. The loader defaults to `data/`, but you can pass a different
> path to `RealUAEDataLoader(data_path=\"data_secret\")`.

## File-by-File Expectations

| File | Purpose | Required Columns / Schema | Validation Notes |
| --- | --- | --- | --- |
| `Penn_table_UAE_2019.csv` | Penn World Table style macro aggregates used to benchmark capital, TFP, and expenditure shares. | `Country`, `Year`, `Variable name`, `Value`. | Header must be UTF-8 without BOM. The loader pivots by `Variable name`. |
| `commerce_registry/full_registry_business.csv` | Legal entity registry containing firm IDs, size buckets, and compliance attributes. | `id`, `company_size`, `ISIC_level_1`, `status`, `annual_revenue`, `employee_count`, `is_free_zone`. | Missing values are imputed, but column names are mandatory. |
| `distributions/company_size_workforce.csv` | Crosswalk between ISIC section, company size, and workforce shares per year. | `ISIC_level_1`, `Year`, `Company_size`, `Percent_of_company_size_total`. | Percent values must sum to 1 within an ISIC-year block. |
| `distributions/economic_activity_distribution.csv` & `macroeconomic/economic_activity_distribution.csv` | Sector GDP weights. Two copies exist for backwards compatibility with earlier modules. | `ISIC_level_1`, `Value`. | Values should sum to 1 (±1e-3). |
| `macroeconomic/sectoral_panel.csv` | National accounts by ISIC section. | `economic_activity`, `year`, `output_in_aed`, `intermediate_consumption_in_aed`, `value_added_in_aed`, `compensation_of_employees_in_aed`, `gross_fixed_capital_formation_in_aed`, `number_of_employees`. | Used to calibrate OG-Core productivity and employment elasticities. |
| `macroeconomic/input_output_matrix.csv` | Supply–use table expressed as shares of sectoral output. | `Input/Output` plus the 18 ISIC columns shown in the header. | Each row must have 18 numeric entries between 0 and 1 that approximately sum to 1. |
| `macroeconomic/advanced_parameters.json` | Higher-level macro settings (growth rates, TFP, capital shares). | JSON object with the keys shipped in `data/macroeconomic/advanced_parameters.json`. | All values are floats between 0 and 1 except `total_factor_productivity`. |
| `macroeconomic/elasticities.yaml` | Literature-based elasticities backing the Policy Shock Engine. | Keys: `gdp_elasticity_tax`, `consumption_elasticity_tax`, `investment_elasticity_tax`, `employment_elasticity_gdp`, `vat_elasticity_gdp`, `vat_elasticity_consumption`, `gov_spending_multiplier_gdp`, `gov_spending_multiplier_employment`. | Update with jurisdiction-specific estimates before publishing results. |
| `corporate_tax_levers.json` | Metadata for statutory schedule, exemptions, and carve-outs. | JSON with `statutory_rate_schedule`, `tax_base`, `thresholds_exemptions`, `tax_credits_rebates`, `payment_rules`, and `sectoral_carve_outs`. | Alter to match the legal framework you are modelling. |

## Using the Mock Data

Each CSV in `data/` contains 10 synthetic rows that illustrate column ordering
and expected data types. The samples are intentionally lightweight so unit tests
and documentation builds can run inside CI without sensitive inputs.

1. Duplicate the directory: `cp -R data data_secret`.
2. Replace every `.csv/.json/.yaml` file in `data_secret/` with the real data.
3. (Optional) Pass `data_path="data_secret"` when instantiating
   `RealUAEDataLoader`.

The loader emits descriptive errors if a column or file is missing. Refer to
`docs/validation_log.md` for the sanity checks executed after each data refresh.

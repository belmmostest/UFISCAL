# Contributing Guidelines

Thank you for your interest in improving the DGCE Model. We welcome
collaboration from economists, civic technologists, and OSS contributors. This
document covers the ground rules for proposing changes.

## Ground Rules

1. **Protect sensitive data.** Never commit confidential datasets. Keep real
   inputs inside `data_secret/`, which is ignored by git. Public pull requests
   should rely on the synthetic `data/` samples.
2. **Stay data-driven.** All parameters (elasticities, capital shares, etc.)
   must originate from documented data or literature. If you add a constant,
   document its source in code comments or `docs/`.
3. **Keep components modular.** OG-Core logic lives in `src/dgce_model/ogcore_firm.py`,
   OpenFisca wrappers in `src/dgce_model/openfisca_runner.py`, and UI concerns in
   `dashboard/`. Avoid cross-layer imports.
4. **Document behaviour.** Update `README.md`, `docs/`, or inline docstrings
   whenever you add user-facing capabilities.

## Development Workflow

1. **Fork & branch**
   ```bash
   git checkout -b feature/<short-description>
   ```
2. **Install dependencies**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure data**
   - Replace the mock files under `data/` with your secure datasets (same schemas).
   - Alternatively point `RealUAEDataLoader(data_path="/path/to/data")` at a private directory.
4. **Run tests**
   ```bash
   pytest
   ```
5. **Open a PR**
   - Describe the motivation and include before/after validation metrics.
   - Reference any issues you are closing (`Fixes #123`).

## Coding Standards

- Python code follows PEP 8 with type hints. Apply `ruff` or `black` if
  available, but do not auto-format unrelated files in the same commit.
- Prefer pure pandas/numpy vectorisation to Python loops for calibration logic.
- Keep comments high-signal; explain economic reasoning rather than restating
  code.

## Testing & Validation

- Add unit tests under `tests/` for any new model behaviour.
- Update `docs/validation_log.md` with macro benchmarks when you introduce new
  calibration data or elasticities.
- For dashboard or API changes, include sample payloads/responses in the PR.

## Community & Support

- Use Issues for bugs and feature requests.
- Use Discussions for research ideas or modelling questions.
- Respect the [Code of Conduct](CODE_OF_CONDUCT.md). Violations can be reported
  privately to the maintainer team via issues or direct email.

We are excited to collaborate on cutting-edge public-finance modelling—thank
you for contributing!

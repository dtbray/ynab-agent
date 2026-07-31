# ynab-agent

`ynab-agent` turns YNAB into a live data source for budgeting tools and
long-range financial planning. It combines an OAuth-first CLI, a durable SQL
cache, household reporting, and a retirement decision engine. The same typed
services are available through Typer commands and an authenticated FastAPI
application.

The project is designed for people who keep cash, debt, and investment
tracking accounts in YNAB and want reproducible analysis without treating every
tracking-account dollar as spendable.

## Highlights

- OAuth Authorization Code + PKCE with automatic token refresh, plus PAT
  support for local use
- Incremental, restart-safe synchronization into SQLite, PostgreSQL, or MySQL
- Budget, debt, spending, rollover, reconciliation, and account-freshness
  reports
- Seeded Monte Carlo and historical-block-bootstrap retirement simulations
- Explicit tax accounts, progressive federal and Indiana taxes, Roth
  conversions, withdrawal strategies, and capital-gain realization
- One- or two-person household cash flows, pension income, Social Security
  claiming, healthcare, Medicare, IRMAA, and long-term-care risk
- Multi-asset allocation, rebalancing, glide paths, named stress cases, and
  common-path scenario comparison
- Housing decisions modeled separately from the liquid portfolio
- Continuous calibration of saved plans against new YNAB observations
- A typed FastAPI surface with bounded durable planner jobs

## Requirements

- Python 3.13 or newer
- A YNAB personal access token or OAuth application

## Install

```bash
git clone https://github.com/dtbray/ynab-agent.git
cd ynab-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

For retirement simulation, historical models, charts, and PostgreSQL support:

```bash
python -m pip install -e ".[planner,historical,reports,postgres]"
```

## Authenticate

OAuth is the default:

```bash
ynab oauth url
ynab oauth exchange "<code>" --state "<state>"
```

Alternatively, set `YNAB_AUTH_MODE=pat` and provide `YNAB_ACCESS_TOKEN` in
your local `.env`. The example environment file contains the core application
settings; the deployment guides cover HTTP authentication and planner resource
controls.

## Start

```bash
ynab init
ynab plans list
ynab accounts list
ynab transactions list --type unapproved --limit 20
ynab sync
ynab reports cost-to-be-me --month "$(date +%Y-%m-01)"
ynab wealth accounts
```

Write operations require explicit execution and confirmation:

```bash
ynab transactions create \
  --account-id "<uuid>" \
  --amount=-4.25 \
  --payee-name "Coffee" \
  --execute \
  --yes
```

## Wealth planning

Start from the example scenario, select which YNAB accounts are liquid, and
run a deterministic seeded simulation:

```bash
cp examples/wealth-scenario.example.json baseline.local.json
ynab wealth simulate --scenario baseline.local.json
ynab wealth solve \
  --scenario baseline.local.json \
  --for annual_contribution \
  --lower 0 \
  --upper 50000 \
  --target-success 0.90
```

Only accounts classified as `retirement`, `taxable`, or `cash` enter the
spendable portfolio. Real estate, debt, and other tracking accounts remain
separate unless a scenario defines an explicit liquidity event.

Additional workflows include:

```bash
# Derive a spending baseline from mapped YNAB categories
ynab wealth spending baseline --budget-id "<budget-id>" --months 12 --json

# Validate allocation assumptions and compare Social Security strategies
ynab wealth allocation validate --plan allocation.local.json --json
cp examples/wealth-household.example.json household.local.json
ynab wealth optimize-social-security --scenario household.local.json --json

# Evaluate taxes and implementable withdrawal/conversion strategies
ynab wealth tax --input annual-tax.local.json --json
ynab wealth tax-strategy --scenario strategy.local.json

# Keep housing outside the spendable portfolio until an explicit event
ynab wealth housing project \
  --scenario examples/housing-scenario.example.json \
  --json

# Save immutable revisions and compare them on the same market paths
ynab wealth scenarios save --scenario household.local.json --json
ynab wealth scenarios compare \
  --baseline "<revision-uuid>" \
  --alternative "<revision-uuid>" \
  --json
```

Money results are expressed in today's dollars unless the selected output says
otherwise. Success is only one outcome: reports also include funded-spending
ratios, shortfall severity and duration, tax and healthcare effects, guardrail
reductions, and after-tax estate values.

Saved plans can be calibrated as fresh YNAB data arrives:

```bash
ynab wealth calibration create \
  --scenario-revision "<revision-uuid>" \
  --budget-id "<budget-id>"
ynab wealth calibration status --profile "<profile-uuid>" --json
```

Calibration records whether source evidence is fresh, stale, frozen, or
unknown and explains material changes in success, shortfall, tax, and estate
outcomes. It does not silently convert incomplete observations into planning
assumptions.

See [the quick start](docs/QUICKSTART.md),
[database notes](docs/DATABASE_README.md), and the complete
[wealth-planner guide](docs/WEALTH_PLANNER.md) for scenario schemas, model
boundaries, and HTTP operations.

## Development

```bash
python -m pip install -e ".[dev,planner,historical,reports]"
python -m pytest
python -m ruff check src tests migrations
python -m mypy src/ynab_agent
python -m build
python -m twine check dist/*
```

## Security and scope

The project defaults to read-only OAuth access. Write commands require explicit
execution and confirmation. Keep account identifiers, scenario files, database
contents, OAuth tokens, and exported financial data out of version control.
See [SECURITY.md](SECURITY.md) for vulnerability reporting.

Planning output is informational software output, not financial, tax, or legal
advice.

## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).

YNAB is a trademark of YNAB. This independent project is not affiliated with
or endorsed by YNAB.

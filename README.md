# ynab-agent

`ynab-agent` is a Python toolkit for working with YNAB data through the
official API. It provides an OAuth-first CLI, a local SQL cache, read-only
reporting, transaction workflows, and optional retirement-planning models.

## Highlights

- OAuth Authorization Code + PKCE, with automatic token refresh
- Read-only plans, accounts, transactions, and budget reports
- Explicit confirmation gates for write operations
- Incremental synchronization into SQLite, PostgreSQL, or MySQL
- Debt, spending, rollover, reconciliation, and wealth-planning tools
- A typed FastAPI surface for selected planning operations

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
your local `.env`. The example environment file contains the full set of
supported settings.

## Start

```bash
ynab init
ynab plans list
ynab accounts list
ynab transactions list --type unapproved --limit 20
ynab sync
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

See [the quick start](docs/QUICKSTART.md), [database notes](docs/DATABASE_README.md),
and [wealth-planner guide](docs/WEALTH_PLANNER.md) for more.

## Development

```bash
python -m pip install -e ".[dev,planner,historical,reports]"
python -m pytest
python -m ruff check src tests migrations
python -m mypy \
  src/ynab_agent/planning \
  src/ynab_agent/services/wealth.py
python -m build
python -m twine check dist/*
```

## Security and scope

The project defaults to read-only OAuth access. Review command output before
enabling writes, use a dedicated local database, and never commit exported
financial data. See [SECURITY.md](SECURITY.md) for vulnerability reporting.

Planning output is informational software output, not financial, tax, or legal
advice.

## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).

YNAB is a trademark of YNAB. This independent project is not affiliated with
or endorsed by YNAB.

# Changelog

## Unreleased

### Changed
- License the project under the GNU Affero General Public License v3.0 or
  later.

## 0.6.0 - 2026-07-30

### Added
- Report funded-spending ratios, cumulative real shortfalls, conditional
  failure duration and streaks, and recovery probabilities without changing
  the existing all-withdrawals-funded success rate.
- Add bounded named spending tiers, typed goal-attainment outcomes, and
  tax-adjusted real legacy-target probabilities for account-aware scenarios.
- Derive auditable essential, important, and discretionary spending tiers
  from explicit YNAB category mappings, with stale-data diagnostics and
  configurable fallback behavior.
- Add deterministic retirement spending guardrails with floor, ceiling,
  raise, and cut rules, plus annual policy audits and funded-spending totals.
- Save immutable scenario revisions with resolved account valuations,
  historical-data identities, engine versions, and complete reproducibility
  manifests.
- Compare scenario revisions on common market paths and report funded
  spending, shortfalls, guardrail concessions, taxes, and before- and
  after-tax estate deltas.

### Changed
- Advance planner result, reproducibility-manifest, and simulation-engine
  versions for the new outcome semantics.
- Include full-report outcome state and spending-tier work in planner resource
  admission and per-run reservations.
- Share queue, process, memory, request-body, and aggregate compute limits
  between durable planner jobs and synchronous scenario comparisons.
- Verify persisted revision, historical-observation, manifest, and complete
  comparison-result hashes on reads and immediately before execution.

### Tests
- Add deterministic shortfall, recovery, spending-tier, legacy-tax,
  guardrail, common-path comparison, persistence-integrity, cancellation,
  resource-bound, CLI, API, migration, and manifest coverage.

## 0.5.1 - 2026-07-30

### Fixed
- Preserve boolean tax assumptions as booleans in persisted planner results.

## 0.5.0 - 2026-07-30

### Added
- Add explicit tax-deferred, Roth, taxable, HSA, and cash portfolio buckets.
- Add taxable cost basis, contribution destinations, ordered withdrawals,
  Social Security taxation, qualified HSA treatment, and configurable RMDs.
- Report simulated lifetime taxes in real dollars.

### Changed
- Advance planner request, result, and simulation-engine versions for
  account-aware tax semantics.
- Include bounded tax-state memory and compute work in planner admission.
- Scale tax buckets and taxable basis during starting-portfolio threshold
  solves.

### Tests
- Add deterministic coverage for account tax treatment, withdrawal order,
  capital gains, Social Security, HSA-compatible balances, RMD reinvestment,
  contribution routing, solver scaling, and legacy blended-tax behavior.

## 0.4.0 - 2026-07-29

### Added
- Add bounded contribution and expense cash-flow streams to wealth scenarios.
- Add YNAB-derived mortgage projections with principal-and-interest, escrow,
  payoff date, payoff age, and suggested retirement-planning cash flows.
- Add a read-only mortgage-projection API endpoint for cached debt accounts.

### Changed
- Advance the wealth simulation engine version so persisted results are
  recomputed with cash-flow stream semantics.
- Use YNAB's cached monthly income aggregates for cost reporting.
- Generate PostgreSQL plan filters without ambiguous nullable parameters.

### Tests
- Add mortgage amortization, cash-flow stream, API, repository, and cost-report
  regression coverage.

## 0.3.0 - 2026-07-29

### Added
- Add persisted asynchronous planner jobs with authenticated submission,
  status, result retrieval, and cancellation endpoints.
- Add bounded process execution, paged restart recovery, historical dataset
  identities, and reproducible persisted manifests.
- Add engine- and schema-versioned idempotency while allowing retries after
  failed and cancelled attempts.

### Changed
- Bound planner request bodies, nested collections, string lengths, finite
  numeric inputs, memory, temporary storage, and compute work.
- Recover only within worker admission capacity and refill persisted work in
  bounded pages instead of materializing the full backlog.

### Tests
- Add migration round-trip, retry, cancellation, restart recovery, saturation,
  request-limit, and engine-version invalidation coverage.

## 0.2.0 - 2026-07-29

### Added
- Add historical block-bootstrap retirement simulations with paired inflation paths.
- Add scenario threshold solving for spending, contributions, portfolio size, and retirement age.
- Add money-weighted return calculations from irregular YNAB tracking-account cash flows.
- Add self-contained interactive wealth projection reports.
- Add reviewed account-valuation snapshots and reproducible simulation manifests.
- Add read-only FastAPI endpoints for account freshness and scenario validation.
- Add typed, resumable sync, reconciliation, and budget-rollover services.
- Add moving, debt, cost-to-be-me, overview, spending, obligation, and
  budget-activity report services backed by focused repositories.
- Add atomic OCI wheel releases with migration-before-switch, readiness
  checks, bounded rollback retention, and a dedicated API service account.

### Changed
- Install declared runtime extras and dependencies during OCI deployment.
- Preserve the house as an illiquid asset outside the default retirement
  starting portfolio.
- Match current YNAB assigned-plus-underfunded target semantics and cache the
  additional API data required to reproduce the UI.
- Count complete calendar windows, including zero-activity months, in
  spending and income statistics.
- Scope moving inputs by YNAB plan and budget, include split transactions,
  and normalize scheduled debt frequencies.
- Decompose the Typer CLI into command adapters and move database behavior
  into focused implementations.
- Bound planner memory, temporary storage, path generation, and experiment
  concurrency while retaining Python 3.13 as the supported runtime.

### Tests
- Add CLI contract, repository, service, planner reproducibility, resource
  bound, HTTP API, migration, and OCI rollback regression coverage.

## 0.1.0 - 2026-07-28

### Added
- Add Alembic migration support with an initial schema revision for the cached YNAB database.
- Add scheduled delta-sync deployment artifacts, including a lock-safe wrapper script and systemd timer/service templates.
- Add cached report commands for credit-card float, overspending, upcoming obligations, burn rate, and latest sync changes.
- Add OAuth token status reporting plus JSON and quiet output modes for OAuth commands.
- Add sync change tracking through a `budget_changes` cache table populated during delta syncs.
- Add a YNAB-backed wealth planner with explicit liquid-asset roles and seeded Monte Carlo retirement simulations.
- Cache plan currency metadata, complete goal state, payee locations, transaction splits, and scheduled transactions.
- Add observed-spending and current-month funding diagnostics without treating long-horizon goal balances as monthly costs.

### Changed
- Improve YNAB auth configuration errors for OAuth, 1Password, PAT, and unknown auth modes.
- Document migration and scheduled-sync operating procedures.
- Resolve symbolic YNAB plan aliases to their real UUID before writing related cache rows.
- Refresh current-month category detail during scheduled synchronization.
- Move the supported runtime and container baseline to Python 3.13.
- Apply schema migrations before starting the OCI sync worker during deployment.

### Tests
- Add Alembic migration coverage, report command tests, OAuth status/expiry tests, and sync change tracking tests.
- Add wealth simulation, API fidelity, split persistence, target-math, and deployment regression coverage.

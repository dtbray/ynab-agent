# Changelog

## Unreleased

## 0.9.0 - 2026-07-31

### Added
- Continuously calibrate saved wealth plans against YNAB spending,
  contributions, debt, balances, asset allocation, valuation freshness, and
  frozen assumptions.
- Persist content-addressed calibration profiles, snapshots, source
  observations, drift events, durable runs, attribution reports, and
  append-only material-alert lifecycles.
- Explain changes in success, shortfall, lifetime tax, and after-tax estate
  outcomes with deterministic, bounded grouped-factor attribution.
- Expose authenticated calibration profile, capture, status, disable, and
  alert-acknowledgement operations alongside matching Typer commands and a
  restart-safe scheduled worker.

### Fixed
- Bind every sync checkpoint and calibrated cache write to a durable,
  per-budget batch lifecycle with locked ownership, abandoned-batch
  supersession, stale-writer rejection, and atomic completion proofs.
- Recover safely from a crash after the final checkpoint but before
  calibration capture, including later partial deltas, clean legacy bootstrap,
  unbatched notification writes, and histories with more than 1,000 completed
  source owners.
- Bound fixed-length and streamed calibration request bodies before JSON
  parsing.
- Make the private-to-public release publisher portable across older
  self-hosted runners, idempotent after partial publication, and independent of a
  preinstalled GitHub CLI.

### Tests
- Add deterministic replay, tamper, freshness, drift, attribution, resource,
  crash-recovery, overlapping-sync, stale-writer, bootstrap, alert-lifecycle,
  CLI/HTTP parity, migration, and request-body coverage.

## 0.8.0 - 2026-07-30

### Added
- Model household pre-Medicare/ACA, Medicare, out-of-pocket, and separately
  inflated healthcare costs plus seeded long-term-care incidence, duration,
  severity, insurance, and shared home-equity funding.
- Report lifetime and annual healthcare costs, LTC lifetime selection and
  in-plan incidence, funding sources, and conditional shortfall severity with
  replayable assumptions in CLI and HTTP planner results.
- Model homes as explicit illiquid assets with mortgage amortization,
  appreciation, carrying costs, selling costs, and bounded care spending.
- Add auditable keep, sell, downsize, replace, rent, and reverse-mortgage
  decisions plus CLI and authenticated HTTP housing projections.
- Compare housing alternatives on common market paths and report
  housing-inclusive before- and after-tax estates.
- Add allowlisted, fail-closed private-source automation that exports and
  verifies the public distribution before opening a protected GitHub pull
  request.
- Publish source and wheel artifacts as a GitHub release after a release-sync
  pull request passes review and is merged.

### Fixed
- Route realized home-equity proceeds to one exact linked account, preserving
  duplicate-owner/treatment isolation and taxable basis exactly once.
- Apply net housing proceeds before event-year allocation and account returns,
  while keeping gross proceeds, costs, liens, replacement cash, and shortfalls
  separately auditable.
- Stack annual and terminal home gains with tax strategies, household filing
  status, and every keyed portfolio account on one modeled tax return.
- Route allocation, fee, rebalance, and glide-path returns through stable
  account-linked tax buckets instead of applying one blended return to every
  tax treatment.
- Persist and replay named-stress selectors for saved common-path comparisons
  through the CLI, authenticated HTTP API, canonical hashes, and manifests.
- Fail closed when allocation coverage or live linked account valuations do
  not exactly match the selected liquid YNAB accounts.
- Prefer supplied exact IRMAA lookback history over simulated MAGI for the
  same tax year, including post-start observations, and persist provenance.

### Changed
- Charge IRMAA as a survivor- and enrollment-aware healthcare cash flow in
  simulation engine v11 instead of reporting it only as an audit outcome.
  Scenarios that already include IRMAA in `annual_spending` must remove it to
  avoid double counting.
- Apply IRMAA tier thresholds from the filing status on the two-year lookback
  return while using current-year alive, Medicare-enrolled people only as the
  surcharge multiplier.
- Advance planner result, reproducibility, job, comparison, housing-manifest,
  and simulation-engine versions for account-keyed housing replay.
- Fund home-equity care through a typed exact-account reserve port, with only
  unmet care falling back to normal portfolio spending.
- Treat person-level healthcare LTC as the sole cost source when using a
  reserve-only housing care plan; reject conflicting deterministic care costs.
- Declare complete SPDX license, author, project URL, classifier, keyword, and
  typing metadata in built distributions.
- Build and validate both source distributions and wheels in public CI.
- Execute asset-location preferences with a deterministic capacity-constrained
  placement policy and expose per-account effective returns in annual audits.
- Persist account-by-account valuation observations so later YNAB changes
  cannot alter saved planner inputs.

## 0.7.0 - 2026-07-30

### Added
- Add versioned bracket-fill Roth-conversion and capital-gain-harvest rules,
  ordered or proportional withdrawals, and typed advisory asset-location
  preferences.
- Report deterministic annual tax actions, two-year Medicare IRMAA exposure,
  and real lifetime IRMAA surcharges alongside taxes and after-tax estate
  outcomes.
- Submit tax-strategy simulations through a dedicated authenticated API alias
  backed by the durable planner queue, and inspect schedules with
  `ynab wealth tax-strategy`.
- Model one- and two-person household work, pension, longevity, retirement,
  Social Security, survivor-spending, and filing-status timelines.
- Compare Social Security claiming ages 62 through 70 on common market and
  longevity paths using household taxes, funded spending, and portfolio
  outcomes rather than cumulative benefits alone.
- Preserve per-person ownership for duplicate tax-treatment accounts, including
  owner-specific basis, RMDs, withdrawals, and auditable public aggregation.

### Changed
- License the project under the GNU Affero General Public License v3.0 or
  later.
- Compare tax strategies on common market paths with IRMAA, funded-spending,
  shortfall, guardrail, lifetime-tax, and after-tax-estate tradeoffs.
- Jointly solve bracket-fill actions with current-year spending and
  tax-funding withdrawals so the final return honors the configured ceiling
  whenever forced income alone does not exceed it.
- Preserve non-worsening gain harvests when forced ordinary income already
  exceeds a conversion target, and include RMDs in scheduled tax-deferred
  withdrawal actions.
- Include tax-strategy searches and action state in planner compute, memory,
  request-body, and aggregate admission limits.
- Advance planner result, reproducibility-manifest, simulation-engine, and
  scenario-comparison schemas for replayable strategy actions.
- Apply spousal, survivor, RIB-LIM, earnings-test adjustment, and
  married-to-single tax transitions with versioned policy manifests.
- Persist resolved starting-portfolio valuation provenance and progressive-tax
  policy fingerprints in claiming-optimization replay manifests.
- Integrate tax-strategy projections with owner-keyed accounts while retaining
  owner-preserving Roth conversions, proportional withdrawal semantics, and
  conservative planner admission.

### Tests
- Add deterministic conversion-bound, gain-basis, proportional-fallback,
  no-future-path, replay, IRMAA-lookback, comparison, CLI/API, body-limit,
  manifest, and resource-accounting coverage.
- Add deterministic zero-PIA spouse, entitlement-onset, survivor RIB-LIM,
  pre-claim death, benefit-specific earnings-test, owner-isolation, dynamic
  filing, claiming-matrix, and household replay coverage.

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

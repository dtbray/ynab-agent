# Wealth planner

The wealth planner combines deliberately selected balances from the local YNAB
cache with explicit retirement assumptions. It does not infer holdings, taxes,
benefit amounts, or whether an off-budget account is investable.

Install the core planner dependencies:

```bash
pip install -e ".[planner]"
```

Historical block bootstrapping and interactive reports are separate so a
minimal installation does not need pandas, statsmodels, or Plotly:

```bash
pip install -e ".[planner,historical,reports]"
```

Start by reviewing active accounts:

```bash
ynab wealth accounts
```

Copy `examples/wealth-scenario.example.json` to a file ending in
`.local.json`. Those files are ignored by Git. Either set `starting_portfolio`
directly or remove it and classify YNAB accounts:

```json
{
  "accounts": [
    {"id": "ynab-account-id", "role": "retirement"},
    {"id": "another-account-id", "role": "real_estate"}
  ]
}
```

Allowed roles are `retirement`, `taxable`, `cash`, `real_estate`, `debt`,
`other`, and `excluded`. Only retirement, taxable, and cash balances are
included in the simulated portfolio. This is intentional: tracking accounts
can be stale or illiquid.

Run the same seeded scenario in a table or as machine-readable JSON:

```bash
ynab wealth simulate --scenario baseline.local.json
ynab wealth simulate --scenario baseline.local.json --json
ynab wealth simulate --scenario baseline.local.json --html baseline.local.html
```

The HTML report is self-contained. It includes the P10–P90 real-balance band,
median path, retirement boundary, simulated success rate, median funded
spending ratio, and median cumulative real shortfall.

## Shortfall severity and goal attainment

With fixed-real spending, `success_rate` remains backward compatible: a trial
succeeds only when every planned retirement withdrawal is funded through
`end_age`. With a guardrail policy, success instead means every
policy-adjusted withdrawal was funded; reduction and restoration metrics show
the concessions required to achieve that result. The richer outcome fields
explain how severe unsuccessful trials are:

- `funded_spending_ratio` is the P10/P50/P90 distribution of cumulative real
  spending funded divided by cumulative real spending required after applying
  the configured policy.
- `cumulative_shortfall_real` is the P10/P50/P90 distribution of unfunded
  spending summed in today's dollars.
- `failure_duration_years` and `longest_failure_streak_years` are conditional
  on trials with at least one shortfall. Successful trials do not dilute these
  percentiles with zeros.
- `recovery_probability` is conditional on affected trials. Recovery means at
  least one later retirement year returns to full funding after a shortfall;
  it does not erase the earlier failure or turn the trial into a success.

Named spending tiers provide goal primitives for later guardrail and
scenario-comparison features. A tier is attained only when its real annual
amount is funded in every retirement year. Tiers cannot exceed the simulated
base `annual_spending`:

```json
{
  "annual_spending": 70000,
  "spending_tiers": [
    {"name": "essential", "annual_amount": 45000},
    {"name": "comfortable", "annual_amount": 60000}
  ]
}
```

Every result includes a `planned_spending` goal outcome plus one outcome for
each configured tier. This makes it possible for a trial to miss the full plan
while still attaining an essential tier.

An optional `legacy_target_real` adds a legacy goal in today's dollars. When
explicit tax buckets and tax assumptions are available, the planner reports
the P10/P50/P90 tax-adjusted ending estate and the probability of meeting the
target:

```json
{
  "legacy_target_real": 250000
}
```

Tax adjustment applies the configured ordinary rate to tax-deferred and HSA
balances, the configured capital-gains rate to embedded taxable-account gains,
and no liquidation tax to Roth or cash balances. Without explicit tax inputs,
the legacy outcome remains present but is marked `explicit_tax_inputs_required`
with no probability. The planner does not model estate-tax exemptions,
beneficiary-specific HSA treatment, probate costs, or state inheritance taxes.

The same typed outcome fields are returned by `wealth simulate --json`, durable
planner-job API results, solver results, and persisted reproducibility
manifests. The manifest records the versioned definitions used by the engine.

## YNAB spending tiers and retirement guardrails

Map live on-budget categories by stable YNAB category ID. Renaming a category
changes only its display label; it does not lose the mapping. Essential
categories require an explicit monthly floor, including an intentional zero:

```bash
ynab wealth spending map \
  --budget-id your-budget-id \
  --category-id your-category-id \
  --tier essential \
  --essential-floor 650

ynab wealth spending list --budget-id your-budget-id
```

Supported tiers are `essential`, `lifestyle`, `discretionary`, and `one_time`.
The baseline query uses net outflows from on-budget accounts, handles split
transactions at the subtransaction level, ignores transfers, and annualizes a
bounded window whose end month is exclusive:

```bash
ynab wealth spending baseline \
  --budget-id your-budget-id \
  --through-month 2026-08 \
  --months 12 \
  --json
```

The returned `plan` can be copied into a scenario as
`retirement_spending_plan`. Its tier total must equal the scenario's
`annual_spending`; this keeps existing cash-flow streams compatible while the
nested policy can be switched independently. Policies are:

- `fixed_real`, which restores and holds the real baseline.
- `floor_ceiling`, which targets a withdrawal rate with maximum annual
  reduction and restoration percentages.
- `withdrawal_rate_guardrails`, which reduces spending above an upper
  withdrawal rate and restores it below a lower rate.

For example:

```json
{
  "kind": "withdrawal_rate_guardrails",
  "lower_withdrawal_rate": 0.035,
  "upper_withdrawal_rate": 0.05,
  "reduction_rate": 0.1,
  "restoration_rate": 0.1
}
```

Pass policy JSON to `wealth spending baseline --policy policy.local.json`, or
replace only `retirement_spending_plan.policy` in a scenario. Reductions
preserve the aggregate essential floor and remove optional tier spending
before essential spending. Both blended-tax and account-aware simulations use
the selected annual spending. Results include tier percentiles and counts of
every held, reduced, and restored trial in `annual_spending_real`, aggregate
event and cumulative real-dollar values in `guardrail_metrics`, and the full
versioned plan fingerprint in
`reproducibility.retirement_spending`.

The same operations are available over HTTP:

```text
PUT  /wealth/spending-tiers/{category_id}
GET  /wealth/spending-tiers?budget_id=...
GET  /wealth/spending-baseline?budget_id=...&through_month=2026-08-01
POST /wealth/spending-guardrails/preview
```

## Age-bounded cash flows and mortgage payoff

Use `cash_flow_streams` for recurring amounts that begin or end at a known age.
Contribution streams augment `annual_contribution` before retirement. Expense
streams augment `annual_spending` after retirement:

```json
{
  "cash_flow_streams": [
    {
      "name": "Redirected mortgage principal and interest",
      "flow_type": "contribution",
      "start_age": 55,
      "annual_amount": 10378.32,
      "inflation_adjusted": true
    }
  ]
}
```

The HTTP API can derive a payoff estimate and planner-ready stream fragments
from the cached YNAB mortgage balance, interest-rate history, minimum-payment
history, and escrow history:

```text
GET /wealth/accounts/{account_id}/mortgage-projection?current_age=30
```

YNAB does not return a contractual payoff date. The endpoint amortizes the
current balance using the latest terms at or before `as_of`. It separates escrow
from principal and interest because property tax and insurance normally remain
after payoff. Treat the response as an estimate and compare it with the lender's
amortization schedule.

## Account-aware taxes

The optional account-aware tax model replaces `withdrawal_tax_rate` with explicit aggregate
balances for `tax_deferred`, `roth`, `taxable`, `hsa`, and `cash` treatments.
At most one bucket per treatment is allowed, and bucket balances must equal the
explicit `starting_portfolio`. Taxable buckets require an estimated cost basis.
Contribution fractions describe where the base annual contribution lands:

```json
{
  "tax_buckets": [
    {
      "tax_treatment": "tax_deferred",
      "starting_balance": 102700.39,
      "contribution_fraction": 1
    },
    {
      "tax_treatment": "roth",
      "starting_balance": 40575.23
    },
    {
      "tax_treatment": "taxable",
      "starting_balance": 1724.01,
      "taxable_basis": 1724.01
    },
    {
      "tax_treatment": "hsa",
      "starting_balance": 784.51
    }
  ],
  "tax_assumptions": {
    "tax_model": "effective_rates",
    "ordinary_income_tax_rate": 0.12,
    "long_term_capital_gains_tax_rate": 0.15,
    "social_security_taxable_fraction": 0.85,
    "qualified_hsa_withdrawal_fraction": 1,
    "rmd_start_age": 75,
    "apply_required_minimum_distributions": true,
    "withdrawal_order": [
      "taxable",
      "tax_deferred",
      "roth",
      "hsa"
    ],
    "retirement_surplus_destination": "taxable"
  },
  "withdrawal_tax_rate": 0
}
```

Every tax-aware income stream must declare `ordinary`, `social_security`, or
`tax_free`. Contribution cash-flow streams may set
`destination_tax_treatment`; otherwise they use the bucket contribution
fractions.

The `effective_rates` engine applies the configured effective ordinary-income rate to
tax-deferred withdrawals and the configured capital-gains rate only to the
gain portion of taxable withdrawals. It tracks taxable basis proportionally,
supports a configurable qualified HSA fraction, taxes the configured portion
of Social Security, and can enforce current-law Uniform Lifetime Table RMDs.
After-tax RMD or income surplus is reinvested in the configured destination.
The result includes lifetime taxes in real dollars. When the scenario includes
a legacy target, the ending estate calculation uses the same effective-rate
assumptions as described above.

For a versioned current-law calculation, select `progressive_us_indiana` and
provide household facts:

```json
{
  "ordinary_income_tax_rate": 0,
  "long_term_capital_gains_tax_rate": 0,
  "tax_model": "progressive_us_indiana",
  "progressive": {
    "policy_id": "us_in_2026_v1",
    "filing_status": "married_filing_jointly",
    "simulation_start_year": 2026,
    "taxpayer_birth_year": 1964,
    "spouse_birth_year": 1965,
    "future_policy_mode": "inflation_indexed",
    "bracket_inflation_rate": 0.025,
    "indiana_resident": true,
    "aca_household_size": 2,
    "irmaa_lookback_magi": [
      {"tax_year": 2024, "magi": 180000},
      {"tax_year": 2025, "magi": 175000}
    ]
  }
}
```

This engine calculates federal ordinary brackets and standard deductions,
long-term capital-gain stacking, taxable Social Security, the temporary senior
deduction, Indiana adjusted gross income tax and exemptions, and birth-year
RMD start ages. Withdrawals are solved against their incremental tax rather
than grossed up at a flat rate. The result manifest fingerprints the packaged
policy and records its official sources, filing status, effective year, and
future-law assumptions. Progressive simulation results include a bounded
annual audit of P10/P50/P90 total tax plus effective, marginal ordinary, and
marginal long-term-gain rates.

`future_policy_mode=inflation_indexed` projects the 2026 federal brackets,
standard deduction, and capital-gain thresholds at the configured rate.
`fixed_nominal` holds them at their 2026 dollar values. Social Security
thresholds remain nominal under both choices. The 2026 Indiana rate is 2.95%;
the known 2027 rate of 2.9% is used thereafter. These are projections, not a
claim that future law is known.

The same annual calculator is available without a simulation:

```bash
ynab wealth tax --input annual-tax.local.json --json
```

and as `POST /wealth/tax/calculate`. It reports federal and Indiana components,
effective and $1 marginal rates, ACA expected-contribution and premium-tax-
credit hooks, and the two-year Medicare IRMAA lookback tier and surcharge.
ACA credits and Medicare premiums are audit fields only until the separate
healthcare-premium cash-flow model consumes them.

This remains a retirement-planning engine, not tax preparation software. It
does not model itemized deductions, NIIT, AMT, tax-loss harvesting, Roth
conversions, qualified charitable distributions, local Indiana income tax, or
the joint-life RMD exception. Birth-year age tests use calendar-year age; the
pre-1949 70½ rule is represented as age 70. The effective-rate engine remains
useful for sensitivity analysis and for households outside the progressive
policy's supported jurisdiction.

The account distinctions follow the broad rules documented by the
[IRS for traditional and Roth IRAs](https://www.irs.gov/retirement-plans/traditional-and-roth-iras),
[HSA distributions](https://www.irs.gov/publications/p969), and
[RMD tables](https://www.irs.gov/publications/p590b).

PolicyEngine-US and PSL's Tax-Calculator were evaluated. Both are respected
full microsimulation systems, but each brings a much broader policy/dependency
surface and release cadence than this bounded per-trial calculator needs.
Neither is added as a runtime dependency; the local policy remains small,
reviewable, deterministic, and replaceable if the planner later needs
full-return microsimulation.

## Solve a planning threshold

The solver reuses the scenario seed for every candidate, giving each candidate
the same return paths. It can find the maximum annual spending, minimum annual
contribution, minimum starting portfolio, or earliest retirement age that
reaches a target success rate:

```bash
ynab wealth solve \
  --scenario baseline.local.json \
  --for annual_spending \
  --lower 30000 \
  --upper 150000 \
  --target-success 0.90

ynab wealth solve \
  --scenario baseline.local.json \
  --for retirement_age \
  --lower 55 \
  --upper 70 \
  --target-success 0.90 \
  --json
```

Money-variable searches use $100 increments by default, avoiding a false
impression of cent-level precision from Monte Carlo output. Use
`--resolution` to choose another increment. Retirement age is evaluated in
whole years. Simulation output includes a 95% Wilson confidence interval for
the observed success rate.

## Save and compare scenario revisions

Save a scenario after its live YNAB account values have been resolved:

```bash
ynab wealth scenarios save \
  --scenario baseline.local.json \
  --json
```

The returned UUID identifies an immutable revision. Saving another file with
the same scenario `name` appends the next revision; it never edits the prior
one. Each revision manifest records the complete resolved scenario, starting
portfolio and valuation provenance, engine identity, and—when historical
bootstrapping is used—the exact observations and their fingerprints.

Compare one baseline with one or more alternatives:

```bash
ynab wealth scenarios compare \
  --baseline 00000000-0000-4000-8000-000000000001 \
  --alternative 00000000-0000-4000-8000-000000000002 \
  --json
```

Alternatives must have identical horizon, trial count, seed, return model,
return assumptions, inflation, fees, bootstrap settings, and historical
observations. The engine prepares those return and inflation paths once and
uses the same arrays for every revision, so the delta reflects scenario inputs
rather than unrelated Monte Carlo noise. Generated arrays are closed after the
comparison and are never persisted.

The stable comparison model reports success probability, requested and
cumulatively funded spending, cumulative shortfall, lifetime taxes when the
tax-aware model supplies them, median estate value, and P10 estate and
retirement balances. Explicit tax-account scenarios also expose after-tax
estate percentiles; guardrail scenarios expose cumulative spending reductions
so success achieved through lifestyle concessions is not mistaken for an
unqualified improvement. Each alternative also contains
alternative-minus-baseline deltas, changed-input classifications, material
benefits and costs, and any revision that dominates it across all available
objectives.

The HTTP equivalents are authenticated:

```text
POST /wealth/scenarios/revisions
GET  /wealth/scenarios/revisions/{revision_id}
POST /wealth/scenarios/comparisons
GET  /wealth/scenarios/comparisons/{comparison_id}
```

HTTP historical scenarios accept only a server-registered
`historical_dataset_id`; they never accept a server filesystem path. Both CLI
and HTTP serialize the same revision and comparison models. A stored comparison
manifest references every revision manifest hash, the exact common path
specification, execution policy, and engine identity needed to audit or repeat
the run. Reads verify those revision hashes and a separate hash over the full
persisted comparison result. HTTP comparisons share the planner worker's
bounded outstanding-work and aggregate memory admission; sequential
alternatives reuse one path matrix and reserve only the largest evaluation
state, while their compute estimates are summed.

Dominance is a basic multi-objective screen, not a recommendation. It prefers
higher success, funded spending, requested spending, and estate/downside
balances, while preferring lower shortfall, guardrail reductions, and lifetime
tax. A plan can remain non-dominated because it offers a genuine trade-off.
Tax strategy, household optimization, and richer spending-goal semantics plug
into the service's outcome-extractor and change-classifier policy interfaces.

## Historical block bootstrap

Set the scenario's return model and block length:

```json
{
  "return_model": "historical_bootstrap",
  "historical_block_size": 5
}
```

Supply annual observations as decimal rates. Inflation is optional; when
present it is sampled in the same historical blocks as investment returns so
their observed relationship is retained:

```csv
year,nominal_return,inflation_rate
2000,-0.0903,0.0338
2001,-0.1185,0.0283
2002,-0.2197,0.0159
```

```bash
ynab wealth simulate \
  --scenario historical.local.json \
  --returns annual-returns.local.csv
```

Years are sorted and must be contiguous, and the CSV must contain at least as
many annual observations as the scenario horizon. Results record the resolved
CSV path and SHA-256 fingerprint alongside the seed. Keep downloaded or
personally curated data in a `.local.csv` file; those files are ignored by Git.

## Money-weighted investment performance

PyXIRR can calculate a combined money-weighted return from one or more YNAB
tracking accounts:

```bash
ynab wealth performance \
  --account-id first-ynab-account-id \
  --account-id second-ynab-account-id \
  --as-of 2026-07-29
```

Only transfers between a selected investment account and an unselected account,
plus YNAB starting-balance transactions, are treated as external investor cash
flows. Transfers among selected accounts, dividends, fees, reconciliation
adjustments, and market-value updates are excluded. The output reports how many
transactions were excluded.

The current cached account balance is used only when `--as-of` is today. A
historical valuation requires an explicit reviewed `--ending-balance`, because
the YNAB API cache does not yet retain dated balance snapshots.

## Model boundary

The default model uses independent annual lognormal returns described by an
arithmetic mean and volatility. The historical model uses a stationary block
bootstrap, which retains multi-year return sequences and can retain paired
inflation observations. Contributions occur after each pre-retirement year's
return, including active contribution cash-flow streams. Retirement spending
includes active expense cash-flow streams, then subtracts active income
streams, and is withdrawn after each retirement year's return. Fees reduce gross
returns. A scenario may use either the legacy blended withdrawal tax rate or
the explicit account-aware tax model.

The example scenario uses a VTSAX-like total-stock-market profile: a
conservative 6% nominal arithmetic return and 16% annual volatility. These are
editable planning assumptions, not a claim about future VTSAX performance.

Balances at retirement and the plan horizon are reported in today's dollars.
Success means the portfolio never failed to meet a scheduled withdrawal
through `end_age`. The result is a scenario comparison tool, not a forecast or
financial advice.

Important limitations:

- One blended return process is applied to the selected liquid portfolio.
- Parametric returns are independent and do not model regime changes.
- Historical results are bounded by the quality and representativeness of the
  supplied observation set.
- Runs are limited to 100,000 trials until the engine supports bounded
  batch-wise percentile aggregation.
- Account-aware taxes use explicit effective rates rather than progressive
  federal and state tax-return calculations.
- YNAB has account balances, not security holdings or asset allocation.
- YNAB investment tracking does not distinguish market gains from
  reconciliation adjustments.
- Social Security and pension amounts should come from an authoritative
  personal estimate and be entered as scenario inputs.

Natural next steps are periodic balance snapshots, side-by-side scenario
comparison, asset-class allocations and covariance, longevity sampling,
progressive tax-policy plugins, Roth-conversion strategies, and versioned
economic assumption sets.

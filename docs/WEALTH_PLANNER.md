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
## Household and Social Security claiming

Use the optional `household` input for one or two people. The first person's
age on `plan_start_date` must equal the scenario `current_age`. Each person has
a birth date, retirement age in months, SSA primary insurance amount (PIA),
claim age in months, independent work and pension timelines, and deterministic
or bounded probabilistic longevity. See
`examples/wealth-household.example.json` for a complete tax-aware couple.
An omitted claim age means no retirement benefit is paid in a normal
simulation; the optimizer supplies each candidate claim age itself.

Compare whole-year claiming ages from 62 through 70:

```bash
ynab wealth optimize-social-security \
  --scenario household.local.json
ynab wealth optimize-social-security \
  --scenario household.local.json \
  --json
```

The matrix is bounded to 81 strategies for a couple and reuses identical
market paths and seeded longevity assumptions. Ranking is lexicographic:
household success rate, median funded spending, then median after-tax ending
portfolio. Explicit tax buckets and tax assumptions are therefore required.
It deliberately does not maximize raw cumulative benefits.
The optimizer currently accepts the parametric `lognormal` return model; a
historical comparison needs a future registered/prepared-experiment entry
point.

The API exposes the same bounded operation at
`POST /wealth/social-security/optimize`. Regular simulation and durable planner
results include `household_cash_flow_audit`, with annual per-person real-dollar
P10/P50/P90 work, pension, own/spousal/survivor Social Security, earnings-test
withholding, fully withheld-month credits and FRA benefit recomputation, total
paid benefits, alive status, and filing-status probabilities. The
reproducibility manifest records the policy snapshot, official sources,
longevity method, resource estimate, and tax-engine port.

Modeled SSA rules include monthly early and delayed retirement adjustments,
FRA by birth year, spouse and survivor reductions, deemed filing for own and
spouse benefits, the 2026 earnings test, and the 2026 family maximum. Primary
sources:

- [SSA retirement claiming ages](https://www.ssa.gov/benefits/retirement/planner/applying2.html)
- [SSA full retirement age table](https://www.ssa.gov/oact/progdata/nra.html)
- [SSA early/delayed adjustment formulas](https://www.ssa.gov/policy/docs/statcomps/supplement/2025/apnc.html)
- [SSA spouse reduction formulas](https://www.ssa.gov/OP_Home/handbook/handbook.07/handbook-0724.html)
- [SSA spouse benefit rules](https://www.ssa.gov/blog/en/posts/2024-07-11.html)
- [SSA survivor benefit amounts](https://www.ssa.gov/survivor/amount)
- [SSA 2026 earnings-test limits](https://www.ssa.gov/cola/factsheets/2026.html)
- [SSA earnings-test crediting months](https://secure.ssa.gov/poms.nsf/lnx/0302501021)
- [SSA family-maximum formula](https://www.ssa.gov/oact/COLA/familymax.html)

This is a household portfolio comparison, not an SSA entitlement
determination. It does not model disability/dependent benefits,
government-pension offsets, partial calendar-year claiming, grace-year monthly
tests, or future policy changes. At FRA it recomputes retirement and spousal
reductions for months with full or partial work deductions. Probabilistic longevity uses a
seeded bounded correlated-normal sensitivity rather than actuarial mortality
tables. Assets remain in the household portfolio after a death. Years after the
last household death are inactive for recovery and spending-tier evaluation;
survivor tier targets scale with survivor spending. Progressive scenarios apply
married-filing-jointly rules while both spouses live and single-filer rules
after a survivor transition for income, withdrawals, and terminal liquidation. Account
ownership is retained for owner-specific distribution work in issue #94.
`retirement_age_months` is an audited milestone; explicit work and pension
timelines determine each person's cash-flow dates. Pension and Social Security
received before the scenario retirement milestone are audited but are not
automatically invested. Model those savings with an explicit contribution
cash-flow stream.

## Multi-asset allocation and correlated returns

Add `portfolio_allocation` to model US equity, international equity, bonds,
and cash separately. Market assumptions are independent from allocation
strategy, so scenarios with different account targets can reuse the same
seeded common market paths:

```json
{
  "portfolio_allocation": {
    "market": {
      "us_equity": {"expected_return": 0.08, "volatility": 0.18},
      "international_equity": {
        "expected_return": 0.07,
        "volatility": 0.20
      },
      "bonds": {"expected_return": 0.04, "volatility": 0.07},
      "cash": {"expected_return": 0.025, "volatility": 0.01},
      "correlation": {
        "values": [
          [1, 0.75, -0.10, 0],
          [0.75, 1, -0.10, 0],
          [-0.10, -0.10, 1, 0.20],
          [0, 0, 0.20, 1]
        ]
      }
    },
    "accounts": [
      {
        "account_id": "ynab-401k-account-id",
        "portfolio_weight": 0.8,
        "annual_fee_rate": 0.003,
        "target": {
          "us_equity": 0.7,
          "international_equity": 0.2,
          "bonds": 0.1,
          "cash": 0
        },
        "glide_path": [
          {
            "age": 60,
            "weights": {
              "us_equity": 0.5,
              "international_equity": 0.15,
              "bonds": 0.3,
              "cash": 0.05
            }
          }
        ]
      },
      {
        "account_id": "ynab-brokerage-account-id",
        "portfolio_weight": 0.2,
        "target": {
          "us_equity": 0.5,
          "international_equity": 0.3,
          "bonds": 0.1,
          "cash": 0.1
        }
      }
    ],
    "rebalancing": {
      "frequency_years": 1,
      "drift_threshold": 0.05
    }
  }
}
```

The correlation matrix order is US equity, international equity, bonds, then
cash. It must be symmetric, have ones on its diagonal, contain only values
from -1 through 1, and be positive semidefinite. These are requested
correlations of annual simple returns. The engine transforms them into the
lognormal covariance required for sampling and rejects inputs when that
transformed matrix is not positive semidefinite. Account and asset weights
must each total one. Glide-path targets interpolate linearly by age.
`frequency_years: 1` checks rebalancing annually; set it to another bounded
interval or `null` to disable scheduled rebalancing. A drift threshold can
skip trades while weights remain close to target.

Validate the exact same canonical representation through either interface:

```bash
ynab wealth allocation validate --plan allocation.local.json --json
```

```text
POST /wealth/allocations/validate
```

Parametric multi-asset paths use correlated lognormal shocks. Account fees and
the legacy global fee are deducted before rebalancing. Every annual result
records real fee and turnover percentiles, rebalanced-trial counts, and
post-rebalancing account and asset weights in `annual_allocation_real`.
It also records each account's effective gross return after its allocation
and fees. When `accounts` selects live liquid YNAB accounts, the allocation
must cover that set exactly; partial coverage is rejected.
`reproducibility.portfolio_allocation` contains the full assumptions,
covariance matrix, strategy identity, and SHA-256 fingerprint.

For a multi-asset historical bootstrap, add all four asset-return columns to
the existing CSV. `nominal_return` remains required as the compatible
one-asset series:

```csv
year,nominal_return,inflation_rate,us_equity_return,international_equity_return,bonds_return,cash_return
2021,0.18,0.07,0.25,0.08,-0.02,0.001
2022,-0.16,0.065,-0.19,-0.16,-0.12,0.015
2023,0.21,0.034,0.26,0.18,0.05,0.045
```

The bootstrap samples one index sequence for all four assets and inflation,
preserving their observed same-year relationship and multi-year blocks. The
same format is accepted by `--returns` and by server-registered planner
datasets. A multi-asset historical run fails clearly if any asset column is
missing.

The bounded named-stress catalog is available through both adapters:

```bash
ynab wealth allocation stresses --json
ynab wealth simulate --scenario allocation.local.json --stress equity_crash
ynab wealth scenarios compare \
  --baseline BASELINE_UUID \
  --alternative ALTERNATIVE_UUID \
  --named-stress equity_crash
```

```text
GET /wealth/allocations/stresses
POST /planner/jobs
{"scenario": {...}, "named_stress": "equity_crash"}
```

Catalog definitions and their SHA-256 fingerprints are versioned in results.
Each finite sequence repeats for a longer scenario horizon, allowing clients
to submit the same scenario once per selector for deterministic comparisons.

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

The optional account-aware tax model replaces `withdrawal_tax_rate` with
explicit balances for `tax_deferred`, `roth`, `taxable`, `hsa`, and `cash`
treatments. Aggregate scenarios may omit account IDs. Multi-account allocation
scenarios link each bucket to exactly one selected liquid account with
`account_id`; this permits multiple accounts with the same owner and treatment
without conflating their returns, basis, tax drag, withdrawals, or strategy
projections. Linked bucket balances and allocation weights must reconcile to
the account observations. Taxable buckets require an estimated cost basis.
Contribution fractions describe where the base annual contribution lands:

```json
{
  "tax_buckets": [
    {
      "tax_treatment": "tax_deferred",
      "account_id": "ynab-401k-account-id",
      "starting_balance": 102700.39,
      "contribution_fraction": 1
    },
    {
      "tax_treatment": "roth",
      "account_id": "ynab-roth-account-id",
      "starting_balance": 40575.23
    },
    {
      "tax_treatment": "taxable",
      "account_id": "ynab-brokerage-account-id",
      "starting_balance": 1724.01,
      "taxable_basis": 1724.01
    },
    {
      "tax_treatment": "hsa",
      "account_id": "ynab-hsa-account-id",
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

Live linked scenarios persist the sorted account/value observations in
valuation provenance. Submission fails closed if a bucket balance or
allocation share differs from the resolved YNAB account value, and later YNAB
changes cannot alter a saved revision's replay inputs.

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
The standalone tax calculation keeps those as audit fields. A simulation with
`healthcare` assumptions consumes explicit net ACA premiums, Medicare base
premiums, out-of-pocket costs, and IRMAA as healthcare cash flows.

### Tax-strategy comparisons

Add a versioned `strategy` to `tax_assumptions` to compare implementable
current-year rules:

```json
{
  "strategy": {
    "policy_id": "tax_strategy_v1",
    "withdrawal_policy": "proportional",
    "proportional_withdrawal_fractions": [
      {"tax_treatment": "taxable", "fraction": 0.4},
      {"tax_treatment": "tax_deferred", "fraction": 0.3},
      {"tax_treatment": "roth", "fraction": 0.3}
    ],
    "roth_conversion": {
      "policy_id": "roth_bracket_fill_v1",
      "start_age": 60,
      "end_age": 67,
      "target_federal_ordinary_bracket_rate": 0.12,
      "max_annual_conversion_real": 50000
    },
    "capital_gain_harvest": {
      "policy_id": "capital_gain_bracket_fill_v1",
      "start_age": 60,
      "end_age": 67,
      "target_federal_long_term_capital_gains_rate": 0,
      "max_annual_gain_real": 25000
    },
    "asset_location_preferences": [
      {
        "asset_class": "bonds",
        "preferred_tax_treatments": ["tax_deferred"]
      }
    ]
  }
}
```

The proportional fractions describe target shares of the annual net cash
need. If one treatment cannot fund its share, the configured
`withdrawal_order` deterministically supplies the remainder. Proportional
withdrawals work with either tax engine. Roth conversion and gain-harvest
rules require `progressive_us_indiana`, matching tax-deferred and Roth
buckets for conversions and a taxable bucket with basis for harvesting.

Each annual rule sees only the current tax year, current income, current
account balances and basis, the current-year return already observed, and the
versioned tax policy. It jointly projects the configured current-year cash
need and withdrawal policy so the final return, including withdrawals used to
fund an action's tax, remains inside the selected bracket when forced income
alone permits it. If forced income or an RMD already exceeds an active
conversion target, gain harvesting may continue only while it does not worsen
the baseline ordinary-income position. It never reads a future return path.
Conversion and harvest caps are today's dollars and scale with the
simulation's inflation path. Gain harvesting models a sell-and-repurchase
basis reset; any resulting tax is funded through the configured withdrawal
policy. The asset-location list is a typed, manifested advisory hook until the
multi-asset engine consumes it.

Simulation output includes `annual_tax_strategy_actions` with P10/P50/P90
conversion, gain-harvest, and per-treatment withdrawal amounts. It also
includes required minimum distributions in the tax-deferred withdrawal total.
It also reports annual IRMAA surcharges from the two-year MAGI lookback, real
lifetime IRMAA surcharge, and exposure probability. Simulation engine v12
charges IRMAA exactly once as a healthcare cash flow. Tier thresholds use the
filing status on the lookback-year return; current-year survival and each
person's configured `medicare_start_age` determine only how many enrolled
people owe the surcharge. An exact supplied tax-year-minus-two observation
overrides simulated MAGI for the same year; the selected source and precedence
are persisted in healthcare and tax audits and replay manifests. Historical
lookback entries can set
`filing_status`; when omitted it is inferred from the scenario's initial
status. A separate historical "MFS lived with spouse" fact is not modeled, so
the scenario-wide `married_filing_separately_lived_with_spouse` flag applies
to those entries. Do not also include IRMAA in `annual_spending`.

### Healthcare and long-term care

An optional `healthcare` object covers every configured household person
exactly once. Pre-Medicare premiums are explicit ACA or other premiums net of
any expected premium tax credit. Medicare premiums exclude IRMAA, which the
progressive tax policy calculates separately. Routine costs and LTC severity
follow `medical_inflation_rate`. Set `ltc_funding_source` to `portfolio` or
`home_equity`. Home-equity funding requires a positive real funding limit and
a reserve-only `housing_plan.care` naming the exact cash or taxable proceeds
account. Each year's post-insurance LTC demand is debited from that account,
up to the inflation-adjusted limit; only the actual unmet amount falls back to
normal portfolio spending.

LTC assumptions separately declare lifetime selection probability, bounded
onset, duration, lognormal cost severity, and insurance benefits. Results
distinguish the lifetime Bernoulli selection from care that actually becomes
active during the simulated retirement window, then report insurance,
home-equity use, portfolio cost, and conditional shortfall severity. These
costs are incremental: exclude them from `annual_spending` and spending-plan
baselines. Healthcare LTC assumptions are the single cost source when the
housing care object is reserve-only. A deterministic housing care schedule and
person-level healthcare LTC cannot be configured together.

For human review:

```bash
ynab wealth tax-strategy --scenario conversion.local.json
```

The authenticated `POST /wealth/tax/strategies/jobs` route accepts the same
planner-job body as `POST /planner/jobs` and returns the standard durable job
URLs. It shares queue, process, memory, body-size, and compute admission with
all other planner work.

To compare benefits and tradeoffs, save a baseline and each strategy as
immutable scenario revisions, then run `ynab wealth scenarios compare`.
Common-path results classify strategy inputs as tax policy and report
lifetime taxes, funded spending, shortfalls, guardrail concessions, IRMAA
exposure, and after-tax estate deltas. The stored scenario, policy IDs,
complete strategy recipe, engine identity, and common-path manifest make the
recommendation replayable.

This remains a retirement-planning engine, not tax preparation software. It
does not model itemized deductions, NIIT, AMT, tax-loss harvesting, Roth
conversion seasoning, qualified charitable distributions, local Indiana
income tax, or the joint-life RMD exception. Birth-year age tests use
calendar-year age; the
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

## Model housing decisions without making the house spendable

Add `housing_plan` only to an account-aware scenario with explicit tax buckets.
The configured home value is separate from `starting_portfolio`: a `keep`
decision can raise the reported estate, but it can never fund spending. Cash
enters the portfolio only through an explicit `sell`, `downsize`, `replace`,
`rent`, or `reverse_mortgage` event. Every such event names one exact linked
`cash` or `taxable` account with `proceeds_destination_account_id`; proceeds
are never distributed across accounts merely because they share a tax
treatment or owner.

Housing assumptions cover current value and cost basis, appreciation,
maintenance, property tax, insurance, selling costs, and a fixed-rate
remaining mortgage schedule. Replacement homes inherit those property-rate
assumptions, and their new mortgage cannot exceed property value. An optional
care plan adds real annual spending. Its optional `funding_account_id` must be
the exact proceeds destination of a home-equity liquidity event no later than
care begins. That reserve is debited first, and only unmet care falls back to
the normal portfolio withdrawal policy, avoiding double-counted spending.
Liquidity events, portfolio-funded housing costs, and care cannot begin before
`retirement_age`; earlier forward-mortgage payments and carrying costs remain
externally funded.

A reverse mortgage is modeled as gross nonrecourse principal. It is bounded by
`reverse_mortgage_max_ltv` (80% by default), pays origination costs and the
existing forward lien before creating portfolio cash, and stops that lien's
payment schedule. At terminal disposition, reverse debt can consume the
collateral but cannot create a negative estate claim.

For a sale, selling costs, mortgage debt, the primary-residence gain exclusion,
and taxable gain are audited separately. The progressive engine stacks that
gain with the same year's modeled ordinary income, Social Security, RMDs, and
withdrawals, then funds one combined liability. Net proceeds are deposited at
the start of the event year, before allocation alignment and account-specific
returns; an external deposit to a taxable account increases securities basis
exactly once while the home gain remains separately auditable. This is a bounded planning
estimate—not tax-preparation fidelity—and excludes unmodeled earned income,
deductions, local rules, and transaction details.
At the terminal estate calculation, home disposition gain stacks with
tax-deferred, nonqualified HSA, and taxable-account liquidation on one modeled
return rather than receiving an independent set of brackets.

Project the deterministic housing ledger before running Monte Carlo:

```bash
ynab wealth housing project \
  --scenario examples/housing-scenario.example.json \
  --json
```

The authenticated HTTP equivalent is `POST /wealth/housing/project` with
`{"scenario": ...}`. CLI and HTTP return the same manifest, annual
home/equity/cash-flow/action rows, and ending disposition estimate. Full
simulation output additionally reports liquid ending balance separately from
housing-inclusive before- and after-tax estate percentiles.

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
portfolio and per-account valuation provenance, engine identity, and—when historical
bootstrapping is used—the exact observations and their fingerprints.

Compare one baseline with one or more alternatives:

```bash
ynab wealth scenarios compare \
  --baseline 00000000-0000-4000-8000-000000000001 \
  --alternative 00000000-0000-4000-8000-000000000002 \
  --named-stress equity_crash \
  --json
```

Alternatives must have identical horizon, trial count, seed, return model,
return assumptions, inflation, fees, bootstrap settings, and historical
observations. The engine prepares those return and inflation paths once and
uses the same arrays for every revision, so the delta reflects scenario inputs
rather than unrelated Monte Carlo noise. Generated arrays are closed after the
comparison and are never persisted.
The optional named-stress selector is part of the common-path manifest and
comparison hash. It is replayed through the same registered stress definition
for every alternative; generated arrays remain ephemeral.

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

The default one-asset model uses independent annual lognormal returns described
by an arithmetic mean and volatility. The multi-asset model draws correlated
lognormal shocks from the validated market matrix. The historical model uses a
stationary block bootstrap, which retains multi-year return sequences and can
retain paired inflation observations. Contributions occur after each
pre-retirement year's return, including active contribution cash-flow streams.
Retirement spending includes active expense cash-flow streams, then subtracts
active income streams, and is withdrawn after each retirement year's return.
Fees reduce gross returns. A scenario may use either the legacy blended
withdrawal tax rate or the explicit account-aware tax model.

The example scenario uses a VTSAX-like total-stock-market profile: a
conservative 6% nominal arithmetic return and 16% annual volatility. These are
editable planning assumptions, not a claim about future VTSAX performance.

Balances at retirement and the plan horizon are reported in today's dollars.
Success means the portfolio never failed to meet a scheduled withdrawal
through `end_age`. The result is a scenario comparison tool, not a forecast or
financial advice.

Important limitations:

- Parametric return vectors are independent from year to year and do not model
  regime changes.
- Historical results are bounded by the quality and representativeness of the
  supplied observation set.
- Runs are limited to 100,000 trials until the engine supports bounded
  batch-wise percentile aggregation.
- Account-aware taxes use explicit effective rates rather than progressive
  federal and state tax-return calculations.
- YNAB has account balances, not security holdings; allocations are explicit
  planning assumptions.
- YNAB investment tracking does not distinguish market gains from
  reconciliation adjustments.
- Social Security PIAs and pension amounts should come from authoritative
  personal estimates.

## Continuous-calibration evidence boundary

Calibration drift is plan-versus-observation, not an arbitrary comparison of
two signed observations. A snapshot accepts the following typed plan baselines:

- spending selects and exactly equals `resolved_scenario.annual_spending`;
- contributions select and exactly equals
  `resolved_scenario.annual_contribution`; and
- whole-portfolio balance selects and exactly equals
  `resolved_scenario.starting_portfolio`.

The copied baseline observation must carry the matching `plan_field` selector.
Account-level balance selects and exactly equals the linked
`tax_buckets[account_id=…].starting_balance`. Account-level allocation selects
and exactly equals
`portfolio_allocation.accounts[account_id=…].target`. The snapshot verifier
resolves both selectors from the immutable scenario revision; external
observations cannot declare those plan values for themselves.

Debt payoff is different because the scenario currently has no typed
point-in-time debt balance. It compares one prior signed snapshot observation
with one later signed current observation. Both roles, their chronology, and
their source evidence are mandatory. This is a historical change detector, not
a claim that either amount came from a plan.

One exact upstream fact cannot be copied into multiple snapshot observations by
changing only its presentation URI. Within either side of a scalar sum, a
stable `(source_kind, source_id)` may appear only once even when its digest,
timestamp, URI, or batch version differs. A stable source may still appear in
different chronological roles (for example, plan baseline and current) when
those roles refer to distinct signed versions and every earlier-role source
`observed_at` strictly precedes the corresponding later-role source timestamp.
Equal-time rehashes fail closed.

## Living-plan automation

An immutable calibration profile binds a saved scenario revision to one YNAB
budget, a versioned materiality/freshness policy, and any reviewed allocation
observations. A completed sync invokes calibration only after every cache row
and server-knowledge checkpoint is durable. The resulting snapshot is written
before its durable run is admitted.

Each snapshot contains the resolved scenario, source batch and watermarks,
copied observations with stable source identities, stale/frozen assessments,
and replay-verified drift events for spending, contributions, linked account
balances, reviewed allocation weights, debt payoff, and valuation freshness.
Contribution actuals include only positive transfers from outside the selected
liquid-account set. Direct non-transfer activity is excluded because YNAB
cannot safely distinguish a contribution from investment income,
reconciliation, or a market-value adjustment. Because that evidence is
incomplete, it remains informational and does not overwrite the saved
contribution plan without a future complete, explicitly reviewed source.
The `(profile, sync batch)` identity is unique, so a retry or process restart
cannot create a different snapshot or a second run.

The worker evaluates the original plan, the combined calibrated inputs, and
bounded one-factor counterfactuals using the scenario's seeded simulation. Its
report attributes changes in success probability, median cumulative shortfall,
median lifetime tax, and median after-tax estate value. An explicit interaction
residual reconciles the individual marginals to the combined result. Only
material, sufficiently confident planning-input drift changes the calibrated
scenario. Events are grouped by driving kind before resource admission, while
immaterial and observational-only context stays evidence-linked with a zero
direct model effect. Historical scenarios copy the saved revision's immutable
dataset into the snapshot and replay from that exact series.

Material alerts contain only the drift kind, stable subject ID, lifecycle
state, and timestamps. Dollar values, transaction details, account display
names, source URIs, and evidence payloads remain in authenticated snapshot
storage. Opening, acknowledgement, resolution, and reopening are append-only
lifecycle events.

Completed-sync markers bind watermarks to change-batch identities. Accounts,
transactions, and spending categories retain their last change batch; capture
checks both before and after its source reads and fails closed if a newer sync
is incomplete or the checkpoint moves. Profile state is append-only. Disabling
an obsolete profile stops future post-sync captures without deleting evidence.
A failed capture on one active profile is recorded with a privacy-bounded error
code and does not prevent other active profiles from capturing the same batch.

The supported operator surface is:

```text
ynab wealth calibration create --scenario-revision UUID --budget-id UUID
ynab wealth calibration capture --profile UUID --sync-batch UUID
ynab wealth calibration process
ynab wealth calibration status --profile UUID --json
ynab wealth calibration disable --profile UUID
ynab wealth calibration acknowledge --profile UUID --alert UUID
```

The authenticated HTTP equivalents live under `/wealth/calibration`. The API
lifespan runs the same durable worker; `wealth calibration process` is suitable
for a systemd timer when the HTTP service is not running.

Natural next steps include additional tax jurisdictions and policy plugins,
broader investment-data adapters, versioned economic assumption sets, and
review workflows for adopting calibrated observations into new scenario
revisions.

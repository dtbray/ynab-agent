# Debt Payoff Planning

`ynab-agent reports debt-plan` projects payoff timing from cached YNAB account
balances plus optional local metadata that YNAB does not reliably expose for
credit cards.

YNAB can provide current card balances, card payment category coverage, and in
some cases debt metadata. APRs, true card minimums, payoff priorities, and
separate pay-over-time balances should be supplied in a local JSON file.

Example:

```json
{
  "accounts": {
    "Rewards Visa": {
      "apr": 24.99,
      "minimum_payment": 75,
      "priority": 1
    }
  },
  "pay_over_time": [
    {
      "name": "Vet Bill Pay Over Time",
      "balance": 1200,
      "minimum_payment": 200,
      "monthly_fee": 12,
      "promo_end": "2026-09-01",
      "priority": 0
    }
  ]
}
```

Run:

```bash
ynab-agent reports debt-plan \
  --month 2026-07-01 \
  --monthly-payment 750 \
  --strategy avalanche \
  --config debt-plan.local.json
```

Strategies:

- `avalanche`: pays highest APR plus fee rate first, after minimums.
- `snowball`: pays the smallest balance first, after minimums.
- `priority`: pays lower `priority` values first, after minimums.

Keep real balances and account names in an ignored local config file.

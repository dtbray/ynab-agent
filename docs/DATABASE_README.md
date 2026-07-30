# YNAB M&T Legacy Budget - SQLite Database

## Database File
**Location:** `./ynab_mt_legacy.db`  
**Size:** 260KB  
**Budget:** M&T (legacy/old info)  
**Date Range:** 2020-09-01 to 2023-02-09

## Database Schema

### Tables

| Table | Records | Description |
|-------|---------|-------------|
| `budgets` | 1 | Budget metadata |
| `accounts` | 12 | Bank accounts, credit cards, investments |
| `payees` | 44 | Transaction payees |
| `category_groups` | 10 | Category groupings |
| `categories` | 64 | Budget categories |
| `transactions` | 215 | All transactions |
| `budget_months` | 0 | Monthly budget data (not populated) |
| `month_categories` | 0 | Category data per month (not populated) |

## Sample Queries

### 1. View All Accounts
```sql
SELECT name, type, printf('$%.2f', balance/1000.0) as balance
FROM accounts
ORDER BY balance DESC;
```

### 2. Spending by Year
```sql
SELECT 
    substr(t.date, 1, 4) as year,
    printf('$%.2f', SUM(ABS(t.amount))/1000.0) as total_spending,
    COUNT(*) as transactions,
    printf('$%.2f', AVG(ABS(t.amount))/1000.0) as avg_transaction
FROM transactions t
LEFT JOIN payees p ON t.payee_id = p.id
WHERE t.amount < 0 
    AND (p.name IS NULL OR (p.name NOT LIKE '%Starting Balance%' AND p.name NOT LIKE '%Transfer%'))
GROUP BY year
ORDER BY year;
```

### 3. Top Spending by Payee (All Time)
```sql
SELECT 
    COALESCE(p.name, 'Unknown') as payee,
    printf('$%.2f', SUM(ABS(t.amount))/1000.0) as total,
    COUNT(*) as transactions
FROM transactions t
LEFT JOIN payees p ON t.payee_id = p.id
WHERE t.amount < 0 
    AND (p.name IS NULL OR (p.name NOT LIKE '%Starting Balance%' AND p.name NOT LIKE '%Transfer%'))
GROUP BY payee
ORDER BY SUM(ABS(t.amount)) DESC
LIMIT 15;
```

### 4. Monthly Spending Trend
```sql
SELECT 
    substr(date, 1, 7) as month,
    printf('$%.2f', SUM(ABS(amount))/1000.0) as total_spent,
    COUNT(*) as transactions
FROM transactions
WHERE amount < 0
GROUP BY month
ORDER BY month;
```

### 5. Largest Single Transactions
```sql
SELECT 
    t.date,
    p.name as payee,
    printf('$%.2f', ABS(t.amount)/1000.0) as amount,
    t.memo
FROM transactions t
LEFT JOIN payees p ON t.payee_id = p.id
WHERE t.amount < 0
ORDER BY ABS(t.amount) DESC
LIMIT 20;
```

### 6. Transactions by Account
```sql
SELECT 
    a.name as account,
    COUNT(*) as transaction_count,
    printf('$%.2f', SUM(t.amount)/1000.0) as net_amount
FROM transactions t
JOIN accounts a ON t.account_id = a.id
GROUP BY a.name
ORDER BY transaction_count DESC;
```

### 7. Search Transactions by Payee
```sql
SELECT 
    t.date,
    printf('$%.2f', t.amount/1000.0) as amount,
    t.memo
FROM transactions t
JOIN payees p ON t.payee_id = p.id
WHERE p.name LIKE '%Target%'
ORDER BY t.date;
```

### 8. Spending by Category Group
```sql
SELECT 
    cg.name as category_group,
    COUNT(*) as transactions,
    printf('$%.2f', SUM(ABS(t.amount))/1000.0) as total
FROM transactions t
JOIN categories c ON t.category_id = c.id
JOIN category_groups cg ON c.category_group_id = cg.id
WHERE t.amount < 0
GROUP BY cg.name
ORDER BY SUM(ABS(t.amount)) DESC;
```

## Key Statistics

### Yearly Breakdown
| Year | Total Spending | Transactions | Avg Transaction | Top Payee |
|------|----------------|--------------|-----------------|-----------|
| 2020 | $200.70 | 7 | $28.67 | Central Checkout |
| 2021 | $9,337.88 | 100 | $93.38 | Wealthfront |
| 2022 | $3,346.93 | 25 | $133.88 | Amazon |

### Top 10 Payees (All Time)
| Rank | Payee | Amount | Transactions |
|------|-------|--------|--------------|
| 1 | Wealthfront | $7,250.00 | 6 |
| 2 | Target | $2,241.62 | 59 |
| 3 | Central Checkout | $1,455.66 | 27 |
| 4 | Amazon | $1,508.20 | 8 |
| 5 | Central Checkout Indy | $230.36 | 13 |

## Notes

- All monetary amounts are stored in **milliunits** (thousandths of currency)
- Use `amount/1000.0` to convert to dollars
- Negative amounts represent spending/outflow
- Positive amounts represent income/inflow
- The budget has no meaningful category assignments (all "Uncategorized")

## Export Tool

To re-export the data:
```powershell
./Export-Simple.ps1
```

To export a different budget:
```powershell
./Export-Simple.ps1 -BudgetName "Our Budget" -DatabasePath "./our_budget.db"
```

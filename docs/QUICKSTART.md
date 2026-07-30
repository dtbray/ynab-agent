# YNAB Secure Module - Quick Start Guide

## 5-Minute Setup

### 1. Set Your Token

**Option A: Environment Variable (Recommended)**
```bash
# Create .env file (NEVER commit this!)
echo 'YNAB_PAT="your-token-here"' > .env
source .env
```

```powershell
# In PowerShell
Import-Module ./YNAB-Secure.psm1
Set-YnabAccessToken -FromEnvironment
```

**Option B: Interactive (One-time)**
```powershell
Import-Module ./YNAB-Secure.psm1
Set-YnabAccessToken -Token (Read-Host -AsSecureString "Enter YNAB token")
```

### 2. Test Connection

```powershell
Get-YnabBudgets | Format-Table name, last_modified_on
```

### 3. View Your Data

```powershell
# List accounts
Get-YnabAccounts | Format-Table name, type, @{N="Balance"; E={Format-YnabCurrency $_.balance}}

# List categories
Get-YnabCategories | ForEach-Object { $_.categories } | Select-Object name, balance | Format-Table

# Recent transactions
Get-YnabTransactions -SinceDate "2024-01-01" | Select-Object date, payee_name, @{N="Amount"; E={Format-YnabCurrency $_.amount}} | Format-Table
```

### 4. Export to Database

```powershell
./Export-Secure.ps1 -FromEnvironment -DatabasePath "./my_budget.db"
```

### 5. Clean Up

```powershell
Clear-YnabAccessToken
```

## Common Tasks

### Find a Category
```powershell
Get-YnabCategories | ForEach-Object { $_.categories } | Where-Object { $_.name -like "*Groceries*" }
```

### Add a Transaction (Safely)
```powershell
# Preview first with -WhatIf
New-YnabTransaction -WhatIf `
    -AccountId "your-account-id" `
    -Date (Get-Date -Format "yyyy-MM-dd") `
    -Amount (ConvertTo-Milliunits -25.00) `
    -PayeeName "Coffee Shop"

# If it looks good, run without -WhatIf
New-YnabTransaction `
    -AccountId "your-account-id" `
    -Date (Get-Date -Format "yyyy-MM-dd") `
    -Amount (ConvertTo-Milliunits -25.00) `
    -PayeeName "Coffee Shop"
```

### Budget Money to Category
```powershell
# Get category ID
$category = Get-YnabCategories | ForEach-Object { $_.categories } | Where-Object { $_.name -eq "Groceries" }

# Budget $500 (preview first)
Update-YnabMonthCategory -WhatIf -CategoryId $category.id -Budgeted (ConvertTo-Milliunits 500.00)
```

### Query Your Database
```bash
# Total spending by year
sqlite3 my_budget.db "SELECT substr(date, 1, 4) as year, printf('\$%.2f', SUM(ABS(amount))/1000.0) FROM transactions WHERE amount < 0 GROUP BY year;"

# Top payees
sqlite3 my_budget.db "SELECT p.name, printf('\$%.2f', SUM(ABS(t.amount))/1000.0) FROM transactions t JOIN payees p ON t.payee_id = p.id WHERE t.amount < 0 GROUP BY p.name ORDER BY SUM(ABS(t.amount)) DESC LIMIT 10;"
```

## Security Reminders

- ✅ Always use `-FromEnvironment` or `Read-Host -AsSecureString` for tokens
- ✅ Always call `Clear-YnabAccessToken` when done
- ✅ Always use `-WhatIf` before making changes
- ✅ Never commit `.env` files or `.db` databases
- ✅ Never share your YNAB token

## Need Help?

- Full docs: `README-SECURE.md`
- Security info: `SECURITY.md`
- Migration guide: `SECURITY_HARDENING_SUMMARY.md`
- API reference: `YNAB_API_Reference.md`

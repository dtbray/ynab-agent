# YNAB PowerShell Module - Security Hardened

A secure PowerShell module for interacting with the You Need A Budget (YNAB) API.

> ⚠️ **SECURITY NOTICE**: This is the secure version (2.0) with hardened token storage and input validation. For the original version, see `YNAB.psm1`.

## Quick Start

### 1. Install

```powershell
# Clone or download the module
# Place YNAB-Secure.psm1 in your module path
```

### 2. Configure Token (Choose ONE method)

**Method A: Environment Variable (Recommended)**
```bash
# Add to your .env file
echo 'YNAB_PAT="your-token-here"' > .env
```

```powershell
# In PowerShell
Import-Module ./YNAB-Secure.psm1
Set-YnabAccessToken -FromEnvironment
```

**Method B: SecureString (Interactive)**
```powershell
Import-Module ./YNAB-Secure.psm1
Set-YnabAccessToken -Token (Read-Host -AsSecureString "Enter YNAB token")
```

**Method C: SecureString (Scripted)**
```powershell
$token = "your-token" | ConvertTo-SecureString -AsPlainText -Force
Set-YnabAccessToken -Token $token
```

### 3. Start Using

```powershell
# List budgets
Get-YnabBudgets

# Get accounts
Get-YnabAccounts

# Create a transaction (with -WhatIf safety)
New-YnabTransaction -WhatIf -AccountId "..." -Amount -5000 -PayeeName "Test"

# Clear token when done
Clear-YnabAccessToken
```

## Security Features

| Feature | Implementation |
|---------|---------------|
| Token Storage | SecureString in memory |
| Memory Cleanup | Automatic + manual clear function |
| Rate Limiting | Built-in tracking (200/hr) |
| Input Validation | UUID regex, parameter bounds |
| SQL Injection Prevention | Parameterized queries |
| TLS Enforcement | TLS 1.2+ required |
| Safe Defaults | `-WhatIf` support on destructive ops |

## Export to SQLite (Secure)

```powershell
# Using environment variable
./Export-Secure.ps1 -BudgetName "My Budget" -FromEnvironment

# Using SecureString
./Export-Secure.ps1 -BudgetName "My Budget" -Token (Read-Host -AsSecureString)

# Custom output path
./Export-Secure.ps1 -BudgetName "My Budget" -FromEnvironment -DatabasePath "./my_budget.db"
```

## Function Reference

### Authentication
- `Set-YnabAccessToken` - Set token securely
- `Clear-YnabAccessToken` - Clear token from memory

### Read Operations
- `Get-YnabBudgets` - List all budgets
- `Get-YnabBudget` - Get budget details
- `Get-YnabAccounts` - List accounts
- `Get-YnabCategories` - List categories
- `Get-YnabPayees` - List payees
- `Get-YnabTransactions` - List transactions

### Write Operations (with -WhatIf support)
- `New-YnabAccount` - Create account
- `New-YnabTransaction` - Create transaction
- `Update-YnabMonthCategory` - Update category budget

### Utilities
- `ConvertTo-Milliunits` - Dollars to milliunits
- `ConvertFrom-Milliunits` - Milliunits to dollars
- `Format-YnabCurrency` - Format for display

## Migration from v1.x

### Breaking Changes

| Old (v1) | New (v2) |
|----------|----------|
| `Set-YnabAccessToken -Token "plain"` | `Set-YnabAccessToken -FromEnvironment` or `-Token (Read-Host -AsSecureString)` |
| String token parameter | SecureString token parameter |
| Direct execution | Supports `-WhatIf` on writes |

### Migration Steps

1. Move your token to environment variable:
   ```powershell
   [Environment]::SetEnvironmentVariable("YNAB_PAT", "your-token", "User")
   ```

2. Update your scripts:
   ```powershell
   # Old
   Set-YnabAccessToken -Token "abc123"
   
   # New
   Set-YnabAccessToken -FromEnvironment
   ```

3. Add error handling:
   ```powershell
   try {
       Set-YnabAccessToken -FromEnvironment
       Get-YnabBudgets
   }
   finally {
       Clear-YnabAccessToken
   }
   ```

## Security Checklist

Before using in production:

- [ ] Token stored in environment variable or secure vault
- [ ] `.env` file added to `.gitignore`
- [ ] No hardcoded tokens in any scripts
- [ ] `Clear-YnabAccessToken` called when done
- [ ] `-WhatIf` tested on all write operations
- [ ] Module path is secure (no world-readable permissions)

## Troubleshooting

### "Access token not set"
Call `Set-YnabAccessToken` before using other functions.

### "YNAB_PAT environment variable not set"
Set the environment variable or pass `-Token` directly.

### Rate limit errors
The module tracks API calls. Wait an hour if limit exceeded.

### "Invalid ID format"
All IDs (BudgetId, AccountId, etc.) must be valid UUIDs.

## Files

| File | Purpose |
|------|---------|
| `YNAB-Secure.psm1` | Secure module (v2) |
| `Export-Secure.ps1` | Secure SQLite export |
| `schema.sql` | Database schema |
| `.env.template` | Template for environment variables |
| `.gitignore` | Prevents committing secrets |
| `SECURITY.md` | Security policy |

## License

See LICENSE file.

## Contributing

See CONTRIBUTING.md (when created).

Run security checks before submitting:
```powershell
Invoke-ScriptAnalyzer -Path ./YNAB-Secure.psm1 -Severity Warning
```

# YNAB API Reference

## Overview

- **Base URL:** `https://api.ynab.com/v1`
- **Format:** REST, JSON
- **Security:** HTTPS enforced, Bearer token authentication
- **Documentation:** https://api.ynab.com/

## Authentication

### Personal Access Token

1. Sign into YNAB web app
2. Go to Account Settings → Developer Settings
3. Under "Personal Access Tokens", click "New Token"
4. Enter password and generate

### Using the Token

HTTP Bearer Authentication (RFC6750):

```bash
curl -H "Authorization: Bearer <ACCESS_TOKEN>" https://api.ynab.com/v1/budgets
```

Or as query parameter:
```bash
curl https://api.ynab.com/v1/budgets?access_token=<ACCESS_TOKEN>
```

## Endpoints

### User
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/user` | Get authenticated user info |

### Budgets
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets` | List all budgets (summary) |
| GET | `/budgets/{budget_id}` | Get single budget with all related entities |
| GET | `/budgets/{budget_id}/settings` | Get budget settings |

### Accounts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/accounts` | List all accounts |
| POST | `/budgets/{budget_id}/accounts` | Create new account |
| GET | `/budgets/{budget_id}/accounts/{account_id}` | Get single account |

### Categories
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/categories` | List all categories |
| GET | `/budgets/{budget_id}/categories/{category_id}` | Get single category |
| PATCH | `/budgets/{budget_id}/categories/{category_id}` | Update category |
| GET | `/budgets/{budget_id}/months/{month}/categories/{category_id}` | Get category for specific month |
| PATCH | `/budgets/{budget_id}/months/{month}/categories/{category_id}` | Update category for specific month |

### Payees
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/payees` | List all payees |
| GET | `/budgets/{budget_id}/payees/{payee_id}` | Get single payee |
| PATCH | `/budgets/{budget_id}/payees/{payee_id}` | Update payee |

### Transactions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/transactions` | List all transactions |
| POST | `/budgets/{budget_id}/transactions` | Create transaction(s) |
| PATCH | `/budgets/{budget_id}/transactions` | Update multiple transactions |
| POST | `/budgets/{budget_id}/transactions/import` | Import transactions |
| GET | `/budgets/{budget_id}/transactions/{transaction_id}` | Get single transaction |
| PUT | `/budgets/{budget_id}/transactions/{transaction_id}` | Update transaction |
| GET | `/budgets/{budget_id}/accounts/{account_id}/transactions` | List transactions for account |
| GET | `/budgets/{budget_id}/categories/{category_id}/transactions` | List transactions for category |
| GET | `/budgets/{budget_id}/payees/{payee_id}/transactions` | List transactions for payee |
| GET | `/budgets/{budget_id}/months/{month}/transactions` | List transactions for month |

### Scheduled Transactions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/scheduled_transactions` | List all scheduled transactions |
| POST | `/budgets/{budget_id}/scheduled_transactions` | Create scheduled transaction |
| GET | `/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}` | Get single scheduled transaction |
| PUT | `/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}` | Update scheduled transaction |
| DELETE | `/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}` | Delete scheduled transaction |

### Months
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/months` | List all budget months |
| GET | `/budgets/{budget_id}/months/{month}` | Get single budget month |

### Payee Locations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/budgets/{budget_id}/payee_locations` | List all payee locations |
| GET | `/budgets/{budget_id}/payee_locations/{payee_location_id}` | Get single payee location |
| GET | `/budgets/{budget_id}/payees/{payee_id}/payee_locations` | List locations for payee |

## Data Types

### Amounts (Milliunits)
All currency amounts are in **milliunits** - integer representation of amounts × 1000.

| Currency | Example |
|----------|---------|
| $10.50 | 10500 |
| -$25.00 | -25000 |
| $0.01 | 10 |

### Dates
ISO 8601 format: `YYYY-MM-DD`

Example: `2024-03-04`

### UUIDs
Budgets, accounts, categories, etc. use UUID format: `6ee704d9-ee24-4c36-b1a6-cb8ccf6a216c`

## Response Format

All responses follow this structure:

```json
{
  "data": {
    // Resource-specific data
  }
}
```

### Error Responses

Errors include details in the response body:

```json
{
  "error": {
    "id": "error_id",
    "name": "ErrorName",
    "detail": "Description of what went wrong"
  }
}
```

## Rate Limiting

- API enforces rate limiting
- Returns `429 Too Many Requests` if exceeded
- Implement caching to avoid unnecessary requests

## SDKs and Libraries

### JavaScript/TypeScript (Official)
```bash
npm install ynab
```

```javascript
const ynab = require("ynab");
const ynabAPI = new ynab.API("<access_token>");
```

### Rust
```toml
ynab-api = "4.0.0"
```

## Best Practices

1. **Cache data** when possible to reduce API calls
2. **Use milliunits** for all monetary values
3. **Handle rate limits** (429 responses)
4. **Keep tokens secret** - don't commit to version control
5. **Use HTTPS** - enforced by API

## Example Response: List Budgets

```json
{
  "data": {
    "budgets": [
      {
        "id": "6ee704d9-ee24-4c36-b1a6-cb8ccf6a216c",
        "name": "My Budget",
        "last_modified_on": "2017-12-01T12:40:37.867Z",
        "first_month": "2017-11-01",
        "last_month": "2017-11-01"
      }
    ]
  }
}
```

## Related Links

- [Official Documentation](https://api.ynab.com/)
- [JavaScript SDK](https://github.com/ynab/ynab-sdk-js)
- [API Status](https://ynabstatus.com)
- [Works with YNAB](https://api.ynab.com/#works-with-ynab)
- [Starter Kit](https://github.com/ynab/ynab-sdk-js#starter-kit)

## Migration Note

The API was previously at `https://api.youneedabudget.com/v1` and moved to `https://api.ynab.com/v1` in 2023. The old URL still works but redirects to the new one.

---

*Last updated: 2026-03-04*

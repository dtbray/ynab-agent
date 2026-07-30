-- YNAB Database Schema
-- Legacy SQLite bootstrap schema.
--
-- The Python agent now initializes tables through SQLAlchemy metadata in
-- ynab_agent.db.models so the cache can run on SQLite, Postgres, or
-- MariaDB/MySQL. Keep this file only as a reference for the original schema
-- shape until migrations fully replace it.

-- Enable foreign key support
PRAGMA foreign_keys = ON;

-- Budgets table
CREATE TABLE IF NOT EXISTS budgets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    first_month TEXT,
    last_month TEXT,
    last_modified_on TEXT,
    date_format TEXT,
    currency_format_iso_code TEXT,
    currency_format_example TEXT
);

-- Accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    on_budget INTEGER,
    closed INTEGER,
    note TEXT,
    balance INTEGER,
    cleared_balance INTEGER,
    uncleared_balance INTEGER,
    transfer_payee_id TEXT,
    direct_import_linked INTEGER,
    direct_import_in_error INTEGER,
    last_reconciled_at TEXT,
    debt_original_balance INTEGER,
    debt_interest_rates TEXT,
    debt_minimum_payments TEXT,
    debt_escrow_amounts TEXT,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE
);

-- Payees table
CREATE TABLE IF NOT EXISTS payees (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    name TEXT,
    transfer_account_id TEXT,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE
);

-- Category Groups table
CREATE TABLE IF NOT EXISTS category_groups (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    name TEXT,
    hidden INTEGER DEFAULT 0,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE
);

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    category_group_id TEXT,
    name TEXT,
    hidden INTEGER DEFAULT 0,
    original_category_group_id TEXT,
    note TEXT,
    budgeted INTEGER,
    activity INTEGER,
    balance INTEGER,
    goal_type TEXT,
    goal_needs_whole_amount INTEGER,
    goal_day INTEGER,
    goal_cadence INTEGER,
    goal_cadence_frequency INTEGER,
    goal_creation_month TEXT,
    goal_target INTEGER,
    goal_target_month TEXT,
    goal_percentage_complete INTEGER,
    goal_months_to_budget INTEGER,
    goal_under_funded INTEGER,
    goal_overall_funded INTEGER,
    goal_overall_left INTEGER,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE,
    FOREIGN KEY (category_group_id) REFERENCES category_groups(id) ON DELETE SET NULL
);

-- Budget Months table (for monthly category data)
CREATE TABLE IF NOT EXISTS budget_months (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    month TEXT NOT NULL,
    note TEXT,
    income INTEGER,
    budgeted INTEGER,
    activity INTEGER,
    to_be_budgeted INTEGER,
    age_of_money INTEGER,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE,
    UNIQUE(budget_id, month)
);

-- Month Categories (category data per month)
CREATE TABLE IF NOT EXISTS month_categories (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    month TEXT NOT NULL,
    category_id TEXT,
    budgeted INTEGER,
    activity INTEGER,
    balance INTEGER,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE,
    UNIQUE(budget_id, month, category_id)
);

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    account_id TEXT,
    category_id TEXT,
    payee_id TEXT,
    transfer_account_id TEXT,
    transfer_transaction_id TEXT,
    matched_transaction_id TEXT,
    import_id TEXT,
    date TEXT,
    amount INTEGER,
    memo TEXT,
    cleared TEXT,
    approved INTEGER,
    flag_color TEXT,
    flag_name TEXT,
    foreign_amount INTEGER,
    foreign_currency_code TEXT,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL,
    FOREIGN KEY (payee_id) REFERENCES payees(id) ON DELETE SET NULL
);

-- Sub-transactions table (for split transactions)
CREATE TABLE IF NOT EXISTS subtransactions (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    amount INTEGER,
    memo TEXT,
    payee_id TEXT,
    payee_name TEXT,
    category_id TEXT,
    transfer_account_id TEXT,
    transfer_transaction_id TEXT,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE,
    FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
);

-- Payee Locations table
CREATE TABLE IF NOT EXISTS payee_locations (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    payee_id TEXT,
    latitude TEXT,
    longitude TEXT,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE,
    FOREIGN KEY (payee_id) REFERENCES payees(id) ON DELETE CASCADE
);

-- Scheduled Transactions table
CREATE TABLE IF NOT EXISTS scheduled_transactions (
    id TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    account_id TEXT,
    payee_id TEXT,
    category_id TEXT,
    transfer_account_id TEXT,
    date_first TEXT,
    date_next TEXT,
    frequency TEXT,
    amount INTEGER,
    memo TEXT,
    flag_color TEXT,
    flag_name TEXT,
    deleted INTEGER DEFAULT 0,
    FOREIGN KEY (budget_id) REFERENCES budgets(id) ON DELETE CASCADE
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category_id);
CREATE INDEX IF NOT EXISTS idx_transactions_payee ON transactions(payee_id);
CREATE INDEX IF NOT EXISTS idx_transactions_budget ON transactions(budget_id);
CREATE INDEX IF NOT EXISTS idx_accounts_budget ON accounts(budget_id);
CREATE INDEX IF NOT EXISTS idx_categories_budget ON categories(budget_id);
CREATE INDEX IF NOT EXISTS idx_payees_budget ON payees(budget_id);
CREATE INDEX IF NOT EXISTS idx_month_categories_month ON month_categories(month);

-- Notification tracking for reminder system
CREATE TABLE IF NOT EXISTS transaction_notifications (
    transaction_id TEXT PRIMARY KEY,
    notified_at TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notifications_notified_at 
ON transaction_notifications(notified_at);

-- Cached sync delta rows for answering "what changed since last sync"
CREATE TABLE IF NOT EXISTS budget_changes (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    budget_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_name TEXT,
    payload TEXT NOT NULL,
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_budget_changes_batch ON budget_changes(batch_id);
CREATE INDEX IF NOT EXISTS idx_budget_changes_budget ON budget_changes(budget_id);
CREATE INDEX IF NOT EXISTS idx_budget_changes_resource ON budget_changes(resource);
CREATE INDEX IF NOT EXISTS idx_budget_changes_entity ON budget_changes(entity_id);
CREATE INDEX IF NOT EXISTS idx_budget_changes_recorded_at ON budget_changes(recorded_at);

# Database Migrations

The application still calls `Base.metadata.create_all()` from `DatabaseManager.initialize()` so local
SQLite databases and tests can bootstrap quickly. Shared or long-lived databases should use Alembic so
schema changes are explicit and repeatable.

## Configure

Set the same SQLAlchemy URL the app uses:

```bash
export DATABASE_URL="postgresql+asyncpg://ynab:ynab@localhost:5432/ynab"
```

For local SQLite, omit `DATABASE_URL`; Alembic uses `database_path` from the app settings and defaults
to `sqlite+aiosqlite:///ynab_agent.db`.

## Apply Migrations

```bash
uv run alembic upgrade head
```

## Create a Migration

After changing models in `src/ynab_agent/db/models.py`, generate a revision and review the file before
committing it:

```bash
uv run alembic revision --autogenerate -m "describe schema change"
uv run alembic upgrade head
```

## Postgres Deploys

Run `uv run alembic upgrade head` before starting workers against a Postgres database. This keeps table
creation and later schema changes independent from app startup and avoids relying on opportunistic
`create_all()` behavior in production.

Saved wealth-scenario revisions and comparison results are append-only
application records. Apply migrations before using `ynab wealth scenarios` or
the `/wealth/scenarios/revisions` HTTP endpoints; the service does not create
these tables opportunistically during a command.

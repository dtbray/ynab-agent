FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml LICENSE ./
COPY src ./src
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations

RUN pip install --no-cache-dir ".[postgres]"

VOLUME ["/data"]

CMD ["ynab", "poll"]

from decimal import Decimal

from ynab_agent.db._coercion import as_int


def test_as_int_accepts_sql_decimal_scientific_notation() -> None:
    assert as_int(Decimal("1E+4")) == 10_000
    assert as_int("1E+4") == 10_000
    assert as_int(10_000) == 10_000

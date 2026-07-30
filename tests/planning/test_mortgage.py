from datetime import date

import pytest

from ynab_agent.planning.mortgage import project_mortgage


def test_projects_payoff_from_ynab_dated_loan_terms() -> None:
    projection = project_mortgage(
        account_id="mortgage",
        balance_milliunits=-184_229_470,
        interest_rates={"2021-10-01": 2_750},
        minimum_payments={
            "2021-10-01": 1_144_530,
            "2025-07-01": 1_157_200,
        },
        escrow_amounts={
            "2021-10-01": 189_920,
            "2025-07-01": 292_340,
        },
        as_of=date(2026, 7, 29),
        current_age=30,
    )

    assert projection.current_balance == 184_229.47
    assert projection.annual_interest_rate == 0.0275
    assert projection.monthly_payment == 1_157.20
    assert projection.monthly_escrow == 292.34
    assert projection.monthly_principal_and_interest == pytest.approx(864.86)
    assert projection.annual_principal_and_interest == pytest.approx(
        10_378.32
    )
    assert projection.remaining_months == 293
    assert projection.estimated_payoff_date == date(2050, 12, 1)
    assert projection.estimated_payoff_age == 55
    assert len(projection.source_sha256) == 64


def test_rejects_non_amortizing_payment() -> None:
    with pytest.raises(ValueError, match="does not amortize"):
        project_mortgage(
            account_id="mortgage",
            balance_milliunits=-200_000_000,
            interest_rates={"2026-01-01": 70_000},
            minimum_payments={"2026-01-01": 1_000_000},
            escrow_amounts={},
            as_of=date(2026, 7, 1),
            current_age=30,
        )

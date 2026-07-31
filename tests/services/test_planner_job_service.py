from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from datetime import date, datetime, timezone

import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.paths import ResourceLimitError
from ynab_agent.planning.stress import NamedStressName
from ynab_agent.resources.historical import RegisteredHistoricalDatasets
from ynab_agent.services.planner_jobs import (
    PlannerErrorCode,
    PlannerExecutionOutput,
    PlannerExecutionPolicy,
    PlannerJob,
    PlannerJobCreateResult,
    PlannerJobPayload,
    PlannerJobPublicError,
    PlannerJobService,
    PlannerJobServiceError,
    PlannerJobState,
    PlannerJobSubmission,
    canonical_job_request_hash,
)
from ynab_agent.services.wealth import (
    AccountFreshness,
    CashFlowCandidate,
    WealthAccount,
    WealthService,
)


class _WealthRepository:
    async def get_accounts(
        self,
        account_ids: Collection[str],
    ) -> Sequence[WealthAccount]:
        return ()

    async def list_account_freshness(
        self,
        *,
        tracking_only: bool,
    ) -> Sequence[AccountFreshness]:
        return ()

    async def get_cash_flow_candidates(
        self,
        account_ids: Collection[str],
        *,
        through: date,
    ) -> Sequence[CashFlowCandidate]:
        return ()


class _JobRepository:
    def __init__(self, *, cancel_after_create: bool = False) -> None:
        self.jobs: dict[str, PlannerJob] = {}
        self.by_hash: dict[str, str] = {}
        self.create_calls = 0
        self.cancel_after_create = cancel_after_create

    async def get_by_request_hash(
        self,
        request_hash: str,
    ) -> PlannerJob | None:
        job_id = self.by_hash.get(request_hash)
        job = self.jobs.get(job_id) if job_id is not None else None
        if job is not None and job.state not in {PlannerJobState.FAILED, PlannerJobState.CANCELLED}:
            return job
        return None

    async def create_job(
        self,
        *,
        job_id: str,
        request_hash: str,
        payload: PlannerJobPayload,
    ) -> PlannerJobCreateResult:
        self.create_calls += 1
        existing = await self.get_by_request_hash(request_hash)
        if existing is not None:
            return PlannerJobCreateResult(job=existing, created=False)
        now = datetime.now(timezone.utc)
        job = PlannerJob(
            id=job_id,
            request_hash=request_hash,
            state=PlannerJobState.ACCEPTED,
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job_id] = job
        self.by_hash[request_hash] = job_id
        if self.cancel_after_create:
            raise asyncio.CancelledError()
        return PlannerJobCreateResult(job=job, created=True)

    async def get_job(self, job_id: str) -> PlannerJob | None:
        return self.jobs.get(job_id)

    async def prepare_recovery(self) -> None:
        return None

    async def list_accepted_jobs(
        self,
        *,
        limit: int,
        exclude_job_ids: Collection[str],
    ) -> tuple[PlannerJob, ...]:
        return ()

    async def mark_running(self, job_id: str) -> PlannerJob | None:
        return self.jobs.get(job_id)

    async def mark_succeeded(
        self,
        job_id: str,
        output: PlannerExecutionOutput,
    ) -> PlannerJob | None:
        return self.jobs.get(job_id)

    async def mark_failed(
        self,
        job_id: str,
        error: PlannerJobPublicError,
    ) -> PlannerJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        failed = job.model_copy(
            update={
                "state": PlannerJobState.FAILED,
                "error": error,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.jobs[job_id] = failed
        return failed

    async def request_cancellation(self, job_id: str) -> PlannerJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        cancelled = job.model_copy(
            update={
                "state": PlannerJobState.CANCELLED,
                "cancellation_requested": True,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.jobs[job_id] = cancelled
        return cancelled


class _Dispatcher:
    def __init__(self, *, fail_activation: bool = False) -> None:
        self.reserved: list[tuple[str, int]] = []
        self.activated: list[str] = []
        self.released: list[str] = []
        self.fail_activation = fail_activation

    async def reserve(
        self,
        job_id: str,
        required_working_bytes: int,
    ) -> None:
        self.reserved.append((job_id, required_working_bytes))

    async def activate(self, job_id: str) -> None:
        if self.fail_activation:
            raise RuntimeError("dispatcher unavailable")
        self.activated.append(job_id)

    async def release(self, job_id: str) -> None:
        self.released.append(job_id)


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "durable job",
        "current_age": 40,
        "retirement_age": 60,
        "end_age": 95,
        "starting_portfolio": 125_000,
        "annual_spending": 40_000,
        "trials": 100,
        "seed": 42,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def _allocation() -> dict[str, object]:
    return {
        "market": {
            "us_equity": {"expected_return": 0.08, "volatility": 0.18},
            "international_equity": {
                "expected_return": 0.07,
                "volatility": 0.2,
            },
            "bonds": {"expected_return": 0.04, "volatility": 0.07},
            "cash": {"expected_return": 0.02, "volatility": 0.01},
            "correlation": {
                "values": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            },
        },
        "accounts": [
            {
                "account_id": "portfolio",
                "portfolio_weight": 1,
                "target": {
                    "us_equity": 0.6,
                    "international_equity": 0.2,
                    "bonds": 0.15,
                    "cash": 0.05,
                },
            }
        ],
    }


def _service(
    repository: _JobRepository,
    dispatcher: _Dispatcher,
    *,
    maximum_working_bytes: int = 16 * 1024 * 1024,
    maximum_compute_units: int = 200_000_000,
    historical_datasets: RegisteredHistoricalDatasets | None = None,
) -> PlannerJobService:
    return PlannerJobService(
        repository=repository,
        wealth_service=WealthService(_WealthRepository()),
        historical_datasets=(
            historical_datasets or RegisteredHistoricalDatasets()
        ),
        dispatcher=dispatcher,
        execution_policy=PlannerExecutionPolicy(
            maximum_working_bytes=maximum_working_bytes,
            in_memory_path_bytes=maximum_working_bytes,
            maximum_temporary_bytes=maximum_working_bytes,
            batch_size=100,
            maximum_compute_units=maximum_compute_units,
        ),
    )


@pytest.mark.asyncio
async def test_duplicate_submission_reuses_one_durable_job() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(repository, dispatcher)
    submission = PlannerJobSubmission(scenario=_scenario())

    first = await service.submit(submission)
    duplicate = await service.submit(submission)

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.job.id == first.job.id
    assert duplicate.job.request_hash == first.job.request_hash
    assert repository.create_calls == 1
    assert len(dispatcher.reserved) == 1
    assert dispatcher.activated == [first.job.id]


@pytest.mark.asyncio
async def test_resource_rejection_happens_before_persistence_or_dispatch() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(
        repository,
        dispatcher,
        maximum_working_bytes=1,
    )

    with pytest.raises(PlannerJobServiceError) as error:
        await service.submit(PlannerJobSubmission(scenario=_scenario()))

    assert error.value.code is PlannerErrorCode.RESOURCE_LIMIT
    assert repository.create_calls == 0
    assert dispatcher.reserved == []
    assert dispatcher.activated == []


@pytest.mark.asyncio
async def test_historical_jobs_require_registered_dataset_ids() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(repository, dispatcher)

    with pytest.raises(PlannerJobServiceError) as error:
        await service.submit(
            PlannerJobSubmission(
                scenario=_scenario(
                    return_model="historical_bootstrap",
                    historical_block_size=1,
                )
            )
        )

    assert error.value.code is PlannerErrorCode.HISTORICAL_DATASET_REQUIRED
    assert repository.create_calls == 0


@pytest.mark.asyncio
async def test_named_stress_selector_is_persisted_in_worker_payload() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(repository, dispatcher)
    submitted = await service.submit(
        PlannerJobSubmission(
            scenario=_scenario(portfolio_allocation=_allocation()),
            named_stress=NamedStressName.STAGFLATION,
        )
    )

    assert submitted.job.payload.named_stress is NamedStressName.STAGFLATION


@pytest.mark.asyncio
async def test_registered_multi_asset_history_is_snapshotted_for_worker(
    tmp_path,
) -> None:
    history_path = tmp_path / "multi-asset.csv"
    history_path.write_text(
        "year,nominal_return,inflation_rate,us_equity_return,"
        "international_equity_return,bonds_return,cash_return\n"
        "2020,0.10,0.02,0.11,0.12,0.01,0.001\n"
        "2021,-0.10,0.03,-0.21,-0.22,0.02,0.002\n",
        encoding="utf-8",
    )
    datasets = RegisteredHistoricalDatasets.from_paths(
        {"multi-market": history_path}
    )
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(
        repository,
        dispatcher,
        historical_datasets=datasets,
    )

    submitted = await service.submit(
        PlannerJobSubmission(
            scenario=_scenario(
                return_model="historical_bootstrap",
                historical_block_size=1,
                portfolio_allocation=_allocation(),
            ),
            historical_dataset_id="multi-market",
        )
    )

    snapshot = submitted.job.payload.historical_dataset
    assert snapshot is not None
    assert snapshot.asset_returns == (
        (0.11, -0.21),
        (0.12, -0.22),
        (0.01, 0.02),
        (0.001, 0.002),
    )


@pytest.mark.asyncio
async def test_accepted_job_cancellation_releases_worker_admission() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(repository, dispatcher)
    submitted = await service.submit(PlannerJobSubmission(scenario=_scenario()))

    cancelled = await service.cancel(submitted.job.id)

    assert cancelled.state is PlannerJobState.CANCELLED
    assert cancelled.cancellation_requested is True
    assert dispatcher.released == [submitted.job.id]


@pytest.mark.asyncio
async def test_activation_failure_does_not_leave_an_accepted_orphan() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher(fail_activation=True)
    service = _service(repository, dispatcher)

    with pytest.raises(PlannerJobServiceError) as error:
        await service.submit(PlannerJobSubmission(scenario=_scenario()))

    assert error.value.code is PlannerErrorCode.DISPATCH_FAILED
    assert error.value.message == "planner job dispatch failed"
    assert len(repository.jobs) == 1
    persisted = next(iter(repository.jobs.values()))
    assert persisted.state is PlannerJobState.FAILED
    assert persisted.error is not None
    assert persisted.error.code is PlannerErrorCode.DISPATCH_FAILED
    assert persisted.error.message == "planner job dispatch failed"
    assert dispatcher.released == [persisted.id]


@pytest.mark.asyncio
async def test_cancellation_after_durable_create_does_not_orphan_job() -> None:
    repository = _JobRepository(cancel_after_create=True)
    dispatcher = _Dispatcher()
    service = _service(repository, dispatcher)

    with pytest.raises(asyncio.CancelledError):
        await service.submit(PlannerJobSubmission(scenario=_scenario()))

    assert len(repository.jobs) == 1
    persisted = next(iter(repository.jobs.values()))
    assert persisted.state is PlannerJobState.FAILED
    assert persisted.error is not None
    assert persisted.error.code is PlannerErrorCode.DISPATCH_FAILED
    assert dispatcher.released == [persisted.id]


@pytest.mark.asyncio
async def test_failed_and_cancelled_submissions_can_create_new_attempts() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(repository, dispatcher)
    submission = PlannerJobSubmission(scenario=_scenario())

    first = await service.submit(submission)
    await repository.mark_failed(
        first.job.id,
        PlannerJobPublicError(
            code=PlannerErrorCode.EXECUTION_FAILED,
            message="planner execution failed",
        ),
    )
    after_failure = await service.submit(submission)
    cancelled = await service.cancel(after_failure.job.id)
    after_cancellation = await service.submit(submission)

    assert after_failure.duplicate is False
    assert after_failure.job.id != first.job.id
    assert cancelled.state is PlannerJobState.CANCELLED
    assert after_cancellation.duplicate is False
    assert after_cancellation.job.id not in {
        first.job.id,
        after_failure.job.id,
    }
    assert repository.create_calls == 3


@pytest.mark.asyncio
async def test_engine_identity_participates_in_request_hash() -> None:
    repository = _JobRepository()
    service = _service(repository, _Dispatcher())
    submitted = await service.submit(PlannerJobSubmission(scenario=_scenario()))
    payload = submitted.job.payload
    assert payload.engine_identity is not None
    changed_engine = payload.engine_identity.model_copy(
        update={"simulation_engine_version": "wealth_simulation_v_next"}
    )
    changed_payload = payload.model_copy(update={"engine_identity": changed_engine})

    assert canonical_job_request_hash(changed_payload) != (canonical_job_request_hash(payload))


@pytest.mark.asyncio
async def test_income_stream_compute_preflight_rejects_before_persistence() -> None:
    repository = _JobRepository()
    dispatcher = _Dispatcher()
    service = _service(
        repository,
        dispatcher,
        maximum_compute_units=10_000,
    )
    income_streams = [
        {
            "name": f"stream-{index}",
            "start_age": 60,
            "annual_amount": 1_000,
        }
        for index in range(4)
    ]

    with pytest.raises(PlannerJobServiceError) as error:
        await service.submit(
            PlannerJobSubmission(scenario=_scenario(income_streams=income_streams))
        )

    assert error.value.code is PlannerErrorCode.RESOURCE_LIMIT
    assert repository.create_calls == 0
    assert dispatcher.reserved == []


def test_progressive_tax_compute_preflight_counts_bounded_search_work() -> None:
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=16 * 1024 * 1024,
        in_memory_path_bytes=16 * 1024 * 1024,
        maximum_temporary_bytes=16 * 1024 * 1024,
        batch_size=100,
        maximum_compute_units=100_000,
    )
    scenario = _scenario(
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 125_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1986,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred"],
            "retirement_surplus_destination": "tax_deferred",
        },
    )

    with pytest.raises(ResourceLimitError, match="compute work"):
        policy.required_working_bytes(
            scenario,
            paired_historical_inflation=False,
        )


def test_progressive_tax_comparison_preflight_aggregates_solver_work() -> None:
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=16 * 1024 * 1024,
        in_memory_path_bytes=16 * 1024 * 1024,
        maximum_temporary_bytes=16 * 1024 * 1024,
        batch_size=100,
        maximum_compute_units=1_000_000,
    )
    scenario = _scenario(
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 125_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1986,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred"],
            "retirement_surplus_destination": "tax_deferred",
        },
    )

    assert (
        policy.required_working_bytes(
            scenario,
            paired_historical_inflation=False,
        )
        > 0
    )
    with pytest.raises(ResourceLimitError, match="comparison compute work"):
        policy.required_comparison_working_bytes(
            [scenario, scenario.model_copy(update={"name": "alternative"})],
            paired_historical_inflation=False,
        )

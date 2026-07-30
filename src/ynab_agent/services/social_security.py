"""Execution port for bounded Social Security strategy optimization."""

from __future__ import annotations

from typing import Protocol

from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.social_security_optimizer import (
    SocialSecurityOptimizationResult,
)


class SocialSecurityOptimizationExecutor(Protocol):
    """Admit and execute one CPU-bound optimization outside the event loop."""

    async def execute(
        self,
        scenario: WealthScenario,
        starting_portfolio: float,
        valuation_provenance: ValuationProvenance,
    ) -> SocialSecurityOptimizationResult: ...

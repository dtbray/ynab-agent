"""Optional interactive reports for wealth simulations."""

from __future__ import annotations

from pathlib import Path

from ynab_agent.planning.simulation import SimulationResult


def write_simulation_html(result: SimulationResult, output_path: Path) -> Path:
    """Write a self-contained percentile fan chart to an HTML file."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised without the reports extra
        raise RuntimeError(
            'HTML reporting requires the reports extra: pip install "ynab-agent[reports]"'
        ) from exc

    ages = [int(point["age"]) for point in result.annual_balance_real]
    p10 = [float(point["p10"]) for point in result.annual_balance_real]
    p50 = [float(point["p50"]) for point in result.annual_balance_real]
    p90 = [float(point["p90"]) for point in result.annual_balance_real]
    retirement_age = int(result.assumptions["retirement_age"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ages,
            y=p90,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            name="P90",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ages,
            y=p10,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(64, 114, 255, 0.20)",
            name="P10–P90",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ages,
            y=p50,
            mode="lines",
            line={"color": "#3157d5", "width": 3},
            name="Median",
        )
    )
    figure.add_vline(
        x=retirement_age,
        line_dash="dash",
        annotation_text="Retirement",
        annotation_position="top left",
    )
    recovery_summary = (
        f"{result.recovery_probability:.1%} recovery after failure"
        if result.recovery_probability is not None
        else "no failed trials to recover"
    )
    goal_summary = " · ".join(
        (
            f"{goal.name}: {goal.attainment_probability:.1%}"
            if goal.attainment_probability is not None
            else f"{goal.name}: not evaluated"
        )
        for goal in result.goal_outcomes
    )
    figure.add_annotation(
        x=0,
        y=-0.20,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        text=f"<b>Goal attainment:</b> {goal_summary}",
    )
    figure.update_layout(
        title=(
            f"{result.scenario} — {result.success_rate:.1%} success"
            f"<br><sup>{result.funded_spending_ratio['p50']:.1%} median funded spending; "
            f"${result.cumulative_shortfall_real['p50']:,.0f} median cumulative shortfall; "
            f"{recovery_summary}"
            "</sup>"
        ),
        xaxis_title="Age",
        yaxis_title="Portfolio balance in today's dollars",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        template="plotly_white",
        margin={"b": 100},
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path

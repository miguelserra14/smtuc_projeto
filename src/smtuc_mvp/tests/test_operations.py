from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from smtuc_mvp.gtfs import load_gtfs

from smtuc_mvp.operations import (
    WALK_SPEED_M_MIN,
    commute_options_for_datetime,
    compare_nearest_network,
    find_direct_options,
    line_overlap_top,
    nearest_stop_for_dataset,
    next_monday,
    suggest_current_commute_options,
    suggest_random_commute_options,
)

WALK_5_MIN_M = WALK_SPEED_M_MIN * 5
casa = [40.207883, -8.398107]
trabalho = [40.186724, -8.416078]  # DEI


def _dataset_dir(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / dataset


def _require_dataset(dataset: str) -> Path:
    d = _dataset_dir(dataset)
    required = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    if not d.exists() or any(not (d / f).exists() for f in required):
        pytest.skip(f"Dataset GTFS inválido/incompleto: {d}")
    return d


@pytest.mark.integration
def test_scenario_nearest_home_work_with_prints() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    home_smtuc = nearest_stop_for_dataset("smtuc", casa[0], casa[1])
    home_metro = nearest_stop_for_dataset("metrobus", casa[0], casa[1])
    work_smtuc = nearest_stop_for_dataset("smtuc", trabalho[0], trabalho[1])
    work_metro = nearest_stop_for_dataset("metrobus", trabalho[0], trabalho[1])

    print("\n=== Paragens mais próximas (casa e trabalho) ===")
    print(f"Casa SMTUC   -> {home_smtuc.stop_name} ({home_smtuc.stop_id}) | {home_smtuc.distance_m:.1f} m")
    print(f"Casa Metrobus-> {home_metro.stop_name} ({home_metro.stop_id}) | {home_metro.distance_m:.1f} m")
    print(f"Trab SMTUC   -> {work_smtuc.stop_name} ({work_smtuc.stop_id}) | {work_smtuc.distance_m:.1f} m")
    print(f"Trab Metrobus-> {work_metro.stop_name} ({work_metro.stop_id}) | {work_metro.distance_m:.1f} m")

    assert min(home_smtuc.distance_m, home_metro.distance_m) >= 0
    assert min(work_smtuc.distance_m, work_metro.distance_m) >= 0


@pytest.mark.integration
def test_scenario_best_route_random_monday_now_with_prints() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    rnd = suggest_random_commute_options(casa[0], casa[1], trabalho[0], trabalho[1], limit=1, tries=10)

    monday = next_monday(date.today())
    monday_df = commute_options_for_datetime(
        home_lat=casa[0],
        home_lon=casa[1],
        work_lat=trabalho[0],
        work_lon=trabalho[1],
        day_str=monday.strftime("%Y-%m-%d"),
        time_str="08:40:00",
    )

    now_df = suggest_current_commute_options(casa[0], casa[1], trabalho[0], trabalho[1], limit=1)

    print("\n=== Melhor percurso por cenário ===")
    if rnd.empty:
        print("Random: sem opções diretas encontradas")
    else:
        print("Random:")
        print(rnd.head(1).to_string(index=False))

    if monday_df.empty:
        print(f"Segunda 08:40 ({monday}): sem opções diretas")
    else:
        print(f"Segunda 08:40 ({monday}):")
        print(monday_df.head(1).to_string(index=False))

    if now_df.empty:
        print("Agora: sem opções diretas")
    else:
        print("Agora:")
        print(now_df.head(1).to_string(index=False))

    assert isinstance(rnd, pd.DataFrame)
    assert isinstance(monday_df, pd.DataFrame)
    assert isinstance(now_df, pd.DataFrame)


@pytest.mark.integration
def test_scenario_top5_smtuc_overlap_with_metrobus_with_prints() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    top5 = line_overlap_top()

    print("\n=== Top 5 linhas SMTUC com maior overlap com Metrobus (<=5 min a pé) ===")
    if top5.empty:
        print("Sem dados suficientes para calcular overlap.")
    else:
        display = top5.rename(
            columns={
                "line": "Linha",
                "avg_freq_min": "Freq(min)",
                "overlap_extension_m": "Overlap(m)",
                "line_extension_m": "Extensão(m)",
                "overlap_pct": "Overlap(%)",
                "overlap_stops": "Paragens Overlapped",
                "total_stops": "Paragens Totais",
                "directions_considered": "Sentidos.",
            }
        )
        if "Freq(min)" in display.columns:
            display["Freq(min)"] = display["Freq(min)"].map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
            )
        ordered_cols = [
            c
            for c in [
                "Linha",
                "Overlap(m)",
                "Extensão(m)",
                "Overlap(%)",
                "Paragens Overlapped",
                "Paragens Totais",
                "Sentidos.",
                "Freq(min)",
            ]
            if c in display.columns
        ]
        print(display[ordered_cols].to_string(index=False))

    assert isinstance(top5, pd.DataFrame)
    assert len(top5) <= 5


@pytest.mark.integration
def test_scenario_compare_nearest_network_contract() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    smtuc, metrobus, winner = compare_nearest_network(casa[0], casa[1])
    assert winner in {"smtuc", "metrobus"}
    assert smtuc.distance_m >= 0
    assert metrobus.distance_m >= 0

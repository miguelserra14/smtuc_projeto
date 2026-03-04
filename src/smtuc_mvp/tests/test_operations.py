from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


from smtuc_mvp.operations.operations import (
    WALK_SPEED_M_MIN,
    commute_options_for_datetime,
    compare_nearest_network,
    find_direct_options,
    nearest_stop_for_dataset,
    next_monday,
    suggest_current_commute_options,
    suggest_random_commute_options,
)
from smtuc_mvp.operations.operations_overlap import line_overlap_top
from smtuc_mvp.operations.operations_overlap import line_low_overlap_near_stadium_top

WALK_5_MIN_M = WALK_SPEED_M_MIN * 5
casa = [40.207883, -8.398107]
trabalho = [40.186724, -8.416078]  # DEI
estadio=[40.203809, -8.407904]


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
def test_nearest_home_work() -> None:
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
def test_best_route_scenarios() -> None:
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
def test_top5_overlap() -> None:
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
                "avg_freq_min": "Freq",
                "overlap_extension_m": "Ovlp(m)",
                "line_extension_m": "Ext(m)",
                "overlap_pct": "Ovlp(%)",
                "overlap_stops": "Parag.Ovlp",
                "total_stops": "Parag.Tot",
                "directions_considered": "Sent",
            }
        )
        if "Freq" in display.columns:
            display["Freq"] = display["Freq"].map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
            )
        ordered_cols = [
            c
            for c in [
                "Linha",
                "Ovlp(m)",
                "Ext(m)",
                "Ovlp(%)",
                "Parag.Ovlp",
                "Parag.Tot",
                "Sent",
                "Freq",
            ]
            if c in display.columns
        ]
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            1000,
            "display.expand_frame_repr",
            False,
            "display.colheader_justify",
            "left",
        ):
            print(display[ordered_cols].to_string(index=False, line_width=1000))

    assert isinstance(top5, pd.DataFrame)
    assert len(top5) <= 5


@pytest.mark.integration
def test_bottom5_overlap() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    all_lines = line_overlap_top(top_n=10000)
    bottom5 = all_lines.sort_values(["overlap_pct", "overlap_extension_m"], ascending=[True, True]).head(5)

    print("\n=== Bottom 5 linhas SMTUC com menor overlap com Metrobus (<=5 min a pé) ===")
    if bottom5.empty:
        print("Sem dados suficientes para calcular overlap.")
    else:
        display = bottom5.rename(
            columns={
                "line": "Linha",
                "avg_freq_min": "Freq",
                "overlap_extension_m": "Ovlp(m)",
                "line_extension_m": "Ext(m)",
                "overlap_pct": "Ovlp(%)",
                "overlap_stops": "Parag.Ovlp",
                "total_stops": "Parag.Tot",
                "directions_considered": "Sent",
            }
        )
        if "Freq" in display.columns:
            display["Freq"] = display["Freq"].map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
            )
        ordered_cols = [
            c
            for c in [
                "Linha",
                "Ovlp(m)",
                "Ext(m)",
                "Ovlp(%)",
                "Parag.Ovlp",
                "Parag.Tot",
                "Sent",
                "Freq",
            ]
            if c in display.columns
        ]
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            1000,
            "display.expand_frame_repr",
            False,
            "display.colheader_justify",
            "left",
        ):
            print(display[ordered_cols].to_string(index=False, line_width=1000))

    assert isinstance(bottom5, pd.DataFrame)
    assert len(bottom5) <= 5


@pytest.mark.integration
def test_bottom5_overlap_near_stadium() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    out = line_low_overlap_near_stadium_top(
        stadium_lat=estadio[0],
        stadium_lon=estadio[1],
        radius_m=2000.0,
        min_radius_extension_pct=50.0,
        top_n=5,
    )

    print("\n=== Bottom 5 linhas com menor overlap e >=50% da extensão a <=2km do estádio ===")
    if out.empty:
        print("Sem dados suficientes para o critério de estádio.")
    else:
        display = out.rename(
            columns={
                "line": "Linha",
                "avg_freq_min": "Freq",
                "overlap_extension_m": "Ovlp(m)",
                "line_extension_m": "Ext(m)",
                "overlap_pct": "Ovlp(%)",
                "overlap_stops": "Parag.Ovlp",
                "total_stops": "Parag.Tot",
                "directions_considered": "Sent",
                "radius_extension_m": "Ext2km(m)",
                "radius_extension_pct": "%Ext2km",
            }
        )
        if "Freq" in display.columns:
            display["Freq"] = display["Freq"].map(lambda x: "-" if pd.isna(x) else f"{float(x):.1f}")
        ordered_cols = [
            c
            for c in [
                "Linha",
                "Ovlp(m)",
                "Ext(m)",
                "Ovlp(%)",
                "Ext2km(m)",
                "%Ext2km",
                "Parag.Ovlp",
                "Parag.Tot",
                "Sent",
                "Freq",
            ]
            if c in display.columns
        ]
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            1000,
            "display.expand_frame_repr",
            False,
            "display.colheader_justify",
            "left",
        ):
            print(display[ordered_cols].to_string(index=False, line_width=1000))

    assert isinstance(out, pd.DataFrame)
    assert len(out) <= 5
    if not out.empty:
        assert (out["radius_extension_pct"] >= 50.0).all()


@pytest.mark.integration
def test_compare_nearest_network() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    smtuc, metrobus, winner = compare_nearest_network(casa[0], casa[1])
    assert winner in {"smtuc", "metrobus"}
    assert smtuc.distance_m >= 0
    assert metrobus.distance_m >= 0

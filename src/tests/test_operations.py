from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from config import (
    HOME_COORD,
    OVERLAP_SCAN_TOP_N,
    OVERLAP_TABLE_TOP_N,
    STADIUM_COORD,
    STADIUM_MIN_EXTENSION_PCT,
    STADIUM_RADIUS_M,
    OUTPUTS_OVERLAP_DIR,
    WORK_COORD,
)

from operations.operations import (
    commute_options_for_datetime,
    compare_nearest_network,
    nearest_stop_for_dataset,
    next_monday,
    suggest_current_commute_options,
    suggest_random_commute_options,
)
from operations.operations_overlap import (
    compute_bgri_reachability_now,
    compute_temporal_overlaps_for_db,
    line_low_overlap_near_stadium_top,
    line_overlap_top,
    temporal_overlap_events_for_metrics,
)
from population.data_processing import compute_underserved_zones
from population.visualizations import create_overlap_reachability_map, _write_folium_html

OVERLAP_RENAME = {
    "line": "Linha",
    "avg_freq_min": "Freq",
    "overlap_extension_m": "Ovlp(m)",
    "line_extension_m": "Ext(m)",
    "overlap_pct": "Ovlp(%)",
    "temporal_overlaps_pct": "Temp.Ovlp(%)",
    "overlap_stops": "Parag.Ovlp",
    "total_stops": "Parag.Tot",
    "directions_considered": "Sent",
}


def _print_table(df: pd.DataFrame) -> None:
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
        print(df.to_string(index=False, line_width=1000))


def _format_overlap_table(df: pd.DataFrame, include_radius: bool = False) -> pd.DataFrame:
    display = df.rename(columns=OVERLAP_RENAME)
    if include_radius:
        display = display.rename(
            columns={
                "radius_extension_m": "Ext2km(m)",
                "radius_extension_pct": "%Ext2km",
            }
        )
    if "Freq" in display.columns:
        display["Freq"] = display["Freq"].map(lambda x: "-" if pd.isna(x) else f"{float(x):.1f}")
    ordered = ["Linha", "Ovlp(m)", "Ext(m)", "Ovlp(%)"]
    if include_radius:
        ordered.extend(["Ext2km(m)", "%Ext2km"])
    ordered.extend(["Parag.Ovlp", "Parag.Tot", "Sent", "Freq"])
    ordered_cols = [c for c in ordered if c in display.columns]
    return display[ordered_cols]


def _dataset_dir(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[2]
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

    home_smtuc = nearest_stop_for_dataset("smtuc", HOME_COORD[0], HOME_COORD[1])
    home_metro = nearest_stop_for_dataset("metrobus", HOME_COORD[0], HOME_COORD[1])
    work_smtuc = nearest_stop_for_dataset("smtuc", WORK_COORD[0], WORK_COORD[1])
    work_metro = nearest_stop_for_dataset("metrobus", WORK_COORD[0], WORK_COORD[1])

    print("\n=== Paragens mais próximas (casa e trabalho) ===")
    print(f"Casa SMTUC   -> {home_smtuc.stop_name} ({home_smtuc.stop_id}) | {home_smtuc.distance_m:.1f} m")
    print(f"Casa Metrobus-> {home_metro.stop_name} ({home_metro.stop_id}) | {home_metro.distance_m:.1f} m")
    print(f"Trab SMTUC   -> {work_smtuc.stop_name} ({work_smtuc.stop_id}) | {work_smtuc.distance_m:.1f} m")
    print(f"Trab Metrobus-> {work_metro.stop_name} ({work_metro.stop_id}) | {work_metro.distance_m:.1f} m")

    assert min(home_smtuc.distance_m, home_metro.distance_m) >= 0
    assert min(work_smtuc.distance_m, work_metro.distance_m) >= 0

"""@pytest.mark.integration
def test_best_route_scenarios() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    rnd = suggest_random_commute_options(HOME_COORD[0], HOME_COORD[1], WORK_COORD[0], WORK_COORD[1], limit=1, tries=10)

    monday = next_monday(date.today())
    monday_df = commute_options_for_datetime(
        home_lat=HOME_COORD[0],
        home_lon=HOME_COORD[1],
        work_lat=WORK_COORD[0],
        work_lon=WORK_COORD[1],
        day_str=monday.strftime("%Y-%m-%d"),
        time_str="08:40:00",
    )

    now_df = suggest_current_commute_options(HOME_COORD[0], HOME_COORD[1], WORK_COORD[0], WORK_COORD[1], limit=1)

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

"""
@pytest.mark.integration
def test_top5_overlap() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    top5 = line_overlap_top(top_n=OVERLAP_TABLE_TOP_N)

    print("\n=== Top 5 linhas SMTUC com maior overlap com Metrobus (<=5 min a pé) ===")
    if top5.empty:
        print("Sem dados suficientes para calcular overlap.")
    else:
        _print_table(_format_overlap_table(top5))

    assert isinstance(top5, pd.DataFrame)
    assert len(top5) <= OVERLAP_TABLE_TOP_N


@pytest.mark.integration
def test_bottom5_overlap() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    all_lines = line_overlap_top(top_n=OVERLAP_SCAN_TOP_N)
    bottom5 = all_lines.sort_values(["overlap_pct", "overlap_extension_m"], ascending=[True, True]).head(OVERLAP_TABLE_TOP_N)

    print("\n=== Bottom 5 linhas SMTUC com menor overlap com Metrobus (<=5 min a pé) ===")
    if bottom5.empty:
        print("Sem dados suficientes para calcular overlap.")
    else:
        _print_table(_format_overlap_table(bottom5))

    assert isinstance(bottom5, pd.DataFrame)
    assert len(bottom5) <= OVERLAP_TABLE_TOP_N


@pytest.mark.integration
def test_bottom5_overlap_near_stadium() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    out = line_low_overlap_near_stadium_top(
        stadium_lat=STADIUM_COORD[0],
        stadium_lon=STADIUM_COORD[1],
        radius_m=STADIUM_RADIUS_M,
        min_radius_extension_pct=STADIUM_MIN_EXTENSION_PCT,
        top_n=OVERLAP_TABLE_TOP_N,
    )

    print("\n=== Bottom 5 linhas com menor overlap e >=50% da extensão a <=2km do estádio ===")
    if out.empty:
        print("Sem dados suficientes para o critério de estádio.")
    else:
        _print_table(_format_overlap_table(out, include_radius=True))

    assert isinstance(out, pd.DataFrame)
    assert len(out) <= OVERLAP_TABLE_TOP_N
    if not out.empty:
        assert (out["radius_extension_pct"] >= 50.0).all()


@pytest.mark.integration
def test_compare_nearest_network() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    smtuc, metrobus, winner = compare_nearest_network(HOME_COORD[0], HOME_COORD[1])
    assert winner in {"smtuc", "metrobus"}
    assert smtuc.distance_m >= 0
    assert metrobus.distance_m >= 0


@pytest.mark.integration
def test_overlap_reachability_map_now() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    now = datetime.now()
    day_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    merged = compute_underserved_zones(
        day_str=day_str,
    )

    reach_gdf = compute_bgri_reachability_now(
        merged_bgri=merged,
        origin_lat=STADIUM_COORD[0],
        origin_lon=STADIUM_COORD[1],
        day_str=day_str,
        time_str=time_str,
    )

    fig_map = create_overlap_reachability_map(
        reach_gdf=reach_gdf,
        origin_lat=STADIUM_COORD[0],
        origin_lon=STADIUM_COORD[1],
        day_str=day_str,
        time_str=time_str,
    )

    out_dir = Path(__file__).resolve().parents[2] / OUTPUTS_OVERLAP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_html = out_dir / "overlap_reachability_now.html"
    _write_folium_html(fig_map, output_html)

    print(f"Mapa overlap reachability gerado: {output_html}")

    assert output_html.exists()
    assert "reach_min" in reach_gdf.columns
    assert "reach_bin" in reach_gdf.columns
    assert "reach_mode" in reach_gdf.columns

@pytest.mark.integration
def test_temporal_overlaps() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    metrics = line_overlap_top(top_n=OVERLAP_SCAN_TOP_N)

    if metrics.empty:
        pytest.skip("Sem dados de overlap para calcular temporal overlaps.")

    sample = metrics[metrics["overlap_pct"] > 0].head(5).copy()
    if sample.empty:
        pytest.skip("Sem linhas SMTUC com overlap espacial para teste temporal.")

    sample = compute_temporal_overlaps_for_db(sample)
    events = temporal_overlap_events_for_metrics(sample)

    assert "temporal_spatial_candidates_count" in sample.columns
    assert "temporal_overlaps_count" in sample.columns
    assert "temporal_overlaps_pct" in sample.columns

    total_spatial_candidates = int(sample["temporal_spatial_candidates_count"].fillna(0).sum())
    total_temporal_overlaps = int(sample["temporal_overlaps_count"].fillna(0).sum())
    overlap_stations = int(sample["overlap_stops"].fillna(0).sum()) if "overlap_stops" in sample.columns else 0
    overlap_lines = int(sample["line"].nunique())
    temporal_overlap_times = int(len(events))
    temporal_overlap_stations = int(events["smtuc_stop_id"].nunique()) if not events.empty else 0
    temporal_overlap_lines = int(events["line"].nunique()) if not events.empty else 0
    temporal_overlap_pct = (
        (total_temporal_overlaps / total_spatial_candidates) * 100.0 if total_spatial_candidates > 0 else 0.0
    )

    print(
        f"\n\n\nDe todas as vezes que um autocarro dos SMTUC passa numa estação com overlap espacial "
        f"({total_spatial_candidates} vezes, em {overlap_stations} estações e {overlap_lines} linhas), "
        f"há overlap temporal em {temporal_overlap_pct:.2f}% delas ({temporal_overlap_times} vezes, "
        f"em {temporal_overlap_stations} estações e em {temporal_overlap_lines} linhas)."
    )

    if events.empty:
        print("Sem eventos de overlap temporal para apresentar top 5.")
    else:
        top5_lines = (
            events.groupby("line", as_index=False)
            .size()
            .rename(columns={"size": "Ovlp.Temp(vezes)", "line": "Linha"})
            .sort_values(["Ovlp.Temp(vezes)", "Linha"], ascending=[False, True])
            .head(5)
        )

        events_with_hour = events.copy()
        events_with_hour["Hora"] = events_with_hour["hour"].astype(int)
        top5_hours = (
            events_with_hour.groupby("Hora", as_index=False)
            .size()
            .rename(columns={"size": "Ovlp.Temp(vezes)"})
            .sort_values(["Ovlp.Temp(vezes)", "Hora"], ascending=[False, True])
            .head(5)
        )
        top5_hours["Hora"] = top5_hours["Hora"].map(lambda h: f"{int(h):02d}:00-{int(h):02d}:59")

        print("\n=== Top 5 linhas com mais overlap temporal ===")
        _print_table(top5_lines)

        print("\n=== Top 5 horários com mais overlap temporal ===")
        _print_table(top5_hours)

    assert total_spatial_candidates >= 0
    assert total_temporal_overlaps >= 0
    assert total_temporal_overlaps <= total_spatial_candidates
    assert temporal_overlap_times == total_temporal_overlaps
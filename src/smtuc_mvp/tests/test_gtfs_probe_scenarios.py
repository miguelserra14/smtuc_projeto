from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from smtuc_mvp.gtfs import load_gtfs
from smtuc_mvp.gtfs_probe import (
    casa,
    compare_nearest_network,
    find_direct_options,
    nearest_stop_for_dataset,
    suggest_current_commute_options,
    suggest_random_commute_options,
    trabalho,
)

WALK_SPEED_M_MIN = 80.0
WALK_5_MIN_M = WALK_SPEED_M_MIN * 5


def _dataset_dir(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / dataset


def _require_dataset(dataset: str) -> Path:
    d = _dataset_dir(dataset)
    required = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    if not d.exists() or any(not (d / f).exists() for f in required):
        pytest.skip(f"Dataset GTFS inválido/incompleto: {d}")
    return d


def _haversine_m(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    import math

    r = 6371000.0
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)

    lat2r = lat2.astype(float).apply(math.radians)
    lon2r = lon2.astype(float).apply(math.radians)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = (dlat / 2).apply(math.sin) ** 2 + pd.Series(
        [math.cos(lat1r) * math.cos(v) for v in lat2r], dtype="float64"
    ) * (dlon / 2).apply(math.sin) ** 2
    c = 2 * a.apply(lambda x: math.atan2(math.sqrt(x), math.sqrt(1 - x)))
    return r * c


def _next_monday(from_day: date) -> date:
    delta = (0 - from_day.weekday()) % 7
    return from_day + timedelta(days=delta)


def _commute_options_for_datetime(day_str: str, time_str: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for dataset in ("smtuc", "metrobus"):
        try:
            near_home = nearest_stop_for_dataset(dataset, casa[0], casa[1])
            near_work = nearest_stop_for_dataset(dataset, trabalho[0], trabalho[1])

            df = find_direct_options(
                dataset=dataset,
                origin_ref=near_home.stop_id,
                dest_ref=near_work.stop_id,
                day_str=day_str,
                time_str=time_str,
            )

            if df.empty:
                continue

            df = df.copy()
            df["dataset"] = dataset
            df["day"] = day_str
            df["time"] = time_str
            df["walk_home_m"] = round(near_home.distance_m, 1)
            df["walk_work_m"] = round(near_work.distance_m, 1)
            df["walk_home_min"] = (df["walk_home_m"] / WALK_SPEED_M_MIN).round(1)
            df["walk_work_min"] = (df["walk_work_m"] / WALK_SPEED_M_MIN).round(1)
            df["total_min_est"] = (df["duration_min"] + df["walk_home_min"] + df["walk_work_min"]).round(1)
            rows.append(df)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["total_min_est", "duration_min", "dataset"])
        .drop_duplicates(subset=["dataset", "route_short_name", "origin_dep_time", "dest_arr_time"], keep="first")
        .reset_index(drop=True)
    )


def _representative_route_stops_for_subset(gtfs, route_trips: pd.DataFrame) -> pd.DataFrame:
    if route_trips.empty:
        return pd.DataFrame()

    st = gtfs.stop_times[gtfs.stop_times["trip_id"].astype(str).isin(route_trips["trip_id"].astype(str))].copy()
    if st.empty:
        return pd.DataFrame()

    trip_sizes = st.groupby("trip_id", as_index=False).agg(n_stops=("stop_id", "count"))
    best_trip = str(trip_sizes.sort_values("n_stops", ascending=False).iloc[0]["trip_id"])

    best = st[st["trip_id"].astype(str) == best_trip].sort_values("stop_sequence")
    best = best.merge(
        gtfs.stops[["stop_id", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    )
    return best[["stop_id", "stop_sequence", "stop_lat", "stop_lon"]].drop_duplicates(subset=["stop_sequence"])


def _route_direction_summaries(gtfs_smtuc, route_id: str, metro_stops: pd.DataFrame) -> list[dict]:
    route_trips = gtfs_smtuc.trips[gtfs_smtuc.trips["route_id"].astype(str) == str(route_id)].copy()
    if route_trips.empty:
        return []

    # Se existir direction_id, calcula por sentido; caso contrário usa a rota inteira.
    if "direction_id" in route_trips.columns:
        direction_values = route_trips["direction_id"].dropna().astype(str).unique().tolist()
        if not direction_values:
            direction_values = ["na"]
    else:
        direction_values = ["na"]

    summaries: list[dict] = []

    for direction in direction_values:
        if direction == "na" and "direction_id" not in route_trips.columns:
            subset = route_trips
        elif direction == "na":
            subset = route_trips
        else:
            subset = route_trips[route_trips["direction_id"].astype(str) == direction]

        route_stops = _representative_route_stops_for_subset(gtfs_smtuc, subset)
        route_stops = route_stops.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)
        if len(route_stops) < 2:
            continue

        min_dist = []
        for _, row in route_stops.iterrows():
            d = _haversine_m(float(row["stop_lat"]), float(row["stop_lon"]), metro_stops["stop_lat"], metro_stops["stop_lon"])
            min_dist.append(float(d.min()))

        route_stops["is_overlap_stop"] = pd.Series(min_dist) <= WALK_5_MIN_M

        total_ext_m = 0.0
        overlap_ext_m = 0.0
        for i in range(len(route_stops) - 1):
            a = route_stops.iloc[i]
            b = route_stops.iloc[i + 1]
            seg = float(
                _haversine_m(
                    float(a["stop_lat"]),
                    float(a["stop_lon"]),
                    pd.Series([b["stop_lat"]]),
                    pd.Series([b["stop_lon"]]),
                ).iloc[0]
            )
            total_ext_m += seg
            if bool(a["is_overlap_stop"]) and bool(b["is_overlap_stop"]):
                overlap_ext_m += seg

        if total_ext_m <= 0:
            continue

        summaries.append(
            {
                "direction": direction,
                "total_ext_m": total_ext_m,
                "overlap_ext_m": overlap_ext_m,
                "overlap_stops": int(route_stops["is_overlap_stop"].sum()),
                "total_stops": int(len(route_stops)),
            }
        )

    return summaries


def _line_avg_frequency_min(gtfs_smtuc, route_ids: list[str]) -> float | None:
    if not route_ids:
        return None

    trips = gtfs_smtuc.trips[gtfs_smtuc.trips["route_id"].astype(str).isin([str(r) for r in route_ids])]
    if trips.empty:
        return None

    stop_times = gtfs_smtuc.stop_times[
        gtfs_smtuc.stop_times["trip_id"].astype(str).isin(trips["trip_id"].astype(str))
    ]
    if stop_times.empty or "departure_seconds" not in stop_times.columns:
        return None

    first_dep = (
        stop_times.groupby("trip_id", as_index=False)
        .agg(first_dep_s=("departure_seconds", "min"))
        .dropna(subset=["first_dep_s"])
    )
    if len(first_dep) < 2:
        return None

    departures = sorted(first_dep["first_dep_s"].astype(float).unique().tolist())
    if len(departures) < 2:
        return None

    headways_s = [b - a for a, b in zip(departures[:-1], departures[1:]) if b > a]
    if not headways_s:
        return None

    return round((sum(headways_s) / len(headways_s)) / 60.0, 1)


def _line_overlap_top5() -> pd.DataFrame:
    gtfs_smtuc = load_gtfs(dataset="smtuc")
    gtfs_metro = load_gtfs(dataset="metrobus")

    metro_stops = gtfs_metro.stops[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    if metro_stops.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []

    routes_map_df = gtfs_smtuc.routes.copy()
    if routes_map_df.empty or "route_id" not in routes_map_df.columns:
        return pd.DataFrame()

    name_col = "route_short_name" if "route_short_name" in routes_map_df.columns else "route_id"
    routes_map_df = routes_map_df[["route_id", name_col]].copy()
    routes_map_df["route_id"] = routes_map_df["route_id"].astype(str)
    routes_map_df["line"] = routes_map_df[name_col].astype(str)

    trips_enriched = gtfs_smtuc.trips.copy()
    trips_enriched["route_id"] = trips_enriched["route_id"].astype(str)
    trips_enriched = trips_enriched.merge(routes_map_df[["route_id", "line"]], on="route_id", how="left")
    trips_enriched["line"] = trips_enriched["line"].fillna(trips_enriched["route_id"])

    line_to_route_ids = (
        trips_enriched[["line", "route_id"]]
        .dropna()
        .drop_duplicates()
        .groupby("line")["route_id"]
        .apply(list)
        .to_dict()
    )

    for line, route_ids in line_to_route_ids.items():
        line_summaries: list[dict] = []
        for route_id in route_ids:
            line_summaries.extend(_route_direction_summaries(gtfs_smtuc, route_id, metro_stops))

        if not line_summaries:
            continue

        total_ext_m = sum(s["total_ext_m"] for s in line_summaries)
        overlap_ext_m = sum(s["overlap_ext_m"] for s in line_summaries)
        overlap_stops = sum(s["overlap_stops"] for s in line_summaries)
        total_stops = sum(s["total_stops"] for s in line_summaries)
        directions_considered = len(line_summaries)
        avg_freq_min = _line_avg_frequency_min(gtfs_smtuc, route_ids)

        if total_ext_m <= 0:
            continue

        overlap_pct = (overlap_ext_m / total_ext_m) * 100.0

        out_rows.append(
            {
                "line": str(line),
                "avg_freq_min": avg_freq_min,
                "overlap_extension_m": round(overlap_ext_m, 1),
                "line_extension_m": round(total_ext_m, 1),
                "overlap_pct": round(overlap_pct, 2),
                "overlap_stops": int(overlap_stops),
                "total_stops": int(total_stops),
                "directions_considered": directions_considered,
            }
        )

    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows)
    df = df[pd.to_numeric(df["line"], errors="coerce") < 100]
    return df.sort_values(["overlap_pct", "overlap_extension_m"], ascending=False).head(5).reset_index(drop=True)


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

    monday = _next_monday(date.today())
    monday_df = _commute_options_for_datetime(day_str=monday.strftime("%Y-%m-%d"), time_str="08:40:00")

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

    top5 = _line_overlap_top5()

    print("\n=== Top 5 linhas SMTUC com maior overlap com Metrobus (<=5 min a pé) ===")
    if top5.empty:
        print("Sem dados suficientes para calcular overlap.")
    else:
        display = top5.rename(
            columns={
                "line": "Linha",
                "avg_freq_min": "Freq média (min)",
                "overlap_extension_m": "Overlap (m)",
                "line_extension_m": "Extensão linha (m)",
                "overlap_pct": "Overlap (%)",
                "overlap_stops": "Paragens overlap",
                "total_stops": "Paragens total",
                "directions_considered": "Sentidos usados",
            }
        )
        if "Freq média (min)" in display.columns:
            display["Freq média (min)"] = display["Freq média (min)"].map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
            )
        ordered_cols = [
            c
            for c in [
                "Linha",
                "Overlap (m)",
                "Extensão linha (m)",
                "Overlap (%)",
                "Paragens overlap",
                "Paragens total",
                "Sentidos usados",
                "Freq média (min)",
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

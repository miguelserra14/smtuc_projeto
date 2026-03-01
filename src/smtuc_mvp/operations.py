from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import pandas as pd

from smtuc_mvp.gtfs import load_gtfs
from smtuc_mvp.gtfs_probe import (
    NearestStopResult,
    _active_service_ids,
    _haversine_m,
    _parse_day,
    _resolve_stop_id,
    _to_seconds,
)

WALK_SPEED_M_MIN = 80.0  # ~4.8 km/h


def find_direct_options(dataset: str, origin_ref: str, dest_ref: str, day_str: str, time_str: str) -> pd.DataFrame:
    gtfs = load_gtfs(dataset=dataset)
    d = _parse_day(day_str)
    t0 = _to_seconds(time_str)

    origin_id = _resolve_stop_id(gtfs, origin_ref)
    dest_id = _resolve_stop_id(gtfs, dest_ref)

    services = _active_service_ids(gtfs, d)
    trips = gtfs.trips.copy()
    trips["service_id"] = trips["service_id"].astype(str)
    if services:
        trips = trips[trips["service_id"].isin(services)]

    st = gtfs.stop_times.copy()
    left = st.rename(
        columns={
            "stop_id": "origin_stop_id",
            "stop_sequence": "origin_seq",
            "departure_seconds": "origin_dep_s",
            "departure_time": "origin_dep_time",
        }
    )
    right = st.rename(
        columns={
            "stop_id": "dest_stop_id",
            "stop_sequence": "dest_seq",
            "arrival_seconds": "dest_arr_s",
            "arrival_time": "dest_arr_time",
        }
    )

    m = (
        left.merge(
            right[["trip_id", "dest_stop_id", "dest_seq", "dest_arr_s", "dest_arr_time"]],
            on="trip_id",
            how="inner",
        )
        .merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")
    )

    m = m[
        (m["origin_stop_id"].astype(str) == origin_id)
        & (m["dest_stop_id"].astype(str) == dest_id)
        & (m["origin_seq"] < m["dest_seq"])
        & (m["origin_dep_s"] >= t0)
    ].copy()

    if m.empty:
        return m

    stops_small = gtfs.stops[["stop_id", "stop_name"]].copy() if "stop_name" in gtfs.stops.columns else gtfs.stops[["stop_id"]].copy()
    m = m.merge(
        stops_small.rename(columns={"stop_id": "origin_stop_id", "stop_name": "origin_name"}),
        on="origin_stop_id",
        how="left",
    ).merge(
        stops_small.rename(columns={"stop_id": "dest_stop_id", "stop_name": "dest_name"}),
        on="dest_stop_id",
        how="left",
    )

    if "route_short_name" in gtfs.routes.columns:
        m = m.merge(gtfs.routes[["route_id", "route_short_name"]], on="route_id", how="left")

    m["duration_min"] = ((m["dest_arr_s"] - m["origin_dep_s"]) / 60).round(1)

    cols = [
        c for c in [
            "route_id",
            "route_short_name",
            "trip_id",
            "origin_name",
            "dest_name",
            "origin_dep_time",
            "dest_arr_time",
            "duration_min",
        ] if c in m.columns
    ]
    return m[cols].sort_values(["origin_dep_time", "route_id"]).reset_index(drop=True)


def nearest_stop_for_dataset(dataset: str, lat: float, lon: float) -> NearestStopResult:
    gtfs = load_gtfs(dataset=dataset)
    stops = gtfs.stops.copy()

    if stops.empty or not {"stop_id", "stop_lat", "stop_lon"}.issubset(stops.columns):
        raise ValueError(f"Dataset sem paragens válidas: {dataset}")

    dists = _haversine_m(lat, lon, stops["stop_lat"], stops["stop_lon"])
    idx = dists.idxmin()

    stop_id = str(stops.loc[idx, "stop_id"])
    stop_name = str(stops.loc[idx, "stop_name"]) if "stop_name" in stops.columns else stop_id
    return NearestStopResult(dataset=dataset, stop_id=stop_id, stop_name=stop_name, distance_m=float(dists.loc[idx]))


def compare_nearest_network(lat: float, lon: float) -> tuple[NearestStopResult, NearestStopResult, str]:
    smtuc = nearest_stop_for_dataset("smtuc", lat, lon)
    metrobus = nearest_stop_for_dataset("metrobus", lat, lon)
    winner = "smtuc" if smtuc.distance_m <= metrobus.distance_m else "metrobus"
    return smtuc, metrobus, winner


def _collect_commute_options(
    home_lat: float,
    home_lon: float,
    work_lat: float,
    work_lon: float,
    day_str: str,
    time_str: str,
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []

    for dataset in ("smtuc", "metrobus"):
        try:
            near_home = nearest_stop_for_dataset(dataset, home_lat, home_lon)
            near_work = nearest_stop_for_dataset(dataset, work_lat, work_lon)

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

    return rows


def _rank_commute_options(rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["total_min_est", "duration_min", "dataset"])
        .drop_duplicates(subset=["dataset", "route_short_name", "origin_dep_time", "dest_arr_time"], keep="first")
        .reset_index(drop=True)
    )


def _pretty_print_commute(out: pd.DataFrame, limit: int, title: str, format_units: bool) -> None:
    disp = out.head(limit).copy().rename(columns={
        "dataset": "Rede",
        "day": "Dia",
        "time": "Hora",
        "route_short_name": "Linha",
        "origin_name": "Origem",
        "dest_name": "Destino",
        "origin_dep_time": "Partida",
        "dest_arr_time": "Chegada",
        "duration_min": "Viagem",
        "walk_home_m": "A pé casa",
        "walk_work_m": "A pé trabalho",
        "total_min_est": "Total estimado",
    })

    if format_units:
        if "Viagem" in disp.columns:
            disp["Viagem"] = disp["Viagem"].map(lambda x: f"{float(x):.1f} min")
        if "Total estimado" in disp.columns:
            disp["Total estimado"] = disp["Total estimado"].map(lambda x: f"{float(x):.1f} min")
        if "A pé casa" in disp.columns:
            disp["A pé casa"] = disp["A pé casa"].map(lambda x: f"{float(x):.0f} m")
        if "A pé trabalho" in disp.columns:
            disp["A pé trabalho"] = disp["A pé trabalho"].map(lambda x: f"{float(x):.0f} m")

    cols = [
        c for c in [
            "Rede", "Dia", "Hora", "Linha", "Origem", "Destino",
            "Partida", "Chegada", "Viagem", "A pé casa", "A pé trabalho", "Total estimado"
        ] if c in disp.columns
    ]

    print(f"\n=== {title} ===")
    print(disp[cols].to_string(index=False))


def suggest_random_commute_options(
    home_lat: float,
    home_lon: float,
    work_lat: float,
    work_lon: float,
    limit: int = 5,
    tries: int = 20,
) -> pd.DataFrame:
    for _ in range(tries):
        d = date.today() + timedelta(days=random.randint(0, 20))
        hh = random.randint(6, 22)
        mm = random.randint(0, 59)
        ss = random.choice([0, 30])

        day_str = d.strftime("%Y-%m-%d")
        time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        rows = _collect_commute_options(
            home_lat=home_lat,
            home_lon=home_lon,
            work_lat=work_lat,
            work_lon=work_lon,
            day_str=day_str,
            time_str=time_str,
        )

        if rows:
            break

    if not rows:
        print("Não foi possível encontrar opções diretas nas tentativas aleatórias.")
        return pd.DataFrame()

    out = _rank_commute_options(rows)
    _pretty_print_commute(out=out, limit=limit, title="Melhores hipóteses casa -> trabalho", format_units=False)

    return out.head(limit)


def suggest_current_commute_options(
    home_lat: float,
    home_lon: float,
    work_lat: float,
    work_lon: float,
    limit: int = 3,
) -> pd.DataFrame:
    now = datetime.now()
    day_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    rows = _collect_commute_options(
        home_lat=home_lat,
        home_lon=home_lon,
        work_lat=work_lat,
        work_lon=work_lon,
        day_str=day_str,
        time_str=time_str,
    )

    if not rows:
        print("Sem opções diretas para a data/hora atual.")
        return pd.DataFrame()

    out = _rank_commute_options(rows)
    _pretty_print_commute(out=out, limit=limit, title=f"Top {limit} opções (agora)", format_units=True)

    return out.head(limit)


def next_monday(from_day: date) -> date:
    delta = (0 - from_day.weekday()) % 7
    return from_day + timedelta(days=delta)


def commute_options_for_datetime(
    home_lat: float,
    home_lon: float,
    work_lat: float,
    work_lon: float,
    day_str: str,
    time_str: str,
) -> pd.DataFrame:
    rows = _collect_commute_options(
        home_lat=home_lat,
        home_lon=home_lon,
        work_lat=work_lat,
        work_lon=work_lon,
        day_str=day_str,
        time_str=time_str,
    )
    return _rank_commute_options(rows)


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


def _route_direction_summaries(
    gtfs_smtuc,
    route_id: str,
    metro_stops: pd.DataFrame,
    walk_5_min_m: float,
) -> list[dict]:
    route_trips = gtfs_smtuc.trips[gtfs_smtuc.trips["route_id"].astype(str) == str(route_id)].copy()
    if route_trips.empty:
        return []

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

        route_stops["is_overlap_stop"] = pd.Series(min_dist) <= walk_5_min_m

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


def line_overlap_top(
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    top_n: int = 5,
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
) -> pd.DataFrame:
    gtfs_smtuc = load_gtfs(dataset=smtuc_dataset)
    gtfs_metro = load_gtfs(dataset=metrobus_dataset)

    walk_5_min_m = walk_speed_m_min * 5
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
            line_summaries.extend(
                _route_direction_summaries(
                    gtfs_smtuc=gtfs_smtuc,
                    route_id=route_id,
                    metro_stops=metro_stops,
                    walk_5_min_m=walk_5_min_m,
                )
            )

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
    return df.sort_values(["overlap_pct", "overlap_extension_m"], ascending=False).head(top_n).reset_index(drop=True)

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from smtuc_mvp.config import WALK_SPEED_M_MIN
from smtuc_mvp.gtfs_processing.gtfs import load_gtfs
from smtuc_mvp.gtfs_processing.gtfs_probe import (
    NearestStopResult,
    _active_service_ids,
    _haversine_m,
    _parse_day,
    _resolve_stop_id,
    _to_seconds,
)

@lru_cache(maxsize=8)
def _load_gtfs_cached(dataset: str):
    return load_gtfs(dataset=dataset)


def find_direct_options(dataset: str, origin_ref: str, dest_ref: str, day_str: str, time_str: str) -> pd.DataFrame:
    gtfs = _load_gtfs_cached(dataset)
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
    gtfs = _load_gtfs_cached(dataset)
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

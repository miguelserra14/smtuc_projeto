from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Iterable

import pandas as pd

from smtuc_mvp.gtfs import load_gtfs
casa=[40.207883,-8.398107]
trabalho= [40.186724, -8.416078] #DEI

def _to_seconds(hhmmss: str) -> int:
    h, m, s = map(int, hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def _parse_day(day_str: str) -> date:
    return datetime.strptime(day_str, "%Y-%m-%d").date()


def _weekday_col(d: date) -> str:
    return d.strftime("%A").lower()  # monday, tuesday, ...


def _active_service_ids(gtfs, d: date) -> set[str]:
    # fallback: sem calendar -> usa todos os services dos trips
    if gtfs.calendar.empty or "service_id" not in gtfs.calendar.columns:
        return set(gtfs.trips["service_id"].dropna().astype(str).unique())

    cal = gtfs.calendar.copy()
    cal["service_id"] = cal["service_id"].astype(str)
    ymd = int(d.strftime("%Y%m%d"))
    wd = _weekday_col(d)

    base = cal[
        (cal.get(wd, 0) == 1)
        & (cal["start_date"] <= ymd)
        & (cal["end_date"] >= ymd)
    ]["service_id"]

    active = set(base.astype(str).tolist())

    # exceções (calendar_dates)
    if not gtfs.calendar_dates.empty and {"service_id", "date", "exception_type"}.issubset(gtfs.calendar_dates.columns):
        cd = gtfs.calendar_dates[gtfs.calendar_dates["date"] == ymd]
        for _, r in cd.iterrows():
            sid = str(r["service_id"])
            ex = int(r["exception_type"])
            if ex == 1:
                active.add(sid)
            elif ex == 2:
                active.discard(sid)

    return active


def _resolve_stop_id(gtfs, ref: str) -> str:
    ref = str(ref).strip()

    # 1) id direto
    ids = gtfs.stops["stop_id"].astype(str)
    if (ids == ref).any():
        return ref

    # 2) nome exato
    names = gtfs.stops["stop_name"].astype(str) if "stop_name" in gtfs.stops.columns else pd.Series([], dtype=str)
    exact = gtfs.stops[names.str.lower() == ref.lower()]
    if len(exact) == 1:
        return str(exact.iloc[0]["stop_id"])

    # 3) nome parcial
    part = gtfs.stops[names.str.contains(ref, case=False, na=False)]
    if len(part) == 1:
        return str(part.iloc[0]["stop_id"])
    if len(part) > 1:
        sample = part[["stop_id", "stop_name"]].head(5).to_dict(orient="records")
        raise ValueError(f"Paragem ambígua '{ref}'. Exemplos: {sample}")

    raise ValueError(f"Paragem não encontrada: '{ref}'")


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


def _haversine_m(lat1: float, lon1: float, lat2: Iterable[float], lon2: Iterable[float]) -> pd.Series:
    r = 6371000.0
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)

    lat2s = pd.Series(lat2, dtype="float64").apply(math.radians)
    lon2s = pd.Series(lon2, dtype="float64").apply(math.radians)

    dlat = lat2s - lat1r
    dlon = lon2s - lon1r

    a = (dlat / 2).apply(math.sin) ** 2 + pd.Series(
        [math.cos(lat1r) * math.cos(v) for v in lat2s], dtype="float64"
    ) * (dlon / 2).apply(math.sin) ** 2

    c = 2 * a.apply(lambda x: math.atan2(math.sqrt(x), math.sqrt(1 - x)))
    return r * c


@dataclass
class NearestStopResult:
    dataset: str
    stop_id: str
    stop_name: str
    distance_m: float


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


def nearest_stop_to_work(work_lat: float = trabalho[0], work_lon: float = trabalho[1]) -> NearestStopResult:
    #smtuc = nearest_stop_for_dataset("smtuc", work_lat, work_lon)
    metrobus = nearest_stop_for_dataset("metrobus", work_lat, work_lon)
    return metrobus


def compare_nearest_network(lat: float, lon: float) -> tuple[NearestStopResult, NearestStopResult, str]:
    smtuc = nearest_stop_for_dataset("smtuc", lat, lon)
    metrobus = nearest_stop_for_dataset("metrobus", lat, lon)
    winner = "smtuc" if smtuc.distance_m <= metrobus.distance_m else "metrobus"
    return smtuc, metrobus, winner


def main() -> None:
    parser = argparse.ArgumentParser(description="Testes rápidos GTFS (A->B e proximidade SMTUC vs Metrobus)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route", help="Ver opções diretas de A para B num dia/hora")
    p_route.add_argument("--dataset", choices=["smtuc", "metrobus"], required=True)
    p_route.add_argument("--origin", required=True, help="stop_id ou parte do nome da origem")
    p_route.add_argument("--dest", required=True, help="stop_id ou parte do nome do destino")
    p_route.add_argument("--day", required=True, help="YYYY-MM-DD")
    p_route.add_argument("--time", required=True, help="HH:MM:SS")
    p_route.add_argument("--limit", type=int, default=10)

    p_near = sub.add_parser("nearest", help="Comparar se o ponto está mais perto de SMTUC ou Metrobus")
    p_near.add_argument("--lat", type=float, required=True)
    p_near.add_argument("--lon", type=float, required=True)

    p_commute = sub.add_parser("commute-random", help="Sugestão aleatória casa->trabalho")
    p_commute.add_argument("--home-lat", type=float, default=casa[0])
    p_commute.add_argument("--home-lon", type=float, default=casa[1])
    p_commute.add_argument("--work-lat", type=float, default=trabalho[0])
    p_commute.add_argument("--work-lon", type=float, default=trabalho[1])
    p_commute.add_argument("--limit", type=int, default=5)
    p_commute.add_argument("--tries", type=int, default=20)

    p_now = sub.add_parser("commute-now", help="Top opções casa->trabalho na data/hora atual")
    p_now.add_argument("--home-lat", type=float, default=casa[0])
    p_now.add_argument("--home-lon", type=float, default=casa[1])
    p_now.add_argument("--work-lat", type=float, default=trabalho[0])
    p_now.add_argument("--work-lon", type=float, default=trabalho[1])
    p_now.add_argument("--limit", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "route":
        df = find_direct_options(
            dataset=args.dataset,
            origin_ref=args.origin,
            dest_ref=args.dest,
            day_str=args.day,
            time_str=args.time,
        )
        print("Sem opções diretas." if df.empty else df.head(args.limit).to_string(index=False))
        return

    if args.cmd == "nearest":
        smtuc, metrobus, winner = compare_nearest_network(args.lat, args.lon)
        print(f"SMTUC   -> {smtuc.stop_name} ({smtuc.stop_id}) | {smtuc.distance_m:.1f} m")
        print(f"Metrobus-> {metrobus.stop_name} ({metrobus.stop_id}) | {metrobus.distance_m:.1f} m")
        print(f"Mais próximo: {winner}")
        return

    if args.cmd == "commute-random":
        suggest_random_commute_options(
            home_lat=args.home_lat,
            home_lon=args.home_lon,
            work_lat=args.work_lat,
            work_lon=args.work_lon,
            limit=args.limit,
            tries=args.tries,
        )
        return

    if args.cmd == "commute-now":
        suggest_current_commute_options(
            home_lat=args.home_lat,
            home_lon=args.home_lon,
            work_lat=args.work_lat,
            work_lon=args.work_lon,
            limit=args.limit,
        )
        return



def suggest_random_commute_options(
    home_lat: float,
    home_lon: float,
    work_lat: float,
    work_lon: float,
    limit: int = 5,
    tries: int = 20,
) -> pd.DataFrame:
    walk_speed_m_min = 80.0  # ~4.8 km/h
    rows: list[pd.DataFrame] = []

    for _ in range(tries):
        d = date.today() + timedelta(days=random.randint(0, 20))
        hh = random.randint(6, 22)
        mm = random.randint(0, 59)
        ss = random.choice([0, 30])

        day_str = d.strftime("%Y-%m-%d")
        time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

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
                df["walk_home_min"] = (df["walk_home_m"] / walk_speed_m_min).round(1)
                df["walk_work_min"] = (df["walk_work_m"] / walk_speed_m_min).round(1)
                df["total_min_est"] = (df["duration_min"] + df["walk_home_min"] + df["walk_work_min"]).round(1)
                rows.append(df)
            except Exception:
                continue

        if rows:
            break

    if not rows:
        print("Não foi possível encontrar opções diretas nas tentativas aleatórias.")
        return pd.DataFrame()

    out = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["total_min_est", "duration_min", "dataset"])
        .drop_duplicates(subset=["dataset", "route_short_name", "origin_dep_time", "dest_arr_time"], keep="first")
        .reset_index(drop=True)
    )

    disp = out.head(limit).rename(columns={
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

    print("\n=== Melhores hipóteses casa -> trabalho ===")
    print(disp[[c for c in [
        "Rede", "Dia", "Hora", "Linha", "Origem", "Destino",
        "Partida", "Chegada", "Viagem", "A pé casa", "A pé trabalho", "Total estimado"
    ] if c in disp.columns]].to_string(index=False))

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
    walk_speed_m_min = 80.0

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
            df["walk_home_min"] = (df["walk_home_m"] / walk_speed_m_min).round(1)
            df["walk_work_min"] = (df["walk_work_m"] / walk_speed_m_min).round(1)
            df["total_min_est"] = (df["duration_min"] + df["walk_home_min"] + df["walk_work_min"]).round(1)

            rows.append(df)
        except Exception:
            continue

    if not rows:
        print("Sem opções diretas para a data/hora atual.")
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True).sort_values(
        ["total_min_est", "duration_min", "dataset"]
    ).drop_duplicates(
        subset=["dataset", "route_short_name", "origin_dep_time", "dest_arr_time"],
        keep="first",
    ).reset_index(drop=True)

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

    if "Viagem" in disp.columns:
        disp["Viagem"] = disp["Viagem"].map(lambda x: f"{float(x):.1f} min")
    if "Total estimado" in disp.columns:
        disp["Total estimado"] = disp["Total estimado"].map(lambda x: f"{float(x):.1f} min")
    if "A pé casa" in disp.columns:
        disp["A pé casa"] = disp["A pé casa"].map(lambda x: f"{float(x):.0f} m")
    if "A pé trabalho" in disp.columns:
        disp["A pé trabalho"] = disp["A pé trabalho"].map(lambda x: f"{float(x):.0f} m")

    print(f"\n=== Top {limit} opções (agora) ===")
    print(disp[[
        c for c in [
            "Rede", "Dia", "Hora", "Linha", "Origem", "Destino",
            "Partida", "Chegada", "Viagem", "A pé casa", "A pé trabalho", "Total estimado"
        ] if c in disp.columns
    ]].to_string(index=False))

    return out.head(limit)

if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Iterable

import pandas as pd

casa = [40.207883, -8.398107]
trabalho = [40.186724, -8.416078]  # DEI
WALK_SPEED_M_MIN = 80.0  # ~4.8 km/h


@dataclass
class NearestStopResult:
    dataset: str
    stop_id: str
    stop_name: str
    distance_m: float


# -----------------------------------------------------------------------------
# 1) Helpers básicos de parsing/normalização (baixo nível)
# -----------------------------------------------------------------------------

def _to_seconds(hhmmss: str) -> int:
    h, m, s = map(int, hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def _parse_day(day_str: str) -> date:
    return datetime.strptime(day_str, "%Y-%m-%d").date()


def _weekday_col(d: date) -> str:
    return d.strftime("%A").lower()  # monday, tuesday, ...


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


# -----------------------------------------------------------------------------
# 2) Helpers GTFS intermédios (filtragem e resolução de entidades)
# -----------------------------------------------------------------------------

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



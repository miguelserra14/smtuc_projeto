from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


REQUIRED_GTFS = ["routes", "trips", "stop_times", "stops"]
OPTIONAL_GTFS = ["shapes", "calendar", "calendar_dates"]


@dataclass
class GTFSData:
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    stops: pd.DataFrame
    shapes: pd.DataFrame
    calendar: pd.DataFrame
    calendar_dates: pd.DataFrame


def _read_csv(source: Path, stem: str) -> pd.DataFrame:
    file_path = source / f"{stem}.txt"
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_gtfs(source_dir: Optional[str]) -> GTFSData:
    if source_dir is None:
        return synthetic_gtfs()

    source = Path(source_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Diretório GTFS inválido: {source_dir}")

    frames: Dict[str, pd.DataFrame] = {}
    for name in REQUIRED_GTFS + OPTIONAL_GTFS:
        frames[name] = _read_csv(source, name)

    missing = [name for name in REQUIRED_GTFS if frames[name].empty]
    if missing:
        missing_list = ", ".join(f"{name}.txt" for name in missing)
        raise ValueError(f"Arquivos GTFS obrigatórios em falta: {missing_list}")

    frames["stop_times"]["arrival_seconds"] = frames["stop_times"]["arrival_time"].apply(_to_seconds)
    frames["stop_times"]["departure_seconds"] = frames["stop_times"]["departure_time"].apply(_to_seconds)

    return GTFSData(
        routes=frames["routes"],
        trips=frames["trips"],
        stop_times=frames["stop_times"],
        stops=frames["stops"],
        shapes=frames["shapes"],
        calendar=frames["calendar"],
        calendar_dates=frames["calendar_dates"],
    )


def _to_seconds(hhmmss: str) -> int:
    if not isinstance(hhmmss, str) or ":" not in hhmmss:
        return 0
    parts = hhmmss.split(":")
    if len(parts) != 3:
        return 0
    h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
    return h * 3600 + m * 60 + s


def synthetic_gtfs() -> GTFSData:
    routes = pd.DataFrame(
        [
            {"route_id": "R1", "route_short_name": "1", "route_long_name": "Centro - Universidade", "route_type": 3},
            {"route_id": "R2", "route_short_name": "2", "route_long_name": "Hospital - Fórum", "route_type": 3},
            {"route_id": "MB", "route_short_name": "M", "route_long_name": "Metrobus Azul", "route_type": 3},
        ]
    )

    trips = pd.DataFrame(
        [
            {"route_id": "R1", "service_id": "WEEK", "trip_id": "R1_T1", "shape_id": "S1"},
            {"route_id": "R1", "service_id": "WEEK", "trip_id": "R1_T2", "shape_id": "S1"},
            {"route_id": "R2", "service_id": "WEEK", "trip_id": "R2_T1", "shape_id": "S2"},
            {"route_id": "MB", "service_id": "WEEK", "trip_id": "MB_T1", "shape_id": "S3"},
            {"route_id": "MB", "service_id": "WEEK", "trip_id": "MB_T2", "shape_id": "S3"},
        ]
    )

    stops = pd.DataFrame(
        [
            {"stop_id": "ST1", "stop_name": "Centro", "stop_lat": 40.211, "stop_lon": -8.429},
            {"stop_id": "ST2", "stop_name": "Baixa", "stop_lat": 40.214, "stop_lon": -8.423},
            {"stop_id": "ST3", "stop_name": "Universidade", "stop_lat": 40.208, "stop_lon": -8.412},
            {"stop_id": "ST4", "stop_name": "Hospital", "stop_lat": 40.225, "stop_lon": -8.401},
            {"stop_id": "ST5", "stop_name": "Fórum", "stop_lat": 40.198, "stop_lon": -8.391},
            {"stop_id": "ST6", "stop_name": "Terminal Metrobus", "stop_lat": 40.219, "stop_lon": -8.381},
        ]
    )

    stop_times = pd.DataFrame(
        [
            {"trip_id": "R1_T1", "arrival_time": "07:00:00", "departure_time": "07:00:00", "stop_id": "ST1", "stop_sequence": 1},
            {"trip_id": "R1_T1", "arrival_time": "07:18:00", "departure_time": "07:18:00", "stop_id": "ST3", "stop_sequence": 2},
            {"trip_id": "R1_T2", "arrival_time": "17:00:00", "departure_time": "17:00:00", "stop_id": "ST3", "stop_sequence": 1},
            {"trip_id": "R1_T2", "arrival_time": "17:18:00", "departure_time": "17:18:00", "stop_id": "ST1", "stop_sequence": 2},
            {"trip_id": "R2_T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "ST4", "stop_sequence": 1},
            {"trip_id": "R2_T1", "arrival_time": "08:26:00", "departure_time": "08:26:00", "stop_id": "ST5", "stop_sequence": 2},
            {"trip_id": "MB_T1", "arrival_time": "07:30:00", "departure_time": "07:30:00", "stop_id": "ST1", "stop_sequence": 1},
            {"trip_id": "MB_T1", "arrival_time": "07:48:00", "departure_time": "07:48:00", "stop_id": "ST6", "stop_sequence": 2},
            {"trip_id": "MB_T2", "arrival_time": "18:00:00", "departure_time": "18:00:00", "stop_id": "ST6", "stop_sequence": 1},
            {"trip_id": "MB_T2", "arrival_time": "18:18:00", "departure_time": "18:18:00", "stop_id": "ST1", "stop_sequence": 2},
        ]
    )
    stop_times["arrival_seconds"] = stop_times["arrival_time"].apply(_to_seconds)
    stop_times["departure_seconds"] = stop_times["departure_time"].apply(_to_seconds)

    shapes = pd.DataFrame(
        [
            {"shape_id": "S1", "shape_pt_lat": 40.211, "shape_pt_lon": -8.429, "shape_pt_sequence": 1},
            {"shape_id": "S1", "shape_pt_lat": 40.214, "shape_pt_lon": -8.423, "shape_pt_sequence": 2},
            {"shape_id": "S1", "shape_pt_lat": 40.208, "shape_pt_lon": -8.412, "shape_pt_sequence": 3},
            {"shape_id": "S2", "shape_pt_lat": 40.225, "shape_pt_lon": -8.401, "shape_pt_sequence": 1},
            {"shape_id": "S2", "shape_pt_lat": 40.212, "shape_pt_lon": -8.395, "shape_pt_sequence": 2},
            {"shape_id": "S2", "shape_pt_lat": 40.198, "shape_pt_lon": -8.391, "shape_pt_sequence": 3},
            {"shape_id": "S3", "shape_pt_lat": 40.211, "shape_pt_lon": -8.429, "shape_pt_sequence": 1},
            {"shape_id": "S3", "shape_pt_lat": 40.216, "shape_pt_lon": -8.412, "shape_pt_sequence": 2},
            {"shape_id": "S3", "shape_pt_lat": 40.219, "shape_pt_lon": -8.381, "shape_pt_sequence": 3},
        ]
    )

    calendar = pd.DataFrame(
        [
            {
                "service_id": "WEEK",
                "monday": 1,
                "tuesday": 1,
                "wednesday": 1,
                "thursday": 1,
                "friday": 1,
                "saturday": 0,
                "sunday": 0,
                "start_date": 20260101,
                "end_date": 20261231,
            }
        ]
    )

    calendar_dates = pd.DataFrame()

    return GTFSData(
        routes=routes,
        trips=trips,
        stop_times=stop_times,
        stops=stops,
        shapes=shapes,
        calendar=calendar,
        calendar_dates=calendar_dates,
    )

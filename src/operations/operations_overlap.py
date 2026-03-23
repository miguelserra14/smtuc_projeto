from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    OVERLAP_TABLE_TOP_N,
    STADIUM_COORD,
    STADIUM_MIN_EXTENSION_PCT,
    STADIUM_RADIUS_M,
    TEMPORAL_OVERLAP_MAX_MIN,
    WALK_SPEED_M_MIN,
)
from operations.operations_overlap_db import (
    build_line_metrics_db,
    load_line_metrics_db,
    _load_gtfs_cached,
    _line_to_route_ids,
)


def _filter_numeric_bus_lines(df: pd.DataFrame) -> pd.DataFrame:
    if "line" not in df.columns:
        return df
    return df[pd.to_numeric(df["line"], errors="coerce") < 100]


def line_overlap_top(
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    top_n: int = OVERLAP_TABLE_TOP_N,
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
) -> pd.DataFrame:
    base = build_line_metrics_db(
        smtuc_dataset=smtuc_dataset,
        metrobus_dataset=metrobus_dataset,
        walk_speed_m_min=walk_speed_m_min,
    )
    if base.empty:
        return pd.DataFrame()

    base = _filter_numeric_bus_lines(base)

    return base.sort_values(["overlap_pct", "overlap_extension_m"], ascending=False).head(top_n).reset_index(drop=True)


def line_low_overlap_near_stadium_top(
    stadium_lat: float = STADIUM_COORD[0],
    stadium_lon: float = STADIUM_COORD[1],
    radius_m: float = STADIUM_RADIUS_M,
    min_radius_extension_pct: float = STADIUM_MIN_EXTENSION_PCT,
    top_n: int = OVERLAP_TABLE_TOP_N,
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
) -> pd.DataFrame:
    base = build_line_metrics_db(
        smtuc_dataset=smtuc_dataset,
        metrobus_dataset=metrobus_dataset,
        walk_speed_m_min=walk_speed_m_min,
        stadium_lat=stadium_lat,
        stadium_lon=stadium_lon,
        radius_m=radius_m,
    )
    if base.empty:
        return pd.DataFrame()

    base = _filter_numeric_bus_lines(base)

    if "radius_m" in base.columns:
        base = base[np.isclose(base["radius_m"].astype(float), float(radius_m), atol=1e-6)]

    filtered = base[base["radius_extension_pct"].fillna(-1) >= min_radius_extension_pct].copy()
    if filtered.empty:
        return filtered

    return (
        filtered.sort_values(
            ["overlap_pct", "overlap_extension_m", "radius_extension_pct"],
            ascending=[True, True, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def _get_time_seconds(row):
    """Extrai segundos desde meia-noite de uma linha de stop_times."""
    if "arrival_seconds" in row and pd.notna(row["arrival_seconds"]):
        try:
            return int(row["arrival_seconds"])
        except (ValueError, TypeError):
            pass
    if "departure_seconds" in row and pd.notna(row["departure_seconds"]):
        try:
            return int(row["departure_seconds"])
        except (ValueError, TypeError):
            pass
    return None


def _service_day_masks(calendar_df: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    day_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if calendar_df is None or calendar_df.empty or "service_id" not in calendar_df.columns:
        return {}

    cal = calendar_df.copy()
    for day_col in day_cols:
        if day_col not in cal.columns:
            cal[day_col] = 0

    out: dict[str, tuple[int, ...]] = {}
    for _, row in cal.iterrows():
        service_id = str(row["service_id"])
        out[service_id] = tuple(int(row[day_col]) for day_col in day_cols)
    return out


def _service_days_overlap(
    smtuc_service_id: str,
    metro_service_id: str,
    smtuc_masks: dict[str, tuple[int, ...]],
    metro_masks: dict[str, tuple[int, ...]],
) -> bool:
    smtuc_mask = smtuc_masks.get(str(smtuc_service_id))
    metro_mask = metro_masks.get(str(metro_service_id))

    if smtuc_mask is None or metro_mask is None:
        return True

    return any((a == 1 and b == 1) for a, b in zip(smtuc_mask, metro_mask))


def _distances_from_point_m(
    query_lat: float,
    query_lon: float,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> np.ndarray:
    r = 6371000.0
    q_lat_r = np.radians(float(query_lat))
    q_lon_r = np.radians(float(query_lon))
    ref_lat_r = np.radians(ref_lat)
    ref_lon_r = np.radians(ref_lon)

    dlat = ref_lat_r - q_lat_r
    dlon = ref_lon_r - q_lon_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(q_lat_r) * np.cos(ref_lat_r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def _normalize_line_value(raw_line: object) -> str | None:
    if pd.isna(raw_line):
        return None

    line = str(raw_line).strip()
    if line.endswith(".0"):
        try:
            as_float = float(line)
            if as_float.is_integer():
                line = str(int(as_float))
        except ValueError:
            pass
    return line


def compute_temporal_overlaps_for_db(
    metrics_df: pd.DataFrame,
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
    temporal_overlap_max_min: float = TEMPORAL_OVERLAP_MAX_MIN,
) -> pd.DataFrame:
    """Adiciona colunas de overlap temporal ao DataFrame de métricas."""
    if metrics_df.empty:
        return metrics_df

    if "temporal_spatial_candidates_count" not in metrics_df.columns:
        metrics_df["temporal_spatial_candidates_count"] = 0
    if "temporal_overlaps_count" not in metrics_df.columns:
        metrics_df["temporal_overlaps_count"] = 0
    if "temporal_overlaps_pct" not in metrics_df.columns:
        metrics_df["temporal_overlaps_pct"] = 0.0
    
    gtfs_smtuc = _load_gtfs_cached(smtuc_dataset)
    gtfs_metro = _load_gtfs_cached(metrobus_dataset)
    
    walk_5_min_m = walk_speed_m_min * 5
    temporal_threshold_s = int(float(temporal_overlap_max_min) * 60)
    metro_stops = gtfs_metro.stops[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    metro_stops["stop_id"] = metro_stops["stop_id"].astype(str)

    metro_trips = gtfs_metro.trips[["trip_id", "service_id"]].copy()
    metro_stop_times = gtfs_metro.stop_times.copy().merge(metro_trips, on="trip_id", how="left")
    
    if metro_stops.empty or metro_stop_times.empty:
        return metrics_df
    
    # Preparar dados Metrobus
    metro_stop_times_with_times = metro_stop_times.copy()
    metro_stop_times_with_times["stop_id"] = metro_stop_times_with_times["stop_id"].astype(str)
    metro_stop_times_with_times["service_id"] = metro_stop_times_with_times["service_id"].astype(str)
    metro_stop_times_with_times["time_sec"] = metro_stop_times_with_times.apply(_get_time_seconds, axis=1)
    metro_stop_times_with_times = metro_stop_times_with_times.dropna(subset=["time_sec"])

    metro_passages = metro_stop_times_with_times.merge(metro_stops, on="stop_id", how="left").dropna(
        subset=["stop_lat", "stop_lon"]
    )

    metro_stop_ids = metro_stops["stop_id"].astype(str).to_numpy()
    metro_stop_lats = metro_stops["stop_lat"].astype(float).to_numpy()
    metro_stop_lons = metro_stops["stop_lon"].astype(float).to_numpy()

    metro_times_by_stop: dict[str, np.ndarray] = {}
    metro_services_by_stop: dict[str, np.ndarray] = {}
    for stop_id, group in metro_passages.groupby("stop_id"):
        metro_times_by_stop[str(stop_id)] = group["time_sec"].astype(int).to_numpy()
        metro_services_by_stop[str(stop_id)] = group["service_id"].astype(str).to_numpy()

    smtuc_day_masks = _service_day_masks(gtfs_smtuc.calendar)
    metro_day_masks = _service_day_masks(gtfs_metro.calendar)
    
    # Calcular para cada linha
    line_to_route_ids = _line_to_route_ids(gtfs_smtuc)
    
    for idx, row in metrics_df.iterrows():
        line = _normalize_line_value(row.get("line"))
        if not line:
            continue
        
        if line not in line_to_route_ids:
            continue
        
        route_ids = line_to_route_ids[line]
        
        # Obter stop_times para esta linha
        smtuc_trips = gtfs_smtuc.trips[
            gtfs_smtuc.trips["route_id"].astype(str).isin([str(r) for r in route_ids])
        ][["trip_id", "service_id"]].copy()
        smtuc_trips["trip_id"] = smtuc_trips["trip_id"].astype(str)
        smtuc_trips["service_id"] = smtuc_trips["service_id"].astype(str)
        
        smtuc_stop_times = gtfs_smtuc.stop_times[
            gtfs_smtuc.stop_times["trip_id"].astype(str).isin(smtuc_trips["trip_id"])
        ].copy()
        smtuc_stop_times["trip_id"] = smtuc_stop_times["trip_id"].astype(str)
        smtuc_stop_times = smtuc_stop_times.merge(smtuc_trips, on="trip_id", how="left")
        
        # Juntar com stops
        smtuc_stop_times = smtuc_stop_times.merge(
            gtfs_smtuc.stops[["stop_id", "stop_lat", "stop_lon"]],
            on="stop_id", how="left"
        ).dropna(subset=["stop_lat", "stop_lon"])
        
        # Calcular tempos
        smtuc_stop_times["time_sec"] = smtuc_stop_times.apply(_get_time_seconds, axis=1)
        smtuc_stop_times = smtuc_stop_times.dropna(subset=["time_sec"])
        
        total_passages_nearby = 0
        temporal_overlaps = 0
        
        for _, passage in smtuc_stop_times.iterrows():
            smtuc_lat = float(passage["stop_lat"])
            smtuc_lon = float(passage["stop_lon"])
            smtuc_time_s = int(passage["time_sec"])
            smtuc_service_id = str(passage.get("service_id", ""))
            
            distances = _distances_from_point_m(smtuc_lat, smtuc_lon, metro_stop_lats, metro_stop_lons)
            nearby_stop_ids = metro_stop_ids[distances <= walk_5_min_m]

            if len(nearby_stop_ids) == 0:
                continue

            total_passages_nearby += 1

            has_temporal_overlap = False
            for nearby_stop_id in nearby_stop_ids:
                stop_id = str(nearby_stop_id)
                stop_times = metro_times_by_stop.get(stop_id)
                if stop_times is None or len(stop_times) == 0:
                    continue

                time_diffs = np.abs(stop_times - smtuc_time_s)
                candidate_idx = np.where(time_diffs <= temporal_threshold_s)[0]
                if len(candidate_idx) == 0:
                    continue

                stop_services = metro_services_by_stop.get(stop_id)
                if stop_services is None or len(stop_services) == 0:
                    has_temporal_overlap = True
                    break

                for candidate in candidate_idx:
                    metro_service_id = str(stop_services[candidate])
                    if _service_days_overlap(
                        smtuc_service_id,
                        metro_service_id,
                        smtuc_day_masks,
                        metro_day_masks,
                    ):
                        has_temporal_overlap = True
                        break

                if has_temporal_overlap:
                    break

            if has_temporal_overlap:
                temporal_overlaps += 1
        
        metrics_df.at[idx, "temporal_spatial_candidates_count"] = total_passages_nearby
        metrics_df.at[idx, "temporal_overlaps_count"] = temporal_overlaps
        if total_passages_nearby > 0:
            metrics_df.at[idx, "temporal_overlaps_pct"] = round(
                (temporal_overlaps / total_passages_nearby) * 100, 2
            )
        else:
            metrics_df.at[idx, "temporal_overlaps_pct"] = 0.0
    
    return metrics_df


def temporal_overlap_events_for_metrics(
    metrics_df: pd.DataFrame,
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
    temporal_overlap_max_min: float = TEMPORAL_OVERLAP_MAX_MIN,
) -> pd.DataFrame:
    """Devolve eventos de overlap temporal para um conjunto de linhas em métricas."""
    if metrics_df.empty or "line" not in metrics_df.columns:
        return pd.DataFrame(
            columns=[
                "line",
                "smtuc_stop_id",
                "smtuc_stop_name",
                "time_sec",
                "time_hhmm",
                "hour",
                "nearby_metro_stops_count",
            ]
        )

    gtfs_smtuc = _load_gtfs_cached(smtuc_dataset)
    gtfs_metro = _load_gtfs_cached(metrobus_dataset)

    walk_5_min_m = walk_speed_m_min * 5
    temporal_threshold_s = int(float(temporal_overlap_max_min) * 60)

    metro_stops = gtfs_metro.stops[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    metro_stops["stop_id"] = metro_stops["stop_id"].astype(str)
    metro_trips = gtfs_metro.trips[["trip_id", "service_id"]].copy()
    metro_stop_times = gtfs_metro.stop_times.copy().merge(metro_trips, on="trip_id", how="left")

    if metro_stops.empty or metro_stop_times.empty:
        return pd.DataFrame(
            columns=[
                "line",
                "smtuc_stop_id",
                "smtuc_stop_name",
                "time_sec",
                "time_hhmm",
                "hour",
                "nearby_metro_stops_count",
            ]
        )

    metro_stop_times["stop_id"] = metro_stop_times["stop_id"].astype(str)
    metro_stop_times["service_id"] = metro_stop_times["service_id"].astype(str)
    metro_stop_times["time_sec"] = metro_stop_times.apply(_get_time_seconds, axis=1)
    metro_stop_times = metro_stop_times.dropna(subset=["time_sec"])

    metro_passages = metro_stop_times.merge(metro_stops, on="stop_id", how="left").dropna(subset=["stop_lat", "stop_lon"])

    metro_stop_ids = metro_stops["stop_id"].astype(str).to_numpy()
    metro_stop_lats = metro_stops["stop_lat"].astype(float).to_numpy()
    metro_stop_lons = metro_stops["stop_lon"].astype(float).to_numpy()

    metro_times_by_stop: dict[str, np.ndarray] = {}
    metro_services_by_stop: dict[str, np.ndarray] = {}
    for stop_id, group in metro_passages.groupby("stop_id"):
        metro_times_by_stop[str(stop_id)] = group["time_sec"].astype(int).to_numpy()
        metro_services_by_stop[str(stop_id)] = group["service_id"].astype(str).to_numpy()

    smtuc_day_masks = _service_day_masks(gtfs_smtuc.calendar)
    metro_day_masks = _service_day_masks(gtfs_metro.calendar)
    line_to_route_ids = _line_to_route_ids(gtfs_smtuc)

    stop_name_col = "stop_name" if "stop_name" in gtfs_smtuc.stops.columns else None
    stop_cols = ["stop_id", "stop_lat", "stop_lon"]
    if stop_name_col:
        stop_cols.append(stop_name_col)

    events: list[dict[str, object]] = []

    for _, row in metrics_df.iterrows():
        line = _normalize_line_value(row.get("line"))
        if not line or line not in line_to_route_ids:
            continue

        route_ids = line_to_route_ids[line]
        smtuc_trips = gtfs_smtuc.trips[
            gtfs_smtuc.trips["route_id"].astype(str).isin([str(r) for r in route_ids])
        ][["trip_id", "service_id"]].copy()
        smtuc_trips["trip_id"] = smtuc_trips["trip_id"].astype(str)
        smtuc_trips["service_id"] = smtuc_trips["service_id"].astype(str)

        smtuc_stop_times = gtfs_smtuc.stop_times[
            gtfs_smtuc.stop_times["trip_id"].astype(str).isin(smtuc_trips["trip_id"])
        ].copy()
        smtuc_stop_times["trip_id"] = smtuc_stop_times["trip_id"].astype(str)
        smtuc_stop_times = smtuc_stop_times.merge(smtuc_trips, on="trip_id", how="left")
        smtuc_stop_times = smtuc_stop_times.merge(gtfs_smtuc.stops[stop_cols], on="stop_id", how="left").dropna(
            subset=["stop_lat", "stop_lon"]
        )

        smtuc_stop_times["time_sec"] = smtuc_stop_times.apply(_get_time_seconds, axis=1)
        smtuc_stop_times = smtuc_stop_times.dropna(subset=["time_sec"])

        for _, passage in smtuc_stop_times.iterrows():
            smtuc_lat = float(passage["stop_lat"])
            smtuc_lon = float(passage["stop_lon"])
            smtuc_time_s = int(passage["time_sec"])
            smtuc_service_id = str(passage.get("service_id", ""))

            distances = _distances_from_point_m(smtuc_lat, smtuc_lon, metro_stop_lats, metro_stop_lons)
            nearby_stop_ids = metro_stop_ids[distances <= walk_5_min_m]
            if len(nearby_stop_ids) == 0:
                continue

            has_temporal_overlap = False
            for nearby_stop_id in nearby_stop_ids:
                stop_id = str(nearby_stop_id)
                stop_times = metro_times_by_stop.get(stop_id)
                if stop_times is None or len(stop_times) == 0:
                    continue

                time_diffs = np.abs(stop_times - smtuc_time_s)
                candidate_idx = np.where(time_diffs <= temporal_threshold_s)[0]
                if len(candidate_idx) == 0:
                    continue

                stop_services = metro_services_by_stop.get(stop_id)
                if stop_services is None or len(stop_services) == 0:
                    has_temporal_overlap = True
                    break

                for candidate in candidate_idx:
                    metro_service_id = str(stop_services[candidate])
                    if _service_days_overlap(
                        smtuc_service_id,
                        metro_service_id,
                        smtuc_day_masks,
                        metro_day_masks,
                    ):
                        has_temporal_overlap = True
                        break

                if has_temporal_overlap:
                    break

            if not has_temporal_overlap:
                continue

            h = smtuc_time_s // 3600
            m = (smtuc_time_s % 3600) // 60
            events.append(
                {
                    "line": line,
                    "smtuc_stop_id": str(passage["stop_id"]),
                    "smtuc_stop_name": str(passage.get(stop_name_col, "")) if stop_name_col else "",
                    "time_sec": smtuc_time_s,
                    "time_hhmm": f"{h:02d}:{m:02d}",
                    "hour": int(h),
                    "nearby_metro_stops_count": int(len(nearby_stop_ids)),
                }
            )

    return pd.DataFrame(events)

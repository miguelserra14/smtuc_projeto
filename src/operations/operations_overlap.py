from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    OVERLAP_TABLE_TOP_N,
    STADIUM_COORD,
    STADIUM_MIN_EXTENSION_PCT,
    STADIUM_RADIUS_M,
    WALK_SPEED_M_MIN,
)
from operations.operations_overlap_db import (
    build_line_metrics_db,
    load_line_metrics_db,
    _load_gtfs_cached,
    _line_to_route_ids,
    _min_distance_to_points_m,
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


def compute_temporal_overlaps_for_db(
    metrics_df: pd.DataFrame,
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
) -> pd.DataFrame:
    """Adiciona colunas de overlap temporal ao DataFrame de métricas."""
    if metrics_df.empty:
        return metrics_df
    
    gtfs_smtuc = _load_gtfs_cached(smtuc_dataset)
    gtfs_metro = _load_gtfs_cached(metrobus_dataset)
    
    walk_5_min_m = walk_speed_m_min * 5
    metro_stops = gtfs_metro.stops[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    metro_stop_times = gtfs_metro.stop_times.copy()
    
    if metro_stops.empty or metro_stop_times.empty:
        return metrics_df
    
    # Preparar dados Metrobus
    metro_stop_times_with_times = metro_stop_times.copy()
    metro_stop_times_with_times["time_sec"] = metro_stop_times_with_times.apply(
        _get_time_seconds, axis=1
    )
    metro_stop_times_with_times = metro_stop_times_with_times.dropna(subset=["time_sec"])
    
    metro_passages = metro_stop_times_with_times.merge(
        metro_stops, on="stop_id", how="left"
    ).dropna(subset=["stop_lat", "stop_lon"])
    
    metro_lats = metro_passages["stop_lat"].astype(float).values
    metro_lons = metro_passages["stop_lon"].astype(float).values
    metro_times = metro_passages["time_sec"].astype(int).values
    
    # Calcular para cada linha
    line_to_route_ids = _line_to_route_ids(gtfs_smtuc)
    
    for idx, row in metrics_df.iterrows():
        line = str(row["line"])
        
        if line not in line_to_route_ids:
            continue
        
        route_ids = line_to_route_ids[line]
        
        # Obter stop_times para esta linha
        smtuc_trips = gtfs_smtuc.trips[
            gtfs_smtuc.trips["route_id"].astype(str).isin([str(r) for r in route_ids])
        ]
        
        smtuc_stop_times = gtfs_smtuc.stop_times[
            gtfs_smtuc.stop_times["trip_id"].astype(str).isin(smtuc_trips["trip_id"].astype(str))
        ]
        
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
            
            # Verificar se há stops Metrobus próximos
            distances = _min_distance_to_points_m(
                np.array([smtuc_lat]), np.array([smtuc_lon]),
                metro_lats, metro_lons
            )[0]
            
            nearby_mask = distances <= walk_5_min_m
            
            if nearby_mask.any():
                total_passages_nearby += 1
                
                # Verificar se há Metrobus dentro de 5 minutos (300 segundos)
                nearby_times = metro_times[nearby_mask]
                time_diffs = np.abs(nearby_times - smtuc_time_s)
                
                if (time_diffs <= 300).any():
                    temporal_overlaps += 1
        
        if total_passages_nearby > 0:
            metrics_df.at[idx, "temporal_overlaps_count"] = temporal_overlaps
            metrics_df.at[idx, "temporal_overlaps_pct"] = round(
                (temporal_overlaps / total_passages_nearby) * 100, 2
            )
    
    return metrics_df

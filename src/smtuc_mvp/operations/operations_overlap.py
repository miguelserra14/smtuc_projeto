from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from smtuc_mvp.config import (
    OVERLAP_TABLE_TOP_N,
    STADIUM_COORD,
    STADIUM_MIN_EXTENSION_PCT,
    STADIUM_RADIUS_M,
    WALK_SPEED_M_MIN,
)
from smtuc_mvp.gtfs_processing.gtfs import load_gtfs


@lru_cache(maxsize=8)
def _load_gtfs_cached(dataset: str):
    return load_gtfs(dataset=dataset)


def _haversine_pairwise_m(
    from_lat: np.ndarray,
    from_lon: np.ndarray,
    to_lat: np.ndarray,
    to_lon: np.ndarray,
) -> np.ndarray:
    r = 6371000.0
    from_lat_r = np.radians(from_lat)
    from_lon_r = np.radians(from_lon)
    to_lat_r = np.radians(to_lat)
    to_lon_r = np.radians(to_lon)

    dlat = to_lat_r - from_lat_r
    dlon = to_lon_r - from_lon_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(from_lat_r) * np.cos(to_lat_r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def _min_distance_to_points_m(
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> np.ndarray:
    r = 6371000.0

    q_lat_r = np.radians(query_lat)[:, None]
    q_lon_r = np.radians(query_lon)[:, None]
    r_lat_r = np.radians(ref_lat)[None, :]
    r_lon_r = np.radians(ref_lon)[None, :]

    dlat = r_lat_r - q_lat_r
    dlon = r_lon_r - q_lon_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(q_lat_r) * np.cos(r_lat_r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dists = r * c
    return dists.min(axis=1)


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

    metro_lat = metro_stops["stop_lat"].astype(float).to_numpy()
    metro_lon = metro_stops["stop_lon"].astype(float).to_numpy()
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

        route_lat = route_stops["stop_lat"].astype(float).to_numpy()
        route_lon = route_stops["stop_lon"].astype(float).to_numpy()

        min_dist = _min_distance_to_points_m(
            query_lat=route_lat,
            query_lon=route_lon,
            ref_lat=metro_lat,
            ref_lon=metro_lon,
        )
        is_overlap = min_dist <= walk_5_min_m

        seg_lengths = _haversine_pairwise_m(
            from_lat=route_lat[:-1],
            from_lon=route_lon[:-1],
            to_lat=route_lat[1:],
            to_lon=route_lon[1:],
        )
        overlap_segments = is_overlap[:-1] & is_overlap[1:]

        total_ext_m = float(seg_lengths.sum())
        overlap_ext_m = float(seg_lengths[overlap_segments].sum())

        if total_ext_m <= 0:
            continue

        summaries.append(
            {
                "direction": direction,
                "total_ext_m": total_ext_m,
                "overlap_ext_m": overlap_ext_m,
                "overlap_stops": int(is_overlap.sum()),
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


def _line_to_route_ids(gtfs_smtuc) -> dict[str, list[str]]:
    routes_map_df = gtfs_smtuc.routes.copy()
    if routes_map_df.empty or "route_id" not in routes_map_df.columns:
        return {}

    name_col = "route_short_name" if "route_short_name" in routes_map_df.columns else "route_id"
    routes_map_df = routes_map_df[["route_id", name_col]].copy()
    routes_map_df["route_id"] = routes_map_df["route_id"].astype(str)
    routes_map_df["line"] = routes_map_df[name_col].astype(str)

    trips_enriched = gtfs_smtuc.trips.copy()
    trips_enriched["route_id"] = trips_enriched["route_id"].astype(str)
    trips_enriched = trips_enriched.merge(routes_map_df[["route_id", "line"]], on="route_id", how="left")
    trips_enriched["line"] = trips_enriched["line"].fillna(trips_enriched["route_id"])

    return (
        trips_enriched[["line", "route_id"]]
        .dropna()
        .drop_duplicates()
        .groupby("line")["route_id"]
        .apply(list)
        .to_dict()
    )


def _overlap_context(
    smtuc_dataset: str,
    metrobus_dataset: str,
    walk_speed_m_min: float,
) -> tuple[object, pd.DataFrame, dict[str, list[str]], float] | tuple[None, pd.DataFrame, dict[str, list[str]], float]:
    gtfs_smtuc = _load_gtfs_cached(smtuc_dataset)
    gtfs_metro = _load_gtfs_cached(metrobus_dataset)

    walk_5_min_m = walk_speed_m_min * 5
    metro_stops = gtfs_metro.stops[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    if metro_stops.empty:
        return None, pd.DataFrame(), {}, walk_5_min_m

    line_to_route_ids = _line_to_route_ids(gtfs_smtuc)
    if not line_to_route_ids:
        return None, pd.DataFrame(), {}, walk_5_min_m

    return gtfs_smtuc, metro_stops, line_to_route_ids, walk_5_min_m


def _aggregate_overlap_for_line(
    gtfs_smtuc,
    route_ids: list[str],
    metro_stops: pd.DataFrame,
    walk_5_min_m: float,
) -> dict | None:
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
        return None

    total_ext_m = sum(s["total_ext_m"] for s in line_summaries)
    if total_ext_m <= 0:
        return None

    overlap_ext_m = sum(s["overlap_ext_m"] for s in line_summaries)
    return {
        "line_extension_m": total_ext_m,
        "overlap_extension_m": overlap_ext_m,
        "overlap_pct": (overlap_ext_m / total_ext_m) * 100.0,
        "overlap_stops": int(sum(s["overlap_stops"] for s in line_summaries)),
        "total_stops": int(sum(s["total_stops"] for s in line_summaries)),
        "directions_considered": len(line_summaries),
    }


def _aggregate_radius_for_line(
    gtfs_smtuc,
    route_ids: list[str],
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> dict | None:
    radius_summaries: list[dict] = []
    for route_id in route_ids:
        radius_summaries.extend(
            _route_direction_radius_coverage_summaries(
                gtfs_smtuc=gtfs_smtuc,
                route_id=route_id,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_m=radius_m,
            )
        )

    if not radius_summaries:
        return None

    radius_total_ext_m = sum(s["total_ext_m"] for s in radius_summaries)
    if radius_total_ext_m <= 0:
        return None

    radius_ext_m = sum(s["in_radius_ext_m"] for s in radius_summaries)
    return {
        "radius_extension_m": radius_ext_m,
        "radius_extension_pct": (radius_ext_m / radius_total_ext_m) * 100.0,
    }


def _finalize_line_rows(
    out_rows: list[dict],
    top_n: int,
    sort_cols: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows)
    df = df[pd.to_numeric(df["line"], errors="coerce") < 100]
    return df.sort_values(sort_cols, ascending=ascending).head(top_n).reset_index(drop=True)


def _route_direction_radius_coverage_summaries(
    gtfs_smtuc,
    route_id: str,
    center_lat: float,
    center_lon: float,
    radius_m: float,
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

        route_lat = route_stops["stop_lat"].astype(float).to_numpy()
        route_lon = route_stops["stop_lon"].astype(float).to_numpy()

        dist_to_center = _haversine_pairwise_m(
            from_lat=np.full(route_lat.shape, center_lat, dtype=float),
            from_lon=np.full(route_lon.shape, center_lon, dtype=float),
            to_lat=route_lat,
            to_lon=route_lon,
        )
        in_radius = dist_to_center <= radius_m

        seg_lengths = _haversine_pairwise_m(
            from_lat=route_lat[:-1],
            from_lon=route_lon[:-1],
            to_lat=route_lat[1:],
            to_lon=route_lon[1:],
        )
        in_radius_segments = in_radius[:-1] & in_radius[1:]

        total_ext_m = float(seg_lengths.sum())
        in_radius_ext_m = float(seg_lengths[in_radius_segments].sum())

        if total_ext_m <= 0:
            continue

        summaries.append(
            {
                "direction": direction,
                "total_ext_m": total_ext_m,
                "in_radius_ext_m": in_radius_ext_m,
            }
        )

    return summaries


def line_overlap_top(
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    top_n: int = OVERLAP_TABLE_TOP_N,
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
) -> pd.DataFrame:
    gtfs_smtuc, metro_stops, line_to_route_ids, walk_5_min_m = _overlap_context(
        smtuc_dataset=smtuc_dataset,
        metrobus_dataset=metrobus_dataset,
        walk_speed_m_min=walk_speed_m_min,
    )
    if gtfs_smtuc is None:
        return pd.DataFrame()

    out_rows: list[dict] = []

    for line, route_ids in line_to_route_ids.items():
        overlap_stats = _aggregate_overlap_for_line(
            gtfs_smtuc=gtfs_smtuc,
            route_ids=route_ids,
            metro_stops=metro_stops,
            walk_5_min_m=walk_5_min_m,
        )
        if overlap_stats is None:
            continue

        out_rows.append(
            {
                "line": str(line),
                "avg_freq_min": _line_avg_frequency_min(gtfs_smtuc, route_ids),
                "overlap_extension_m": round(overlap_stats["overlap_extension_m"], 1),
                "line_extension_m": round(overlap_stats["line_extension_m"], 1),
                "overlap_pct": round(overlap_stats["overlap_pct"], 2),
                "overlap_stops": overlap_stats["overlap_stops"],
                "total_stops": overlap_stats["total_stops"],
                "directions_considered": overlap_stats["directions_considered"],
            }
        )

    return _finalize_line_rows(
        out_rows=out_rows,
        top_n=top_n,
        sort_cols=["overlap_pct", "overlap_extension_m"],
        ascending=[False, False],
    )


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
    gtfs_smtuc, metro_stops, line_to_route_ids, walk_5_min_m = _overlap_context(
        smtuc_dataset=smtuc_dataset,
        metrobus_dataset=metrobus_dataset,
        walk_speed_m_min=walk_speed_m_min,
    )
    if gtfs_smtuc is None:
        return pd.DataFrame()

    out_rows: list[dict] = []

    for line, route_ids in line_to_route_ids.items():
        overlap_stats = _aggregate_overlap_for_line(
            gtfs_smtuc=gtfs_smtuc,
            route_ids=route_ids,
            metro_stops=metro_stops,
            walk_5_min_m=walk_5_min_m,
        )
        radius_stats = _aggregate_radius_for_line(
            gtfs_smtuc=gtfs_smtuc,
            route_ids=route_ids,
            center_lat=stadium_lat,
            center_lon=stadium_lon,
            radius_m=radius_m,
        )
        if overlap_stats is None or radius_stats is None:
            continue

        if radius_stats["radius_extension_pct"] < min_radius_extension_pct:
            continue

        out_rows.append(
            {
                "line": str(line),
                "avg_freq_min": _line_avg_frequency_min(gtfs_smtuc, route_ids),
                "overlap_extension_m": round(overlap_stats["overlap_extension_m"], 1),
                "line_extension_m": round(overlap_stats["line_extension_m"], 1),
                "overlap_pct": round(overlap_stats["overlap_pct"], 2),
                "overlap_stops": overlap_stats["overlap_stops"],
                "total_stops": overlap_stats["total_stops"],
                "directions_considered": overlap_stats["directions_considered"],
                "radius_extension_m": round(radius_stats["radius_extension_m"], 1),
                "radius_extension_pct": round(radius_stats["radius_extension_pct"], 2),
                "radius_m": float(radius_m),
            }
        )

    return _finalize_line_rows(
        out_rows=out_rows,
        top_n=top_n,
        sort_cols=["overlap_pct", "overlap_extension_m", "radius_extension_pct"],
        ascending=[True, True, False],
    )

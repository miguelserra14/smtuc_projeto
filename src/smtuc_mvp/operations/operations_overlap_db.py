from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from smtuc_mvp.config import (
    LINE_METRICS_DB_PATH,
    STADIUM_COORD,
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
        direction_values = route_trips["direction_id"].dropna().astype(str).unique().tolist() or ["na"]
    else:
        direction_values = ["na"]

    metro_lat = metro_stops["stop_lat"].astype(float).to_numpy()
    metro_lon = metro_stops["stop_lon"].astype(float).to_numpy()
    summaries: list[dict] = []

    for direction in direction_values:
        subset = route_trips if direction == "na" else route_trips[route_trips["direction_id"].astype(str) == direction]

        route_stops = _representative_route_stops_for_subset(gtfs_smtuc, subset)
        route_stops = route_stops.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)
        if len(route_stops) < 2:
            continue

        route_lat = route_stops["stop_lat"].astype(float).to_numpy()
        route_lon = route_stops["stop_lon"].astype(float).to_numpy()

        min_dist = _min_distance_to_points_m(route_lat, route_lon, metro_lat, metro_lon)
        is_overlap = min_dist <= walk_5_min_m

        seg_lengths = _haversine_pairwise_m(route_lat[:-1], route_lon[:-1], route_lat[1:], route_lon[1:])
        overlap_segments = is_overlap[:-1] & is_overlap[1:]

        total_ext_m = float(seg_lengths.sum())
        overlap_ext_m = float(seg_lengths[overlap_segments].sum())

        if total_ext_m <= 0:
            continue

        summaries.append(
            {
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

    stop_times = gtfs_smtuc.stop_times[gtfs_smtuc.stop_times["trip_id"].astype(str).isin(trips["trip_id"].astype(str))]
    if stop_times.empty or "departure_seconds" not in stop_times.columns:
        return None

    first_dep = stop_times.groupby("trip_id", as_index=False).agg(first_dep_s=("departure_seconds", "min")).dropna(subset=["first_dep_s"])
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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_db_path(path: str | None = None) -> Path:
    rel = path if path is not None else LINE_METRICS_DB_PATH
    return (_project_root() / rel).resolve()


def _dataset_signature(dataset: str) -> str:
    folder = _project_root() / "data" / dataset
    txt_files = sorted(folder.glob("*.txt")) if folder.exists() else []
    raw = "|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in txt_files)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
        direction_values = route_trips["direction_id"].dropna().astype(str).unique().tolist() or ["na"]
    else:
        direction_values = ["na"]

    summaries: list[dict] = []

    for direction in direction_values:
        subset = route_trips if direction == "na" else route_trips[route_trips["direction_id"].astype(str) == direction]

        route_stops = _representative_route_stops_for_subset(gtfs_smtuc, subset)
        route_stops = route_stops.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)
        if len(route_stops) < 2:
            continue

        route_lat = route_stops["stop_lat"].astype(float).to_numpy()
        route_lon = route_stops["stop_lon"].astype(float).to_numpy()

        dist_to_center = _haversine_pairwise_m(
            np.full(route_lat.shape, center_lat, dtype=float),
            np.full(route_lon.shape, center_lon, dtype=float),
            route_lat,
            route_lon,
        )
        in_radius = dist_to_center <= radius_m

        seg_lengths = _haversine_pairwise_m(route_lat[:-1], route_lon[:-1], route_lat[1:], route_lon[1:])
        in_radius_segments = in_radius[:-1] & in_radius[1:]

        total_ext_m = float(seg_lengths.sum())
        in_radius_ext_m = float(seg_lengths[in_radius_segments].sum())

        if total_ext_m <= 0:
            continue

        summaries.append({"total_ext_m": total_ext_m, "in_radius_ext_m": in_radius_ext_m})

    return summaries


def _compute_line_metrics(
    smtuc_dataset: str,
    metrobus_dataset: str,
    walk_speed_m_min: float,
    stadium_lat: float,
    stadium_lon: float,
    radius_m: float,
) -> pd.DataFrame:
    gtfs_smtuc = _load_gtfs_cached(smtuc_dataset)
    gtfs_metro = _load_gtfs_cached(metrobus_dataset)

    walk_5_min_m = walk_speed_m_min * 5
    metro_stops = gtfs_metro.stops[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    if metro_stops.empty:
        return pd.DataFrame()

    line_to_route_ids = _line_to_route_ids(gtfs_smtuc)
    if not line_to_route_ids:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for line, route_ids in line_to_route_ids.items():
        line_summaries: list[dict] = []
        radius_summaries: list[dict] = []
        for route_id in route_ids:
            line_summaries.extend(_route_direction_summaries(gtfs_smtuc, route_id, metro_stops, walk_5_min_m))
            radius_summaries.extend(
                _route_direction_radius_coverage_summaries(gtfs_smtuc, route_id, stadium_lat, stadium_lon, radius_m)
            )

        if not line_summaries:
            continue

        total_ext_m = sum(s["total_ext_m"] for s in line_summaries)
        if total_ext_m <= 0:
            continue

        overlap_ext_m = sum(s["overlap_ext_m"] for s in line_summaries)
        row = {
            "line": str(line),
            "avg_freq_min": _line_avg_frequency_min(gtfs_smtuc, route_ids),
            "overlap_extension_m": round(overlap_ext_m, 1),
            "line_extension_m": round(total_ext_m, 1),
            "overlap_pct": round((overlap_ext_m / total_ext_m) * 100.0, 2),
            "overlap_stops": int(sum(s["overlap_stops"] for s in line_summaries)),
            "total_stops": int(sum(s["total_stops"] for s in line_summaries)),
            "directions_considered": len(line_summaries),
            "radius_extension_m": np.nan,
            "radius_extension_pct": np.nan,
            "radius_m": float(radius_m),
        }

        if radius_summaries:
            radius_total_ext_m = sum(s["total_ext_m"] for s in radius_summaries)
            if radius_total_ext_m > 0:
                radius_ext_m = sum(s["in_radius_ext_m"] for s in radius_summaries)
                row["radius_extension_m"] = round(radius_ext_m, 1)
                row["radius_extension_pct"] = round((radius_ext_m / radius_total_ext_m) * 100.0, 2)

        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows)
    return df[pd.to_numeric(df["line"], errors="coerce") < 100].reset_index(drop=True)


def build_line_metrics_db(
    smtuc_dataset: str = "smtuc",
    metrobus_dataset: str = "metrobus",
    walk_speed_m_min: float = WALK_SPEED_M_MIN,
    stadium_lat: float = STADIUM_COORD[0],
    stadium_lon: float = STADIUM_COORD[1],
    radius_m: float = STADIUM_RADIUS_M,
    db_path: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    csv_path = _resolve_db_path(db_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    signature_smtuc = _dataset_signature(smtuc_dataset)
    signature_metro = _dataset_signature(metrobus_dataset)

    if csv_path.exists() and not force_refresh:
        cached = pd.read_csv(csv_path)
        if not cached.empty:
            meta_ok = (
                str(cached.iloc[0].get("__meta_smtuc_dataset", "")) == smtuc_dataset
                and str(cached.iloc[0].get("__meta_metro_dataset", "")) == metrobus_dataset
                and float(cached.iloc[0].get("__meta_walk_speed_m_min", -1.0)) == float(walk_speed_m_min)
                and float(cached.iloc[0].get("__meta_stadium_lat", 999.0)) == float(stadium_lat)
                and float(cached.iloc[0].get("__meta_stadium_lon", 999.0)) == float(stadium_lon)
                and float(cached.iloc[0].get("__meta_radius_m", -1.0)) == float(radius_m)
                and str(cached.iloc[0].get("__meta_sig_smtuc", "")) == signature_smtuc
                and str(cached.iloc[0].get("__meta_sig_metro", "")) == signature_metro
            )
            if meta_ok:
                return cached.drop(columns=[c for c in cached.columns if c.startswith("__meta_")], errors="ignore")

    fresh = _compute_line_metrics(
        smtuc_dataset=smtuc_dataset,
        metrobus_dataset=metrobus_dataset,
        walk_speed_m_min=walk_speed_m_min,
        stadium_lat=stadium_lat,
        stadium_lon=stadium_lon,
        radius_m=radius_m,
    )
    if fresh.empty:
        return fresh

    fresh["__meta_smtuc_dataset"] = smtuc_dataset
    fresh["__meta_metro_dataset"] = metrobus_dataset
    fresh["__meta_walk_speed_m_min"] = float(walk_speed_m_min)
    fresh["__meta_stadium_lat"] = float(stadium_lat)
    fresh["__meta_stadium_lon"] = float(stadium_lon)
    fresh["__meta_radius_m"] = float(radius_m)
    fresh["__meta_sig_smtuc"] = signature_smtuc
    fresh["__meta_sig_metro"] = signature_metro
    fresh.to_csv(csv_path, index=False)

    return fresh.drop(columns=[c for c in fresh.columns if c.startswith("__meta_")], errors="ignore")


def load_line_metrics_db(db_path: str | None = None) -> pd.DataFrame:
    csv_path = _resolve_db_path(db_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Base central não encontrada: {csv_path}. "
            "Gera primeiro com build_line_metrics_db(...)."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    return df.drop(columns=[c for c in df.columns if c.startswith("__meta_")], errors="ignore")

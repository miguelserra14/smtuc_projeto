from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from gtfs_processing.gtfs import load_gtfs
from gtfs_processing.gtfs_probe import _active_service_ids, _parse_day
from population._common import (
    ensure_crs,
    infer_bgri_id_col,
    load_bgri_file,
    resolve_path,
)
from config import CATCHMENT_M, DEFAULT_BGRI_GPKG_PATH, DEFAULT_BGRI_LAYER, DEFAULT_OUTPUT_GAP_CSV, UNDERSERVED_TOP_N
try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency guard
    gpd = None



def _require_geopandas() -> None:
    if gpd is None:
        raise ImportError(
            "geopandas não está instalado. Instala com: pip install geopandas"
        )


def _departures_per_stop_for_day(dataset: str, day: date) -> pd.DataFrame:
    gtfs = load_gtfs(dataset=dataset)

    active_services = _active_service_ids(gtfs, day)
    trips = gtfs.trips.copy()
    trips["service_id"] = trips["service_id"].astype(str)
    if active_services:
        trips = trips[trips["service_id"].isin(active_services)]

    if trips.empty:
        return pd.DataFrame(columns=["stop_id", "departures", "stop_lat", "stop_lon", "dataset"])

    active_trip_ids = set(trips["trip_id"].astype(str))
    st = gtfs.stop_times[gtfs.stop_times["trip_id"].astype(str).isin(active_trip_ids)].copy()
    if st.empty:
        return pd.DataFrame(columns=["stop_id", "departures", "stop_lat", "stop_lon", "dataset"])

    dep = (
        st.groupby("stop_id", as_index=False)
        .size()
        .rename(columns={"size": "departures"})
        .merge(gtfs.stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
    )
    dep["dataset"] = dataset
    dep["departures"] = dep["departures"].astype(float)
    return dep.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)

#explicar aprofundadamente como é que o score é calculado. e como adaptar a cada situacao
def compute_bgri_population_transport_gap(
    day_str: str,
    catchment_m: float = CATCHMENT_M,
    datasets: Tuple[str, ...] = ("smtuc", "metrobus"),
    bgri_gpkg_path: str = DEFAULT_BGRI_GPKG_PATH,
    bgri_layer: str = DEFAULT_BGRI_LAYER,
    population_col: str = "N_INDIVIDUOS",
    output_csv_path: Optional[str] = DEFAULT_OUTPUT_GAP_CSV,
) -> pd.DataFrame:
    _require_geopandas()

    d = _parse_day(day_str)
    gpkg = resolve_path(bgri_gpkg_path)
    if not gpkg.exists():
        raise FileNotFoundError(f"Ficheiro BGRI não encontrado: {gpkg}")

    bgri = load_bgri_file(str(gpkg), bgri_layer)
    if bgri.empty:
        return pd.DataFrame()

    bgri_id_col = infer_bgri_id_col(bgri.columns)
    if population_col not in bgri.columns:
        raise ValueError(f"Coluna de população não encontrada: {population_col}")

    bgri = bgri[[bgri_id_col, population_col, "geometry"]].copy()
    bgri[population_col] = pd.to_numeric(bgri[population_col], errors="coerce").fillna(0.0)
    bgri = ensure_crs(bgri, "EPSG:3763")

    dep_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        dep = _departures_per_stop_for_day(dataset=dataset, day=d)
        if not dep.empty:
            dep_frames.append(dep)

    if not dep_frames:
        out = pd.DataFrame(columns=[bgri_id_col, population_col, "supply_departures", "dep_per_1000_pop", "pop_per_departure", "underservice_score"])
        if output_csv_path:
            out_path = resolve_path(output_csv_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
        return out

    departures = pd.concat(dep_frames, ignore_index=True)
    departures = departures.groupby(["stop_id", "stop_lat", "stop_lon"], as_index=False).agg(
        supply_departures=("departures", "sum")
    )

    stops_gdf = gpd.GeoDataFrame(
        departures,
        geometry=gpd.points_from_xy(departures["stop_lon"], departures["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(bgri.crs)

    centroids = bgri[[bgri_id_col, population_col, "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroid_buffers = centroids.copy()
    centroid_buffers["geometry"] = centroid_buffers.geometry.buffer(float(catchment_m))

    joined = gpd.sjoin(
        centroid_buffers[[bgri_id_col, "geometry"]],
        stops_gdf[["supply_departures", "geometry"]],
        how="left",
        predicate="intersects",
    )

    supply_by_bgri = (
        joined.groupby(bgri_id_col, as_index=False)["supply_departures"]
        .sum(min_count=1)
        .rename(columns={"supply_departures": "supply_departures"})
    )

    out = bgri[[bgri_id_col, population_col]].merge(supply_by_bgri, on=bgri_id_col, how="left")
    out["supply_departures"] = out["supply_departures"].fillna(0.0)
    out["dep_per_1000_pop"] = np.where(
        out[population_col] > 0,
        (out["supply_departures"] / out[population_col]) * 1000.0,
        0.0,
    )
    out["pop_per_departure"] = np.where(
        out["supply_departures"] > 0,
        out[population_col] / out["supply_departures"],
        np.inf,
    )
    out["underservice_score"] = out[population_col] / (out["supply_departures"] + 1.0)

    out = out.sort_values(["underservice_score", population_col], ascending=[False, False]).reset_index(drop=True)

    if output_csv_path:
        out_path = resolve_path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

    return out


def top_bgri_underserved(
    day_str: str,
    top_n: int = UNDERSERVED_TOP_N,
    catchment_m: float = CATCHMENT_M,
    datasets: Tuple[str, ...] = ("smtuc", "metrobus"),
    bgri_gpkg_path: str = DEFAULT_BGRI_GPKG_PATH,
    bgri_layer: str = DEFAULT_BGRI_LAYER,
    population_col: str = "N_INDIVIDUOS",
    output_csv_path: Optional[str] = DEFAULT_OUTPUT_GAP_CSV,
) -> pd.DataFrame:
    out = compute_bgri_population_transport_gap(
        day_str=day_str,
        catchment_m=catchment_m,
        datasets=datasets,
        bgri_gpkg_path=bgri_gpkg_path,
        bgri_layer=bgri_layer,
        population_col=population_col,
        output_csv_path=output_csv_path,
    )
    return out.head(top_n).reset_index(drop=True)


__all__ = [
    "compute_bgri_population_transport_gap",
    "top_bgri_underserved",
]

"""Data processing utilities for BGRI population transport gap analysis."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import pytest

from config import STADIUM_RADIUS_M, CATCHMENT_M
from population._common import (
    create_stadium_point,
    ensure_crs,
    load_bgri_file,
    project_root,
)
from population.operations_population import (
    compute_bgri_population_transport_gap,
)

if TYPE_CHECKING:
    import geopandas as gpd


def _project_root() -> Path:
    """Wrapper for project_root() to match test imports."""
    return project_root()


def _next_monday(from_date: date) -> date:
    """
    Get the next Monday from a given date.
    
    Args:
        from_date: Starting date
        
    Returns:
        Next Monday date (or today if today is Monday)
    """
    # Monday is 0 in Python's weekday()
    days_until_monday = (7 - from_date.weekday()) % 7
    if days_until_monday == 0:
        # If today is Monday, return today
        return from_date
    return from_date + timedelta(days=days_until_monday)


def _require_geo_stack() -> None:
    """Check if geospatial stack is available and skip test if not."""
    try:
        import geopandas as gpd
        import shapely
    except ImportError:
        pytest.skip("GeoPandas or Shapely não está disponível")


def _require_bgri_data() -> Path:
    """Require BGRI GeoPackage file and skip test if not found."""
    gpkg = project_root() / "data" / "dadospopulacaoBGRI" / "BGRI2021_0603.gpkg"
    if not gpkg.exists():
        pytest.skip(f"BGRI GPKG não encontrado: {gpkg}")
    return gpkg



def load_and_prepare_bgri(gpkg_path: str, layer: str = "BGRI2021_0603") -> gpd.GeoDataFrame:
    """
    Load and prepare BGRI geodataframe from GeoPackage.

    Args:
        gpkg_path: Path to BGRI GeoPackage file
        layer: Layer name in GeoPackage

    Returns:
        Prepared GeoDataFrame with BGRI2021 and geometry columns
    """
    bgri = load_bgri_file(gpkg_path, layer)
    
    if "N_INDIVIDUOS" not in bgri.columns:
        raise ValueError("Coluna N_INDIVIDUOS não encontrada")

    bgri = bgri[["BGRI2021", "N_INDIVIDUOS", "geometry"]].copy()
    bgri["N_INDIVIDUOS"] = pd.to_numeric(bgri["N_INDIVIDUOS"], errors="coerce").fillna(0.0)
    
    return bgri


def compute_underserved_zones(
    day_str: str,
    catchment_m: float = CATCHMENT_M,
    datasets: tuple = ("smtuc", "metrobus"),
    bgri_gpkg_path: Optional[str] = None,
    bgri_layer: str = "BGRI2021_0603",
    population_col: str = "N_INDIVIDUOS",
    output_csv_path: str = "outputs/bgri_transport_gap.csv",
) -> gpd.GeoDataFrame:
    """
    Compute BGRI underserved zones and merge with geographic data.

    Args:
        day_str: Day in format "YYYY-MM-DD"
        catchment_m: Catchment radius in meters
        datasets: Tuple of datasets to use
        bgri_gpkg_path: Path to BGRI GeoPackage
        bgri_layer: Layer name
        population_col: Population column name
        output_csv_path: Path to output CSV

    Returns:
        Merged GeoDataFrame with BGRI geometries and underservice scores
    """
    if bgri_gpkg_path is None:
        bgri_gpkg_path = str(_require_bgri_data())

    gap = compute_bgri_population_transport_gap(
        day_str=day_str,
        catchment_m=catchment_m,
        datasets=datasets,
        bgri_gpkg_path=bgri_gpkg_path,
        bgri_layer=bgri_layer,
        population_col=population_col,
        output_csv_path=output_csv_path,
    )

    if gap.empty:
        raise ValueError("Sem dados de gap para o dia selecionado")

    bgri = load_and_prepare_bgri(bgri_gpkg_path, bgri_layer)
    bgri = bgri[["BGRI2021", "geometry"]].copy()
    bgri["BGRI2021"] = bgri["BGRI2021"].astype(str)

    gap_plot = gap.copy()
    gap_plot["BGRI2021"] = gap_plot["BGRI2021"].astype(str)

    merged = bgri.merge(gap_plot, on="BGRI2021", how="inner")
    if merged.empty:
        raise ValueError("Join BGRI + gap vazio")

    if merged.crs is None:
        merged = merged.set_crs("EPSG:3763")
    elif str(merged.crs).upper() != "EPSG:3763":
        merged = merged.to_crs("EPSG:3763")

    return merged


def filter_zones_by_distance(
    merged: gpd.GeoDataFrame,
    distance_m: float = STADIUM_RADIUS_M,
) -> gpd.GeoDataFrame:
    """Filter BGRI zones by distance from stadium."""
    stadium_geo = create_stadium_point(merged.crs)
    stadium_point = stadium_geo.geometry.iloc[0]

    filtered = merged[merged.geometry.distance(stadium_point) <= distance_m].copy()
    if filtered.empty:
        raise ValueError(f"Sem zonas BGRI a <={distance_m}m do estádio")

    return filtered


def get_population_near_stadium(
    bgri_gpkg_path: Optional[str] = None,
    radius_m: float = STADIUM_RADIUS_M,
    layer: str = "BGRI2021_0603",
) -> tuple[float, float, float]:
    """
    Calculate population within radius of stadium.

    Args:
        bgri_gpkg_path: Path to BGRI GeoPackage
        radius_m: Radius in meters
        layer: Layer name

    Returns:
        Tuple of (total_population, population_in_radius, percentage)
    """
    if bgri_gpkg_path is None:
        bgri_gpkg_path = str(_require_bgri_data())

    bgri = load_and_prepare_bgri(bgri_gpkg_path, layer)
    stadium_geo = create_stadium_point(bgri.crs)
    stadium_buffer = stadium_geo.geometry.iloc[0].buffer(radius_m)
    
    intersects = bgri[bgri.geometry.intersects(stadium_buffer)].copy()
    if intersects.empty:
        raise ValueError(f"Nenhuma subsecção BGRI intersecta o raio de {radius_m}m do estádio")

    intersects["orig_area"] = intersects.geometry.area
    intersects["int_area"] = intersects.geometry.intersection(stadium_buffer).area
    intersects["area_share"] = (intersects["int_area"] / intersects["orig_area"]).clip(lower=0.0, upper=1.0)
    intersects["pop_in_radius"] = intersects["N_INDIVIDUOS"] * intersects["area_share"]

    total_pop = float(bgri["N_INDIVIDUOS"].sum())
    pop_in_radius = float(intersects["pop_in_radius"].sum())
    pct = (pop_in_radius / total_pop * 100.0) if total_pop > 0 else 0.0

    return total_pop, pop_in_radius, pct

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from smtuc_mvp.config import STADIUM_COORD
from smtuc_mvp.operations.operations_population import (
    compute_bgri_population_transport_gap,
)

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency guard
    gpd = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency guard
    px = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_bgri_data() -> Path:
    gpkg = _project_root() / "data" / "dadospopulacaoBGRI" / "BGRI2021_0603.gpkg"
    if not gpkg.exists():
        pytest.skip(f"BGRI GPKG não encontrado: {gpkg}")
    return gpkg


def _require_geo_stack() -> None:
    if gpd is None:
        pytest.skip("geopandas não está disponível no ambiente")


def _next_monday(from_day: date) -> date:
    delta = (0 - from_day.weekday()) % 7
    return from_day + pd.Timedelta(days=delta)


@pytest.mark.integration
def test_population_detail_near_stadium_1km() -> None:
    _require_geo_stack()
    gpkg = _require_bgri_data()

    bgri = gpd.read_file(gpkg, layer="BGRI2021_0603")
    if bgri.empty:
        pytest.skip("Layer BGRI vazio")

    if "N_INDIVIDUOS" not in bgri.columns:
        pytest.skip("Coluna N_INDIVIDUOS não encontrada")

    bgri = bgri[["BGRI2021", "N_INDIVIDUOS", "geometry"]].copy()
    bgri["N_INDIVIDUOS"] = pd.to_numeric(bgri["N_INDIVIDUOS"], errors="coerce").fillna(0.0)
    bgri = bgri[~bgri.geometry.isna()].copy()

    if bgri.crs is None:
        bgri = bgri.set_crs("EPSG:3763")
    elif str(bgri.crs).upper() != "EPSG:3763":
        bgri = bgri.to_crs("EPSG:3763")

    stadium_geo = gpd.GeoDataFrame(
        {"name": ["stadium"]},
        geometry=gpd.points_from_xy([STADIUM_COORD[1]], [STADIUM_COORD[0]]),
        crs="EPSG:4326",
    ).to_crs(bgri.crs)

    radius_m = 1000.0
    stadium_buffer = stadium_geo.geometry.iloc[0].buffer(radius_m)

    intersects = bgri[bgri.geometry.intersects(stadium_buffer)].copy()
    if intersects.empty:
        pytest.skip("Nenhuma subsecção BGRI intersecta o raio de 1km do estádio")

    intersects["orig_area"] = intersects.geometry.area
    intersects["int_area"] = intersects.geometry.intersection(stadium_buffer).area
    intersects["area_share"] = (intersects["int_area"] / intersects["orig_area"]).clip(lower=0.0, upper=1.0)
    intersects["pop_in_1km"] = intersects["N_INDIVIDUOS"] * intersects["area_share"]

    total_pop = float(bgri["N_INDIVIDUOS"].sum())
    pop_1km = float(intersects["pop_in_1km"].sum())
    pct = (pop_1km / total_pop * 100.0) if total_pop > 0 else 0.0

    print("\n=== População BGRI no raio de 1km do estádio (estimativa areal) ===")
    print(f"Subsecções BGRI totais: {len(bgri)}")
    print(f"Subsecções a intersectar 1km: {len(intersects)}")
    print(f"População total no concelho (BGRI): {total_pop:,.0f}")
    print(f"População estimada a <=1km do estádio: {pop_1km:,.0f} ({pct:.2f}%)")

    assert total_pop > 0
    assert pop_1km > 0


@pytest.mark.integration
def test_bgri_underserved_zones_with_visualizations() -> None:
    _require_geo_stack()
    gpkg = _require_bgri_data()

    monday = _next_monday(date.today())
    day_str = monday.strftime("%Y-%m-%d")

    gap = compute_bgri_population_transport_gap(
        day_str=day_str,
        catchment_m=500.0,
        datasets=("smtuc", "metrobus"),
        bgri_gpkg_path=str(gpkg),
        bgri_layer="BGRI2021_0603",
        population_col="N_INDIVIDUOS",
        output_csv_path="outputs/bgri_transport_gap.csv",
    )

    if gap.empty:
        pytest.skip("Sem dados de gap para o dia selecionado")

    print("\n=== Top 10 zonas BGRI mais subservidas (população vs oferta) ===")
    print(
        gap[["BGRI2021", "N_INDIVIDUOS", "supply_departures", "dep_per_1000_pop", "underservice_score"]]
        .head(10)
        .to_string(index=False)
    )

    if px is None:
        pytest.skip("plotly não está disponível para gerar visualizações")

    bgri = gpd.read_file(gpkg, layer="BGRI2021_0603")
    bgri = bgri[["BGRI2021", "geometry"]].copy()
    bgri["BGRI2021"] = bgri["BGRI2021"].astype(str)

    gap_plot = gap.copy()
    gap_plot["BGRI2021"] = gap_plot["BGRI2021"].astype(str)

    merged = bgri.merge(gap_plot, on="BGRI2021", how="inner")
    if merged.empty:
        pytest.skip("Join BGRI + gap vazio")

    out_dir = _project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson = merged.to_crs("EPSG:4326").__geo_interface__

    fig_map = px.choropleth(
        merged,
        geojson=geojson,
        locations="BGRI2021",
        featureidkey="properties.BGRI2021",
        color="underservice_score",
        hover_data={
            "N_INDIVIDUOS": ":.0f",
            "supply_departures": ":.0f",
            "dep_per_1000_pop": ":.2f",
            "BGRI2021": True,
        },
        title=f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 500m)",
        color_continuous_scale="Reds",
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})

    map_html = out_dir / "bgri_underservice_choropleth.html"
    fig_map.write_html(map_html, include_plotlyjs="cdn")

    scatter_df = gap_plot[gap_plot["N_INDIVIDUOS"] > 0].copy()
    fig_scatter = px.scatter(
        scatter_df,
        x="supply_departures",
        y="N_INDIVIDUOS",
        color="underservice_score",
        size="N_INDIVIDUOS",
        hover_name="BGRI2021",
        color_continuous_scale="Reds",
        title=f"População vs Oferta por BGRI (dia {day_str})",
        labels={
            "supply_departures": "Oferta (n.º de passagens no dia)",
            "N_INDIVIDUOS": "População",
            "underservice_score": "Índice de subserviço",
        },
    )
    fig_scatter.update_layout(margin={"l": 30, "r": 30, "t": 50, "b": 30})

    scatter_html = out_dir / "bgri_population_vs_supply_scatter.html"
    fig_scatter.write_html(scatter_html, include_plotlyjs="cdn")

    print(f"Mapa gerado: {map_html}")
    print(f"Scatter gerado: {scatter_html}")

    assert len(gap) > 0
    assert map_html.exists()
    assert scatter_html.exists()

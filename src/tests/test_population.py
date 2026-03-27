"""Integration tests for BGRI population transport gap analysis and visualizations."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import pytest

from population.data_processing import (
    _next_monday,
    _project_root,
    _require_bgri_data,
    _require_geo_stack,
    compute_underserved_zones,
    filter_zones_by_distance,
    get_population_near_stadium,
)
from population.visualizations import (
    _write_readable_plotly_html,
    create_2km_choropleth_map,
    create_choropleth_map,
    create_population_heatmap,
    create_scatter_plot,
)
from src.config import CATCHMENT_M, STADIUM_RADIUS_M, OUTPUTS_POPULATION_DIR


@pytest.mark.integration
def test_population_detail_near_stadium_1km() -> None:
    """Test population distribution within 1km radius of stadium."""
    _require_geo_stack()

    total_pop, pop_1km, pct = get_population_near_stadium(
        bgri_gpkg_path=str(_require_bgri_data()),
        radius_m=STADIUM_RADIUS_M/2,  # 1km radius
    )

    print("\n=== População BGRI no raio de 1km do estádio (estimativa areal) ===")
    print(f"População total no concelho (BGRI): {total_pop:,.0f}")
    print(f"População estimada a <=1km do estádio: {pop_1km:,.0f} ({pct:.2f}%)")

    assert total_pop > 0
    assert pop_1km > 0


@pytest.fixture(scope="module")
def bgri_underserved_context() -> dict[str, object]:
    """Build shared context for underserved zones and visualization outputs."""
    _require_geo_stack()
    gpkg = _require_bgri_data()

    monday = _next_monday(date.today())
    day_str = monday.strftime("%Y-%m-%d")

    # Compute underserved zones
    merged = compute_underserved_zones(
        day_str=day_str,
        catchment_m=CATCHMENT_M,
        datasets=("smtuc", "metrobus"),
        bgri_gpkg_path=str(gpkg),
        bgri_layer="BGRI2021_0603",
        population_col="N_INDIVIDUOS",
        output_csv_path=f"{OUTPUTS_POPULATION_DIR}/bgri_transport_gap.csv",
    )

    out_dir = _project_root() / OUTPUTS_POPULATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    assert len(merged) > 0
    return {
        "merged": merged,
        "day_str": day_str,
        "out_dir": out_dir,
    }


@pytest.mark.integration
def test_bgri_underserved_zones_dataset(bgri_underserved_context: dict[str, object]) -> None:
    """Validate underserved zones dataset used by all visualization tests."""
    merged = bgri_underserved_context["merged"]
    assert isinstance(merged, pd.DataFrame)
    assert len(merged) > 0

    print("\n=== Top 10 zonas BGRI mais subservidas (população vs oferta) ===")
    print(
        merged[["BGRI2021", "N_INDIVIDUOS", "supply_departures", "dep_per_1000_pop", "underservice_score"]]
        .head(10)
        .to_string(index=False)
    )


@pytest.mark.integration
def test_bgri_underserved_choropleth(bgri_underserved_context: dict[str, object]) -> None:
    """Generate and validate underserved choropleth visualization."""
    merged = bgri_underserved_context["merged"]
    day_str = bgri_underserved_context["day_str"]
    out_dir = bgri_underserved_context["out_dir"]

    # Generate main choropleth
    fig_map = create_choropleth_map(merged, day_str, color_scale="YlOrRd")
    map_html = out_dir / "bgri_underservice_choropleth.html"
    _write_readable_plotly_html(fig_map, map_html, "BGRI Coimbra — Choropleth")

    print(f"Mapa gerado: {map_html}")
    assert map_html.exists()


@pytest.mark.integration
def test_bgri_2km_choropleth(bgri_underserved_context: dict[str, object]) -> None:
    """Generate and validate 2km stadium choropleth visualization."""
    merged = bgri_underserved_context["merged"]
    day_str = bgri_underserved_context["day_str"]
    out_dir = bgri_underserved_context["out_dir"]

    # Generate 2km choropleth
    merged_2km = filter_zones_by_distance(merged, distance_m=STADIUM_RADIUS_M*2)
    fig_map_2km = create_2km_choropleth_map(merged_2km, day_str, color_scale="YlOrRd")
    map_2km_html = out_dir / "2kmstadium.html"
    _write_readable_plotly_html(fig_map_2km, map_2km_html, "BGRI Coimbra — Choropleth 2km")

    print(f"Mapa (<=2km estádio) gerado: {map_2km_html}")
    assert map_2km_html.exists()


@pytest.mark.integration
def test_bgri_population_supply_scatter(bgri_underserved_context: dict[str, object]) -> None:
    """Generate and validate population vs supply scatter visualization."""
    merged = bgri_underserved_context["merged"]
    day_str = bgri_underserved_context["day_str"]
    out_dir = bgri_underserved_context["out_dir"]

    # Generate scatter plot
    scatter_df = merged[merged["N_INDIVIDUOS"] > 0].copy()
    fig_scatter = create_scatter_plot(scatter_df, day_str)
    scatter_html = out_dir / "bgri_population_vs_supply_scatter.html"
    _write_readable_plotly_html(fig_scatter, scatter_html, "BGRI Coimbra — Scatter")

    print(f"Scatter gerado: {scatter_html}")
    assert scatter_html.exists()


@pytest.mark.integration
def test_bgri_population_heatmap(bgri_underserved_context: dict[str, object]) -> None:
    """Generate and validate population heatmap visualization."""
    merged = bgri_underserved_context["merged"]
    day_str = bgri_underserved_context["day_str"]
    out_dir = bgri_underserved_context["out_dir"]

    # Generate population heatmap (red -> green)
    fig_population_heatmap = create_population_heatmap(merged, day_str, color_scale="RdYlGn")
    population_heatmap_html = out_dir / "bgri_population_heatmap.html"
    _write_readable_plotly_html(
        fig_population_heatmap,
        population_heatmap_html,
        "BGRI Coimbra — Heatmap de População",
    )
    print(f"Heatmap de população gerado: {population_heatmap_html}")
    assert population_heatmap_html.exists()

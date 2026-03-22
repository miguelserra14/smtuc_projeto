"""Integration tests for BGRI population transport gap analysis and visualizations."""

from __future__ import annotations

from datetime import date

import pytest

from smtuc_mvp.population.data_processing import (
    _next_monday,
    _project_root,
    _require_bgri_data,
    _require_geo_stack,
    compute_underserved_zones,
    filter_zones_by_distance,
    get_population_near_stadium,
)
from smtuc_mvp.population.visualizations import (
    _write_readable_plotly_html,
    create_2km_choropleth_map,
    create_choropleth_map,
    create_scatter_plot,
)


@pytest.mark.integration
def test_population_detail_near_stadium_1km() -> None:
    """Test population distribution within 1km radius of stadium."""
    _require_geo_stack()

    total_pop, pop_1km, pct = get_population_near_stadium(
        bgri_gpkg_path=str(_require_bgri_data()),
        radius_m=1000.0,
    )

    print("\n=== População BGRI no raio de 1km do estádio (estimativa areal) ===")
    print(f"População total no concelho (BGRI): {total_pop:,.0f}")
    print(f"População estimada a <=1km do estádio: {pop_1km:,.0f} ({pct:.2f}%)")

    assert total_pop > 0
    assert pop_1km > 0


@pytest.mark.integration
def test_bgri_underserved_zones_with_visualizations() -> None:
    """Test underserved zones analysis and generate visualizations."""
    _require_geo_stack()
    gpkg = _require_bgri_data()

    monday = _next_monday(date.today())
    day_str = monday.strftime("%Y-%m-%d")

    # Compute underserved zones
    merged = compute_underserved_zones(
        day_str=day_str,
        catchment_m=500.0,
        datasets=("smtuc", "metrobus"),
        bgri_gpkg_path=str(gpkg),
        bgri_layer="BGRI2021_0603",
        population_col="N_INDIVIDUOS",
        output_csv_path="outputs/bgri_transport_gap.csv",
    )

    print("\n=== Top 10 zonas BGRI mais subservidas (população vs oferta) ===")
    print(
        merged[["BGRI2021", "N_INDIVIDUOS", "supply_departures", "dep_per_1000_pop", "underservice_score"]]
        .head(10)
        .to_string(index=False)
    )

    out_dir = _project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get color range for choropleth
    score_min = float(merged["underservice_score"].min())
    score_max = float(merged["underservice_score"].max())
    if score_max <= score_min:
        score_max = score_min + 1.0

    # Generate main choropleth
    fig_map = create_choropleth_map(merged, day_str, color_scale="Reds")
    map_html = out_dir / "bgri_underservice_choropleth.html"
    _write_readable_plotly_html(fig_map, map_html, "BGRI Coimbra — Choropleth")

    # Generate 2km choropleth
    merged_2km = filter_zones_by_distance(merged, distance_m=2000.0)
    fig_map_2km = create_2km_choropleth_map(merged_2km, day_str, score_min, score_max, color_scale="Reds")
    map_2km_html = out_dir / "2kmstadium.html"
    _write_readable_plotly_html(fig_map_2km, map_2km_html, "BGRI Coimbra — Choropleth 2km")

    # Generate scatter plot
    scatter_df = merged[merged["N_INDIVIDUOS"] > 0].copy()
    fig_scatter = create_scatter_plot(scatter_df, day_str)
    scatter_html = out_dir / "bgri_population_vs_supply_scatter.html"
    _write_readable_plotly_html(fig_scatter, scatter_html, "BGRI Coimbra — Scatter")

    print(f"Mapa gerado: {map_html}")
    print(f"Mapa (<=2km estádio) gerado: {map_2km_html}")
    print(f"Scatter gerado: {scatter_html}")

    assert len(merged) > 0
    assert map_html.exists()
    assert map_2km_html.exists()
    assert scatter_html.exists()

from __future__ import annotations

from typing import Dict, Optional, Tuple

import geopandas as gpd

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency guard
    px = None


def _create_choropleth_generic(
    gdf: gpd.GeoDataFrame,
    title: str,
    color_scale: str = "YlOrRd",
    color_col: str = "underservice_score",
    range_color: Optional[Tuple[float, float]] = None,
    hover_data: Optional[Dict] = None,
) -> object:
    """Create generic choropleth map."""
    if px is None:
        raise ImportError("plotly não está disponível para gerar visualizações")

    if not hover_data:
        hover_data = {
            "N_INDIVIDUOS": ":.0f",
            "supply_departures": ":.0f",
            "dep_per_1000_pop": ":.2f",
            "BGRI2021": True,
        }

    if range_color is None:
        score_p5 = float(gdf[color_col].quantile(0.05))
        score_p95 = float(gdf[color_col].quantile(0.90))
        if score_p95 <= score_p5:
            score_p5 = float(gdf[color_col].min())
            score_p95 = float(gdf[color_col].max())
        if score_p95 <= score_p5:
            score_p95 = score_p5 + 1.0
        range_color = (score_p5, score_p95)

    geojson = gdf.to_crs("EPSG:4326").__geo_interface__

    fig = px.choropleth(
        gdf,
        geojson=geojson,
        locations="BGRI2021",
        featureidkey="properties.BGRI2021",
        color=color_col,
        hover_data=hover_data,
        title=title,
        color_continuous_scale=color_scale,
        range_color=range_color,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})

    return fig


def create_choropleth_map(
    merged: gpd.GeoDataFrame,
    day_str: str,
    color_scale: str = "YlOrRd",
) -> object:
    map_title = f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 500m)"
    return _create_choropleth_generic(merged, map_title, color_scale)


def create_2km_choropleth_map(
    merged_2km: gpd.GeoDataFrame,
    day_str: str,
    color_scale: str = "YlOrRd",
) -> object:
    map_title = f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 2km)"
    return _create_choropleth_generic(
        merged_2km,
        map_title,
        color_scale,
    )


def create_population_heatmap(
    merged: gpd.GeoDataFrame,
    day_str: str,
    color_scale: str = "RdYlGn",
) -> object:
    map_title = f"BGRI Coimbra — Heatmap de População (dia {day_str})"
    return _create_choropleth_generic(
        merged,
        map_title,
        color_scale,
        color_col="N_INDIVIDUOS",
        hover_data={
            "N_INDIVIDUOS": ":.0f",
            "BGRI2021": True,
        },
    )


def create_scatter_plot(
    scatter_df,
    day_str: str,
) -> object:
    if px is None:
        raise ImportError("plotly não está disponível para gerar visualizações")

    scatter_score_p5 = float(scatter_df["underservice_score"].quantile(0.05))
    scatter_score_p95 = float(scatter_df["underservice_score"].quantile(0.95))
    if scatter_score_p95 <= scatter_score_p5:
        scatter_score_p5 = float(scatter_df["underservice_score"].min())
        scatter_score_p95 = float(scatter_df["underservice_score"].max())
    if scatter_score_p95 <= scatter_score_p5:
        scatter_score_p95 = scatter_score_p5 + 1.0

    fig_scatter = px.scatter(
        scatter_df,
        x="supply_departures",
        y="N_INDIVIDUOS",
        color="underservice_score",
        size="N_INDIVIDUOS",
        hover_name="BGRI2021",
        color_continuous_scale="YlOrRd",
        range_color=(scatter_score_p5, scatter_score_p95),
        title=f"População vs Oferta por BGRI (dia {day_str})",
        labels={
            "supply_departures": "Oferta (n.º de passagens no dia)",
            "N_INDIVIDUOS": "População",
            "underservice_score": "Índice de subserviço",
        },
    )
    fig_scatter.update_traces(
        marker={
            "opacity": 0.9,
            "line": {"color": "black", "width": 0.7},
        },
        selector={"mode": "markers"},
    )
    fig_scatter.update_layout(
        margin={"l": 0, "r": 30, "t": 50, "b": 0},
        plot_bgcolor="white",
        xaxis={
            "gridcolor": "black",
            "showgrid": False,
            "showline": False,
            "zeroline": True,
            "zerolinecolor": "black",
            "zerolinewidth": 2,
            "range": [0, None],
        },
        yaxis={
            "gridcolor": "black",
            "showgrid": False,
            "showline": False,
            "zeroline": True,
            "zerolinecolor": "black",
            "zerolinewidth": 2,
            "range": [0, None],
        },
    )

    return fig_scatter

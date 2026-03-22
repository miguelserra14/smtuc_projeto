"""Visualization utilities for BGRI population transport gap analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import geopandas as gpd

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency guard
    px = None


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="color-scheme" content="light only" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
  <style>
    :root {{ color-scheme: light only; }}
    html, body {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
    #plot {{ width: 100%; height: 100%; isolation: isolate; }}
    #plot, #plot * {{ forced-color-adjust: none !important; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    function applyDarkModeGuard() {{
      const plot = document.getElementById('plot');
      if (!plot) return;
      const htmlFilter = getComputedStyle(document.documentElement).filter;
      const bodyFilter = getComputedStyle(document.body).filter;
      const pageFilter = htmlFilter && htmlFilter !== 'none' ? htmlFilter : (bodyFilter && bodyFilter !== 'none' ? bodyFilter : 'none');
      plot.style.setProperty('background', '#ffffff', 'important');
      plot.style.setProperty('color-scheme', 'light', 'important');
      plot.style.setProperty('forced-color-adjust', 'none', 'important');
      if (pageFilter !== 'none') {{
        plot.style.setProperty('filter', pageFilter, 'important');
      }} else {{
        plot.style.removeProperty('filter');
      }}
    }}

    applyDarkModeGuard();
    const figure = {figure_json};
    Plotly.newPlot('plot', figure.data, figure.layout, {{ responsive: true }}).then(() => {{
      applyDarkModeGuard();
      setTimeout(applyDarkModeGuard, 100);
      setTimeout(applyDarkModeGuard, 500);
    }});

    const darkModeObserver = new MutationObserver(() => applyDarkModeGuard());
    darkModeObserver.observe(document.documentElement, {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }});
    darkModeObserver.observe(document.body, {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }});
  </script>
</body>
</html>
"""


def _write_readable_plotly_html(fig, html_path: Path, page_title: str) -> None:
    """Write Plotly figure to readable HTML file with dark-mode guard."""
    figure_json = fig.to_json()
    html = _HTML_TEMPLATE.format(title=page_title, figure_json=figure_json)
    html_path.write_text(html, encoding="utf-8")


def _create_choropleth_generic(
    gdf: gpd.GeoDataFrame,
    title: str,
    color_scale: str = "Reds",
    color_col: str = "underservice_score",
    range_color: Optional[Tuple[float, float]] = None,
    hover_data: Optional[Dict] = None,
) -> object:
    """
    Create generic choropleth map.
    
    Args:
        gdf: GeoDataFrame to visualize
        title: Map title
        color_scale: Plotly color scale name
        color_col: Column to use for coloring
        range_color: Tuple of (min, max) for color range
        hover_data: Dictionary of hover data columns
    
    Returns:
        Plotly figure object
    """
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
        score_min = float(gdf[color_col].min())
        score_max = float(gdf[color_col].max())
        if score_max <= score_min:
            score_max = score_min + 1.0
        range_color = (score_min, score_max)
    
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
    color_scale: str = "Reds",
) -> object:
    """
    Create choropleth map for underservice score.

    Args:
        merged: GeoDataFrame with underservice_score column
        day_str: Day in format "YYYY-MM-DD"
        color_scale: Plotly color scale name

    Returns:
        Plotly figure object
    """
    map_title = f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 500m)"
    return _create_choropleth_generic(merged, map_title, color_scale)


def create_2km_choropleth_map(
    merged_2km: gpd.GeoDataFrame,
    day_str: str,
    score_min: float,
    score_max: float,
    color_scale: str = "Reds",
) -> object:
    """
    Create choropleth map for zones within 2km of stadium.

    Args:
        merged_2km: GeoDataFrame with zones within 2km
        day_str: Day in format "YYYY-MM-DD"
        score_min: Minimum score for color range
        score_max: Maximum score for color range
        color_scale: Plotly color scale name

    Returns:
        Plotly figure object
    """
    map_title = f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 500m)"
    return _create_choropleth_generic(
        merged_2km,
        map_title,
        color_scale,
        range_color=(score_min, score_max),
    )


def create_scatter_plot(
    scatter_df,
    day_str: str,
) -> object:
    """
    Create scatter plot of population vs supply.

    Args:
        scatter_df: DataFrame with supply_departures, N_INDIVIDUOS, underservice_score
        day_str: Day in format "YYYY-MM-DD"

    Returns:
        Plotly figure object
    """
    if px is None:
        raise ImportError("plotly não está disponível para gerar visualizações")

    scatter_score_min = float(scatter_df["underservice_score"].min())
    scatter_score_q95 = float(scatter_df["underservice_score"].quantile(0.95))
    if scatter_score_q95 <= scatter_score_min:
        scatter_score_q95 = float(scatter_df["underservice_score"].max())
    if scatter_score_q95 <= scatter_score_min:
        scatter_score_q95 = scatter_score_min + 1.0

    fig_scatter = px.scatter(
        scatter_df,
        x="supply_departures",
        y="N_INDIVIDUOS",
        color="underservice_score",
        size="N_INDIVIDUOS",
        hover_name="BGRI2021",
        color_continuous_scale="YlOrRd",
        range_color=(scatter_score_min, scatter_score_q95),
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

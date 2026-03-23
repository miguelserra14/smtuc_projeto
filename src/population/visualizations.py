"""Visualization utilities for BGRI population transport gap analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import geopandas as gpd

try:
    import plotly.express as px
    import plotly.io as pio
except Exception:  # pragma: no cover - optional dependency guard
    px = None
    pio = None

try:
    import folium
except Exception:  # pragma: no cover - optional dependency guard
    folium = None


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


def _write_readable_plotly_html(
    fig: object,
    output_path: Path | str,
    title: str = "Visualization",
) -> None:
    """
    Write a readable Plotly figure to an HTML file with dark mode protection.

    Args:
        fig: Plotly figure object
        output_path: Path where to save the HTML file
        title: HTML page title
    """
    if pio is None:
        raise ImportError("plotly.io não está disponível")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get figure JSON
    fig_json = pio.to_json(fig)
    
    # Fill template with figure JSON and title
    html_content = _HTML_TEMPLATE.format(figure_json=fig_json, title=title)
    
    # Write to file
    output_path.write_text(html_content, encoding="utf-8")


def _write_folium_html(map_obj: object, output_path: Path | str) -> None:
    """Write a Folium map to an HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))


def _create_choropleth_generic(
    gdf: gpd.GeoDataFrame,
    title: str,
    color_scale: str = "YlOrRd",
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
        # Use robust percentiles instead of min/max to avoid outlier distortion
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
    color_scale: str = "YlOrRd",
) -> object:
    """
    Create choropleth map for zones within 2km of stadium.

    Args:
        merged_2km: GeoDataFrame with zones within 2km
        day_str: Day in format "YYYY-MM-DD"
        color_scale: Plotly color scale name

    Returns:
        Plotly figure object
    """
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
    """
    Create choropleth heatmap for population distribution.

    Args:
        merged: GeoDataFrame with N_INDIVIDUOS column
        day_str: Day in format "YYYY-MM-DD"
        color_scale: Plotly color scale name

    Returns:
        Plotly figure object
    """
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

    # Use robust percentiles to handle outliers gracefully
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


def create_overlap_reachability_map(
    reach_gdf: gpd.GeoDataFrame,
    origin_lat: float,
    origin_lon: float,
    day_str: str,
    time_str: str,
) -> object:
    """
    Create overlap reachability map with 10/15/30/60 minute classes.

    Args:
        reach_gdf: GeoDataFrame with reach_min, reach_mode, reach_bin columns
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        day_str: Day in format "YYYY-MM-DD"
        time_str: Time in format "HH:MM:SS"

    Returns:
        Folium map object
    """
    if folium is None:
        raise ImportError("folium não está disponível para gerar visualizações de mapa")

    gdf = reach_gdf.to_crs("EPSG:4326").copy()
    if "BGRI2021" not in gdf.columns:
        gdf["BGRI2021"] = gdf.index.astype(str)

    gdf["reach_min"] = gdf["reach_min"].astype(float)
    gdf["reach_min_display"] = gdf["reach_min"].map(lambda v: f"{v:.1f} min" if v <= 9999 else ">60 min")

    color_by_bin = {
        "0-10": "#d73027",
        "10-15": "#fc8d59",
        "15-30": "#fee08b",
        "30-60": "#91cf60",
        ">60": "#1a9850",
    }

    m = folium.Map(
        location=[origin_lat, origin_lon],
        zoom_start=13,
        tiles="cartodbpositron",
        control_scale=True,
    )

    def _style_fn(feature):
        reach_bin = str(feature["properties"].get("reach_bin", ">60"))
        return {
            "fillColor": color_by_bin.get(reach_bin, "#1a9850"),
            "color": "#333333",
            "weight": 0.6,
            "fillOpacity": 0.55,
        }

    tooltip_fields = [
        "BGRI2021",
        "reach_min_display",
        "reach_bin",
        "reach_mode",
        "N_INDIVIDUOS",
    ]
    tooltip_aliases = [
        "Zona BGRI",
        "Tempo estimado",
        "Classe",
        "Modo dominante",
        "População",
    ]

    present_fields = [f for f in tooltip_fields if f in gdf.columns]
    present_aliases = [tooltip_aliases[tooltip_fields.index(f)] for f in present_fields]

    folium.GeoJson(
        data=gdf,
        name="Alcance temporal",
        style_function=_style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=present_fields,
            aliases=present_aliases,
            localize=True,
            sticky=True,
            style=(
                "background-color: #ffffff;"
                " color: #222222;"
                " border: 1px solid #999999;"
                " border-radius: 4px;"
                " box-shadow: 0 1px 3px rgba(0,0,0,0.2);"
                " padding: 6px;"
                " font-size: 12px;"
            ),
        ),
    ).add_to(m)

    folium.CircleMarker(
        location=[origin_lat, origin_lon],
        radius=7,
        color="#b30000",
        fill=True,
        fill_color="#e34a33",
        fill_opacity=1.0,
        tooltip=f"Origem ({day_str} {time_str})",
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; border: 1px solid #999; padding: 10px 12px;
                font-size: 12px; line-height: 1.3;">
      <div style="font-weight: 600; margin-bottom: 6px;">Tempo de alcance (min)</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#d73027;margin-right:6px;"></span>0-10</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#fc8d59;margin-right:6px;"></span>10-15</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#fee08b;margin-right:6px;"></span>15-30</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#91cf60;margin-right:6px;"></span>30-60</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#1a9850;margin-right:6px;"></span>&gt;60</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m

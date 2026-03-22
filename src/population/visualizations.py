"""Visualization utilities for BGRI population transport gap analysis."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency guard
    px = None


def _write_readable_plotly_html(fig, html_path: Path, page_title: str) -> None:
    """
    Write Plotly figure to readable HTML file with dark-mode guard.

    Args:
        fig: Plotly figure object
        html_path: Path to write HTML file
        page_title: Title for HTML page
    """
    figure_json = fig.to_json()
    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"pt\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
        "  <meta name=\"color-scheme\" content=\"light only\" />\n"
        f"  <title>{page_title}</title>\n"
        "  <script src=\"https://cdn.plot.ly/plotly-3.4.0.min.js\"></script>\n"
        "  <style>\n"
        "    :root { color-scheme: light only; }\n"
        "    html, body { width: 100%; height: 100%; margin: 0; padding: 0; }\n"
        "    #plot { width: 100%; height: 100%; isolation: isolate; }\n"
        "    #plot, #plot * { forced-color-adjust: none !important; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div id=\"plot\"></div>\n"
        "  <script>\n"
        "    function applyDarkModeGuard() {\n"
        "      const plot = document.getElementById('plot');\n"
        "      if (!plot) return;\n"
        "      const htmlFilter = getComputedStyle(document.documentElement).filter;\n"
        "      const bodyFilter = getComputedStyle(document.body).filter;\n"
        "      const pageFilter = htmlFilter && htmlFilter !== 'none' ? htmlFilter : (bodyFilter && bodyFilter !== 'none' ? bodyFilter : 'none');\n"
        "      plot.style.setProperty('background', '#ffffff', 'important');\n"
        "      plot.style.setProperty('color-scheme', 'light', 'important');\n"
        "      plot.style.setProperty('forced-color-adjust', 'none', 'important');\n"
        "      if (pageFilter !== 'none') {\n"
        "        plot.style.setProperty('filter', pageFilter, 'important');\n"
        "      } else {\n"
        "        plot.style.removeProperty('filter');\n"
        "      }\n"
        "    }\n"
        "\n"
        "    applyDarkModeGuard();\n"
        "    const figure = "
        f"{figure_json}"
        ";\n"
        "    Plotly.newPlot('plot', figure.data, figure.layout, { responsive: true }).then(() => {\n"
        "      applyDarkModeGuard();\n"
        "      setTimeout(applyDarkModeGuard, 100);\n"
        "      setTimeout(applyDarkModeGuard, 500);\n"
        "    });\n"
        "\n"
        "    const darkModeObserver = new MutationObserver(() => applyDarkModeGuard());\n"
        "    darkModeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] });\n"
        "    darkModeObserver.observe(document.body, { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    html_path.write_text(html, encoding="utf-8")


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
    if px is None:
        raise ImportError("plotly não está disponível para gerar visualizações")

    map_title = f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 500m)"
    score_min = float(merged["underservice_score"].min())
    score_max = float(merged["underservice_score"].max())
    if score_max <= score_min:
        score_max = score_min + 1.0

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
        title=map_title,
        color_continuous_scale=color_scale,
        range_color=(score_min, score_max),
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})

    return fig_map


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
    if px is None:
        raise ImportError("plotly não está disponível para gerar visualizações")

    map_title = f"BGRI Coimbra — Índice de Subserviço (dia {day_str}, raio 500m)"
    geojson_2km = merged_2km.to_crs("EPSG:4326").__geo_interface__

    fig_map_2km = px.choropleth(
        merged_2km,
        geojson=geojson_2km,
        locations="BGRI2021",
        featureidkey="properties.BGRI2021",
        color="underservice_score",
        hover_data={
            "N_INDIVIDUOS": ":.0f",
            "supply_departures": ":.0f",
            "dep_per_1000_pop": ":.2f",
            "BGRI2021": True,
        },
        title=map_title,
        color_continuous_scale=color_scale,
        range_color=(score_min, score_max),
    )
    fig_map_2km.update_geos(fitbounds="locations", visible=False)
    fig_map_2km.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})

    return fig_map_2km


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

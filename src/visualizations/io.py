from __future__ import annotations

from pathlib import Path

try:
    import plotly.io as pio
except Exception:  # pragma: no cover - optional dependency guard
    pio = None


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

    fig_json = pio.to_json(fig)
    html_content = _HTML_TEMPLATE.format(figure_json=fig_json, title=title)
    output_path.write_text(html_content, encoding="utf-8")


def _write_folium_html(map_obj: object, output_path: Path | str) -> None:
    """Write a Folium map to an HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))

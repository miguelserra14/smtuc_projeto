from __future__ import annotations

import json

import geopandas as gpd
from shapely.ops import unary_union

try:
    import folium
except Exception:  # pragma: no cover - optional dependency guard
    folium = None


def create_overlap_reachability_map(
    reach_gdf: gpd.GeoDataFrame,
    origin_lat: float,
    origin_lon: float,
    day_str: str,
    time_str: str,
) -> object:
    """Create a true isochrone map with 10-minute bands from a fixed origin."""
    if folium is None:
        raise ImportError("folium não está disponível para gerar visualizações de mapa")

    def _add_mouse_origin_marker(map_obj: object) -> None:
        map_name = map_obj.get_name()
        mouse_origin_script = f"""
        (function attachMouseOrigin() {{
            var mapObj = window.{map_name};
            if (!mapObj) {{
                setTimeout(attachMouseOrigin, 50);
                return;
            }}

            mapObj.getContainer().style.cursor = 'crosshair';

            var originHalo = L.circleMarker([{origin_lat}, {origin_lon}], {{
                radius: 14,
                color: '#111111',
                fillColor: '#ffffff',
                fillOpacity: 0.18,
                opacity: 0.85,
                weight: 1.2,
                interactive: false
            }}).addTo(mapObj);

            var originPoint = L.circleMarker([{origin_lat}, {origin_lon}], {{
                radius: 7,
                color: '#b30000',
                fillColor: '#e34a33',
                fillOpacity: 1.0,
                weight: 2
            }}).addTo(mapObj);

            originPoint.bindTooltip('Origem dinâmica ({day_str} {time_str})', {{sticky: true, permanent: false}});

            mapObj.on('mousemove', function(e) {{
                originHalo.setLatLng(e.latlng);
                originPoint.setLatLng(e.latlng);
                originHalo.bringToFront();
                originPoint.bringToFront();
                originPoint.setTooltipContent(
                    'Origem dinâmica ({day_str} {time_str})' +
                    '<br>Lat: ' + e.latlng.lat.toFixed(5) +
                    ' | Lon: ' + e.latlng.lng.toFixed(5)
                );
            }});
        }})();
        """
        map_obj.get_root().script.add_child(folium.Element(mouse_origin_script))

    gdf = reach_gdf.copy()
    if gdf.empty or "geometry" not in gdf.columns:
        m = folium.Map(location=[origin_lat, origin_lon], zoom_start=13, tiles="cartodbpositron", control_scale=True)
        _add_mouse_origin_marker(m)
        return m

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    gdf["reach_min"] = gdf["reach_min"].astype(float)
    gdf_ll = gdf.to_crs("EPSG:4326").copy()
    gdf_centers = gdf.to_crs("EPSG:3763").copy()
    gdf_centers.geometry = gdf_centers.geometry.centroid
    gdf_ll["center_lat"] = gdf_centers.to_crs("EPSG:4326").geometry.y.astype(float)
    gdf_ll["center_lon"] = gdf_centers.to_crs("EPSG:4326").geometry.x.astype(float)

    try:
        metric_crs = gdf.estimate_utm_crs()
        gdf_metric = gdf.to_crs(metric_crs) if metric_crs else gdf.to_crs("EPSG:3857")
    except Exception:
        gdf_metric = gdf.to_crs("EPSG:3857")

    thresholds = [10, 20, 30, 40, 50, 60]
    smooth_m = 120.0
    color_by_band = {
        "0-10": "#d73027",
        "10-20": "#f46d43",
        "20-30": "#fdae61",
        "30-40": "#fee08b",
        "40-50": "#a6d96a",
        "50-60": "#1a9850",
    }

    cumulative_geoms: dict[int, object | None] = {}
    for upper in thresholds:
        subset = gdf_metric[gdf_metric["reach_min"] <= float(upper)]
        if subset.empty:
            cumulative_geoms[upper] = None
            continue
        merged = unary_union(subset.geometry.values)
        if merged.is_empty:
            cumulative_geoms[upper] = None
            continue
        smoothed = merged.buffer(smooth_m).buffer(-smooth_m)
        if smoothed.is_empty:
            smoothed = merged
        cumulative_geoms[upper] = smoothed.buffer(0)

    band_records = []
    prev_geom = None
    for upper in thresholds:
        curr_geom = cumulative_geoms.get(upper)
        if curr_geom is None or curr_geom.is_empty:
            continue
        band_geom = curr_geom if prev_geom is None else curr_geom.difference(prev_geom)
        band_geom = band_geom.buffer(0)
        if band_geom.is_empty:
            prev_geom = curr_geom
            continue
        lower = upper - 10
        band_label = f"{lower}-{upper}"
        band_records.append(
            {
                "band_label": band_label,
                "upper_min": upper,
                "area_km2": float(band_geom.area) / 1_000_000.0,
                "geometry": band_geom,
            }
        )
        prev_geom = curr_geom

    if not band_records:
        m = folium.Map(location=[origin_lat, origin_lon], zoom_start=13, tiles="cartodbpositron", control_scale=True)
        _add_mouse_origin_marker(m)
        return m

    iso_gdf_metric = gpd.GeoDataFrame(band_records, geometry="geometry", crs=gdf_metric.crs)
    iso_gdf = iso_gdf_metric.to_crs("EPSG:4326")
    dynamic_zones = gdf_ll[["reach_min", "center_lat", "center_lon", "geometry"]].copy()
    dynamic_zones = dynamic_zones[dynamic_zones.geometry.notna()].copy()
    dynamic_zones = dynamic_zones[dynamic_zones.geometry.is_valid].copy()
    dynamic_zone_geojson = json.dumps(dynamic_zones.__geo_interface__)

    m = folium.Map(
        location=[origin_lat, origin_lon],
        zoom_start=13,
        tiles="cartodbpositron",
        control_scale=True,
    )

    def _style_fn(feature):
        label = str(feature["properties"].get("band_label", "50-60"))
        return {
            "fillColor": color_by_band.get(label, "#1a9850"),
            "color": "#ffffff",
            "weight": 1.0,
            "fillOpacity": 0.55,
        }

    iso_layer = folium.GeoJson(
        data=iso_gdf,
        name="Isócronas (10 min)",
        style_function=_style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["band_label", "area_km2"],
            aliases=["Intervalo (min)", "Área (km²)"],
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

    _add_mouse_origin_marker(m)

    map_name = m.get_name()
    iso_layer_name = iso_layer.get_name()
    dynamic_iso_script = f"""
    (function attachDynamicIsoColor() {{
        var mapObj = window.{map_name};
        var isoLayer = window.{iso_layer_name};
        if (!mapObj || !isoLayer) {{
            setTimeout(attachDynamicIsoColor, 50);
            return;
        }}

        var baseOrigin = L.latLng({origin_lat}, {origin_lon});
        var zoneGeoJson = {dynamic_zone_geojson};
        var thresholds = [10, 20, 30, 40, 50, 60];

        function colorByBand(label) {{
            if (label === '0-10') return '#d73027';
            if (label === '10-20') return '#f46d43';
            if (label === '20-30') return '#fdae61';
            if (label === '30-40') return '#fee08b';
            if (label === '40-50') return '#a6d96a';
            return '#1a9850';
        }}

        function unionFeatures(features) {{
            if (!features || features.length === 0) return null;
            var acc = features[0];
            for (var i = 1; i < features.length; i++) {{
                try {{
                    acc = turf.union(acc, features[i]) || acc;
                }} catch (e) {{}}
            }}
            return acc;
        }}

        function diffSafe(a, b) {{
            if (!a) return null;
            if (!b) return a;
            try {{
                return turf.difference(a, b) || null;
            }} catch (e) {{
                return a;
            }}
        }}

        function updateLegendAreaMap(areaByBand) {{
            var keys = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'];
            keys.forEach(function(k) {{
                var id = 'legend-area-' + k.replace('-', '_');
                var el = document.getElementById(id);
                if (el) el.textContent = (areaByBand[k] || 0).toFixed(3) + ' km²';
            }});
            var total = (areaByBand['0-10'] || 0) + (areaByBand['10-20'] || 0) + (areaByBand['20-30'] || 0);
            var totalEl = document.getElementById('legend-area-total-30');
            if (totalEl) totalEl.textContent = total.toFixed(3) + ' km²';
        }}

        function buildDynamicIsochrones(latlng) {{
            var byThreshold = {{}};
            thresholds.forEach(function(t) {{ byThreshold[t] = []; }});

            zoneGeoJson.features.forEach(function(ft) {{
                if (!ft || !ft.properties) return;
                var cLat = Number(ft.properties.center_lat);
                var cLon = Number(ft.properties.center_lon);
                var baseReach = Number(ft.properties.reach_min);
                if (isNaN(cLat) || isNaN(cLon) || isNaN(baseReach)) return;

                var cPoint = L.latLng(cLat, cLon);
                var dNow = mapObj.distance(latlng, cPoint);
                var dBase = mapObj.distance(baseOrigin, cPoint);
                var estMin = baseReach + ((dNow - dBase) / 80.0);

                thresholds.forEach(function(t) {{
                    if (estMin <= t) byThreshold[t].push(ft);
                }});
            }});

            var cumGeo = {{}};
            thresholds.forEach(function(t) {{
                cumGeo[t] = unionFeatures(byThreshold[t]);
            }});

            var bandFeatures = [];
            var areaByBand = {{'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 0, '50-60': 0}};
            var prev = null;

            thresholds.forEach(function(upper) {{
                var curr = cumGeo[upper];
                if (!curr) {{
                    prev = curr || prev;
                    return;
                }}
                var band = prev ? diffSafe(curr, prev) : curr;
                if (!band) {{
                    prev = curr;
                    return;
                }}

                var lower = upper - 10;
                var label = lower + '-' + upper;
                var areaKm2 = 0;
                try {{ areaKm2 = turf.area(band) / 1000000.0; }} catch (e) {{ areaKm2 = 0; }}

                band.properties = band.properties || {{}};
                band.properties.band_label = label;
                band.properties.area_km2 = areaKm2;
                bandFeatures.push(band);
                areaByBand[label] = areaKm2;
                prev = curr;
            }});

            return {{
                geojson: {{ type: 'FeatureCollection', features: bandFeatures }},
                areaByBand: areaByBand,
            }};
        }}

        function refreshIsochrones(latlng) {{
            var rebuilt = buildDynamicIsochrones(latlng);
            isoLayer.clearLayers();
            isoLayer.addData(rebuilt.geojson);
            isoLayer.eachLayer(function(layer) {{
                var label = '50-60';
                if (layer.feature && layer.feature.properties && layer.feature.properties.band_label) {{
                    label = String(layer.feature.properties.band_label);
                }}
                if (layer.setStyle) {{
                    layer.setStyle({{
                        fillColor: colorByBand(label),
                        color: '#ffffff',
                        weight: 1.0,
                        fillOpacity: 0.55
                    }});
                }}
            }});
            updateLegendAreaMap(rebuilt.areaByBand || {{}});
        }}

        var pendingLatLng = null;
        var pendingForce = false;
        var debounceTimer = null;
        var lastComputedLatLng = null;
        var minMoveMeters = 80;
        var debounceMs = 90;

        function shouldRecompute(latlng, force) {{
            if (force) return true;
            if (!lastComputedLatLng) return true;
            return mapObj.distance(latlng, lastComputedLatLng) >= minMoveMeters;
        }}

        function scheduleRefresh(latlng, force) {{
            pendingLatLng = latlng;
            pendingForce = pendingForce || !!force;
            if (debounceTimer) clearTimeout(debounceTimer);

            debounceTimer = setTimeout(function() {{
                window.requestAnimationFrame(function() {{
                    if (!pendingLatLng) return;
                    if (!shouldRecompute(pendingLatLng, pendingForce)) return;
                    refreshIsochrones(pendingLatLng);
                    lastComputedLatLng = pendingLatLng;
                    pendingForce = false;
                }});
            }}, debounceMs);
        }}

        function bindMove() {{
            mapObj.on('mousemove', function(e) {{
                scheduleRefresh(e.latlng, false);
            }});
            mapObj.on('click', function(e) {{
                scheduleRefresh(e.latlng, true);
            }});
            scheduleRefresh(baseOrigin, true);
        }}

        if (!window.turf) {{
            var turfScript = document.createElement('script');
            turfScript.src = 'https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js';
            turfScript.onload = bindMove;
            document.head.appendChild(turfScript);
        }} else {{
            bindMove();
        }}
    }})();
    """
    m.get_root().script.add_child(folium.Element(dynamic_iso_script))

    area_lookup = {str(row["band_label"]): float(row["area_km2"]) for _, row in iso_gdf_metric.iterrows()}
    total_30 = area_lookup.get("0-10", 0.0) + area_lookup.get("10-20", 0.0) + area_lookup.get("20-30", 0.0)

    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; border: 1px solid #999; padding: 10px 12px;
                font-size: 12px; line-height: 1.3;">
    <div style="font-weight: 600; margin-bottom: 6px;">Mapa de Isócronas (intervalos de 10 min)</div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#d73027;margin-right:6px;"></span>0-10: <span id="legend-area-0_10">{a0:.3f} km²</span></div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#f46d43;margin-right:6px;"></span>10-20: <span id="legend-area-10_20">{a1:.3f} km²</span></div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#fdae61;margin-right:6px;"></span>20-30: <span id="legend-area-20_30">{a2:.3f} km²</span></div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#fee08b;margin-right:6px;"></span>30-40: <span id="legend-area-30_40">{a3:.3f} km²</span></div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#a6d96a;margin-right:6px;"></span>40-50: <span id="legend-area-40_50">{a4:.3f} km²</span></div>
            <div><span style="display:inline-block;width:12px;height:12px;background:#1a9850;margin-right:6px;"></span>50-60: <span id="legend-area-50_60">{a5:.3f} km²</span></div>
            <div style="margin-top:6px;padding-top:6px;border-top:1px solid #ddd;"><strong>Total ≤ 30 min:</strong> <span id="legend-area-total-30">{at:.3f} km²</span></div>
    </div>
    """.format(
        a0=area_lookup.get("0-10", 0.0),
        a1=area_lookup.get("10-20", 0.0),
        a2=area_lookup.get("20-30", 0.0),
        a3=area_lookup.get("30-40", 0.0),
        a4=area_lookup.get("40-50", 0.0),
        a5=area_lookup.get("50-60", 0.0),
        at=total_30,
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m

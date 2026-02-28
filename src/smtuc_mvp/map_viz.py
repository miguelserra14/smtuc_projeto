from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

from .gtfs import GTFSData


def build_map(gtfs: GTFSData, allocation: pd.DataFrame, output_html: str) -> None:
    center_lat = float(gtfs.stops["stop_lat"].mean())
    center_lon = float(gtfs.stops["stop_lon"].mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    route_peak = allocation.groupby("route_id", as_index=False)["allocated_buses"].max().rename(
        columns={"allocated_buses": "peak_buses"}
    )
    route_peak = route_peak.merge(
        allocation[["route_id", "route_short_name", "route_long_name", "is_metrobus"]]
        .drop_duplicates(subset=["route_id"]),
        on="route_id",
        how="left",
    )

    if not gtfs.shapes.empty and "shape_id" in gtfs.trips.columns:
        _add_shapes_layer(fmap, gtfs, route_peak)
    else:
        _add_stop_links_layer(fmap, gtfs, route_peak)

    _add_stops_layer(fmap, gtfs)

    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output))


def _add_shapes_layer(fmap: folium.Map, gtfs: GTFSData, route_peak: pd.DataFrame) -> None:
    trip_shapes = gtfs.trips[["route_id", "shape_id"]].dropna().drop_duplicates(subset=["route_id"])

    for _, row in route_peak.iterrows():
        route_id = row["route_id"]
        shape_match = trip_shapes[trip_shapes["route_id"] == route_id]
        if shape_match.empty:
            continue

        shape_id = shape_match.iloc[0]["shape_id"]
        shape_points = gtfs.shapes[gtfs.shapes["shape_id"] == shape_id].copy()
        if shape_points.empty:
            continue

        shape_points = shape_points.sort_values("shape_pt_sequence")
        polyline = shape_points[["shape_pt_lat", "shape_pt_lon"]].values.tolist()

        is_metrobus = bool(row.get("is_metrobus", False))
        color = "#0057b8" if is_metrobus else "#1f7a1f"
        weight = 7 if is_metrobus else 4

        folium.PolyLine(
            locations=polyline,
            color=color,
            weight=weight,
            opacity=0.8,
            tooltip=(
                f"Linha {row.get('route_short_name', route_id)} | "
                f"pico de frota: {int(row['peak_buses'])}"
            ),
        ).add_to(fmap)


def _add_stop_links_layer(fmap: folium.Map, gtfs: GTFSData, route_peak: pd.DataFrame) -> None:
    merged = gtfs.stop_times.merge(gtfs.trips[["trip_id", "route_id"]], on="trip_id", how="left")
    merged = merged.sort_values(["trip_id", "stop_sequence"])

    for route_id in route_peak["route_id"].unique():
        route_stops = merged[merged["route_id"] == route_id]
        if route_stops.empty:
            continue

        first_trip_id = route_stops["trip_id"].iloc[0]
        trip_stops = route_stops[route_stops["trip_id"] == first_trip_id]
        with_coords = trip_stops.merge(gtfs.stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
        polyline = with_coords[["stop_lat", "stop_lon"]].dropna().values.tolist()
        if len(polyline) < 2:
            continue

        line_info = route_peak[route_peak["route_id"] == route_id].iloc[0]
        is_metrobus = bool(line_info.get("is_metrobus", False))
        color = "#0057b8" if is_metrobus else "#1f7a1f"
        weight = 7 if is_metrobus else 4

        folium.PolyLine(
            locations=polyline,
            color=color,
            weight=weight,
            opacity=0.8,
            tooltip=(
                f"Linha {line_info.get('route_short_name', route_id)} | "
                f"pico de frota: {int(line_info['peak_buses'])}"
            ),
        ).add_to(fmap)


def _add_stops_layer(fmap: folium.Map, gtfs: GTFSData) -> None:
    for _, stop in gtfs.stops.iterrows():
        folium.CircleMarker(
            location=[stop["stop_lat"], stop["stop_lon"]],
            radius=2,
            color="#5b5b5b",
            fill=True,
            fill_opacity=0.7,
            popup=f"{stop.get('stop_name', '')} ({stop.get('stop_id', '')})",
        ).add_to(fmap)

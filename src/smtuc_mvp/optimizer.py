from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from ortools.sat.python import cp_model

from .gtfs import GTFSData


@dataclass
class OptimizationConfig:
    total_fleet: int = 50
    bus_capacity: int = 70
    metrobus_capacity: int = 110
    min_service_buses: int = 1
    max_buses_per_route_hour: int = 20
    demand_buffer: float = 0.05
    unmet_penalty: int = 100
    excess_penalty: int = 1
    bus_cost_per_hour: int = 8


def build_route_hour_demand(gtfs: GTFSData, demand_csv: Optional[str] = None) -> pd.DataFrame:
    if demand_csv:
        demand = pd.read_csv(demand_csv)
        required = {"route_id", "hour", "demand_pax"}
        missing = required.difference(set(demand.columns))
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"CSV de procura sem colunas obrigatórias: {missing_cols}")
        demand["hour"] = demand["hour"].astype(int)
        demand["demand_pax"] = demand["demand_pax"].astype(int)
        return demand[["route_id", "hour", "demand_pax"]]

    route_departures = _departures_per_route_hour(gtfs)
    demand = route_departures.copy()

    demand["peak_factor"] = demand["hour"].apply(lambda h: 1.35 if h in {7, 8, 9, 17, 18, 19} else 0.9)
    demand["demand_pax"] = (demand["departures"] * 38 * demand["peak_factor"]).round().astype(int)
    demand["demand_pax"] = demand["demand_pax"].clip(lower=12)

    return demand[["route_id", "hour", "demand_pax"]]


def estimate_cycle_minutes(gtfs: GTFSData) -> pd.DataFrame:
    merged = gtfs.stop_times.merge(gtfs.trips[["trip_id", "route_id"]], on="trip_id", how="left")
    trip_span = (
        merged.groupby(["route_id", "trip_id"], as_index=False)
        .agg(start=("departure_seconds", "min"), end=("arrival_seconds", "max"))
    )
    trip_span["trip_minutes"] = ((trip_span["end"] - trip_span["start"]) / 60.0).clip(lower=8)

    route_cycle = trip_span.groupby("route_id", as_index=False).agg(one_way_minutes=("trip_minutes", "median"))
    route_cycle["cycle_minutes"] = (route_cycle["one_way_minutes"] * 2.1).round().clip(lower=20)
    return route_cycle[["route_id", "cycle_minutes"]]


def optimize_allocation(gtfs: GTFSData, demand: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    cycles = estimate_cycle_minutes(gtfs)
    route_info = gtfs.routes[["route_id", "route_short_name", "route_long_name"]].copy()
    if "route_type" in gtfs.routes.columns:
        route_info["route_type"] = gtfs.routes["route_type"]
    else:
        route_info["route_type"] = 3

    table = demand.merge(cycles, on="route_id", how="left").merge(route_info, on="route_id", how="left")
    table["cycle_minutes"] = table["cycle_minutes"].fillna(50).astype(int)
    table["is_metrobus"] = table["route_short_name"].fillna("").str.upper().str.contains("M")

    model = cp_model.CpModel()

    x_vars = {}
    unmet_vars = {}
    excess_vars = {}

    for idx, row in table.iterrows():
        route_id = row["route_id"]
        hour = int(row["hour"])
        key = (route_id, hour)

        demand_pax = int(row["demand_pax"] * (1 + config.demand_buffer))
        cycle_minutes = max(1, int(row["cycle_minutes"]))
        capacity = config.metrobus_capacity if bool(row["is_metrobus"]) else config.bus_capacity
        seats_per_bus_hour = max(1, int((60 * capacity) / cycle_minutes))

        lower = config.min_service_buses if demand_pax > 0 else 0
        x = model.NewIntVar(lower, config.max_buses_per_route_hour, f"x_{route_id}_{hour}")
        unmet = model.NewIntVar(0, demand_pax * 2 + 500, f"unmet_{route_id}_{hour}")
        excess = model.NewIntVar(0, demand_pax * 2 + 500, f"excess_{route_id}_{hour}")

        offered = seats_per_bus_hour * x
        model.Add(offered + unmet >= demand_pax)
        model.Add(offered - demand_pax <= excess)

        x_vars[key] = x
        unmet_vars[key] = unmet
        excess_vars[key] = excess

    for hour in sorted(table["hour"].unique()):
        hour_routes = [x_vars[(r, int(hour))] for r in table.loc[table["hour"] == hour, "route_id"].tolist()]
        if hour_routes:
            model.Add(sum(hour_routes) <= config.total_fleet)

    model.Minimize(
        sum(config.bus_cost_per_hour * x for x in x_vars.values())
        + sum(config.unmet_penalty * u for u in unmet_vars.values())
        + sum(config.excess_penalty * e for e in excess_vars.values())
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Não foi possível encontrar solução viável para a alocação.")

    result_rows = []
    for idx, row in table.iterrows():
        key = (row["route_id"], int(row["hour"]))
        cycle_minutes = int(row["cycle_minutes"])
        capacity = config.metrobus_capacity if bool(row["is_metrobus"]) else config.bus_capacity
        seats_per_bus_hour = max(1, int((60 * capacity) / cycle_minutes))
        allocated = solver.Value(x_vars[key])
        offered = seats_per_bus_hour * allocated
        unmet = solver.Value(unmet_vars[key])

        headway = round(cycle_minutes / allocated, 1) if allocated > 0 else None
        load_factor = round(row["demand_pax"] / offered, 3) if offered > 0 else None

        result_rows.append(
            {
                "route_id": row["route_id"],
                "route_short_name": row.get("route_short_name", ""),
                "route_long_name": row.get("route_long_name", ""),
                "hour": int(row["hour"]),
                "demand_pax": int(row["demand_pax"]),
                "allocated_buses": int(allocated),
                "offered_seats": int(offered),
                "unmet_pax": int(unmet),
                "cycle_minutes": cycle_minutes,
                "headway_minutes": headway,
                "load_factor": load_factor,
                "is_metrobus": bool(row["is_metrobus"]),
            }
        )

    return pd.DataFrame(result_rows)


def summarize_kpis(allocation: pd.DataFrame) -> dict:
    total_demand = int(allocation["demand_pax"].sum())
    total_unmet = int(allocation["unmet_pax"].sum())
    served = max(0, total_demand - total_unmet)

    waits = allocation["headway_minutes"].fillna(60) / 2.0
    weighted_wait = float((waits * allocation["demand_pax"]).sum() / max(1, total_demand))

    avg_load = float(allocation["load_factor"].fillna(0).mean())

    return {
        "total_demand_pax": total_demand,
        "served_pax": served,
        "unmet_pax": total_unmet,
        "service_level": round(served / max(1, total_demand), 4),
        "avg_wait_minutes": round(weighted_wait, 2),
        "avg_load_factor": round(avg_load, 3),
        "max_hour_buses": int(allocation.groupby("hour")["allocated_buses"].sum().max()),
    }


def _departures_per_route_hour(gtfs: GTFSData) -> pd.DataFrame:
    starts = gtfs.stop_times.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id", as_index=False).first()
    starts = starts[["trip_id", "departure_seconds"]]
    starts["hour"] = (starts["departure_seconds"] // 3600).astype(int)

    by_route = starts.merge(gtfs.trips[["trip_id", "route_id"]], on="trip_id", how="left")
    departures = by_route.groupby(["route_id", "hour"], as_index=False).size().rename(columns={"size": "departures"})

    if departures.empty:
        route_ids = gtfs.routes["route_id"].unique().tolist()
        rows = [{"route_id": r, "hour": h, "departures": 1} for r in route_ids for h in range(6, 23)]
        return pd.DataFrame(rows)

    return departures

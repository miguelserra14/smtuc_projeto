from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .gtfs import load_gtfs
from .map_viz import build_map
from .optimizer import OptimizationConfig, build_route_hour_demand, optimize_allocation, summarize_kpis


@dataclass
class RunConfig:
    gtfs_dir: Optional[str]
    demand_csv: Optional[str]
    out_dir: str
    total_fleet: int
    bus_capacity: int
    metrobus_capacity: int


def run_pipeline(config: RunConfig) -> dict:
    gtfs = load_gtfs(config.gtfs_dir)

    opt_config = OptimizationConfig(
        total_fleet=config.total_fleet,
        bus_capacity=config.bus_capacity,
        metrobus_capacity=config.metrobus_capacity,
    )

    demand = build_route_hour_demand(gtfs, config.demand_csv)
    allocation = optimize_allocation(gtfs, demand, opt_config)
    kpis = summarize_kpis(allocation)

    out_path = Path(config.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    demand_file = out_path / "demand_route_hour.csv"
    allocation_file = out_path / "allocation_route_hour.csv"
    kpi_file = out_path / "kpis.json"
    map_file = out_path / "network_map.html"

    demand.to_csv(demand_file, index=False)
    allocation.to_csv(allocation_file, index=False)

    payload = {
        "run_config": asdict(config),
        "optimizer_config": asdict(opt_config),
        "kpis": kpis,
        "outputs": {
            "demand_csv": str(demand_file),
            "allocation_csv": str(allocation_file),
            "kpis_json": str(kpi_file),
            "map_html": str(map_file),
        },
    }

    with kpi_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    build_map(gtfs, allocation, str(map_file))
    return payload

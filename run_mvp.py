from __future__ import annotations

import argparse
from pathlib import Path

from src.smtuc_mvp.pipeline import RunConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVP para alocação eficiente de autocarros/metrobus com GTFS"
    )
    parser.add_argument(
        "--gtfs-dir",
        type=str,
        default=None,
        help="Diretório com arquivos GTFS (*.txt). Se omitido, usa dados sintéticos.",
    )
    parser.add_argument(
        "--demand-csv",
        type=str,
        default=None,
        help="CSV opcional de procura com colunas: route_id,hour,demand_pax",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Diretório de saída para relatórios e mapa",
    )
    parser.add_argument("--fleet", type=int, default=50, help="Frota total disponível")
    parser.add_argument("--bus-capacity", type=int, default=70, help="Capacidade média dos autocarros")
    parser.add_argument("--metrobus-capacity", type=int, default=110, help="Capacidade média do metrobus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunConfig(
        gtfs_dir=args.gtfs_dir,
        demand_csv=args.demand_csv,
        out_dir=args.out_dir,
        total_fleet=args.fleet,
        bus_capacity=args.bus_capacity,
        metrobus_capacity=args.metrobus_capacity,
    )

    result = run_pipeline(config)

    kpis = result["kpis"]
    print("=== Resultado MVP de Alocação ===")
    print(f"Nível de serviço: {kpis['service_level'] * 100:.1f}%")
    print(f"Espera média estimada: {kpis['avg_wait_minutes']:.1f} min")
    print(f"Lotação média: {kpis['avg_load_factor']:.2f}")
    print(f"Pico de autocarros em operação: {kpis['max_hour_buses']}")
    print("Saídas:")
    for key, path in result["outputs"].items():
        print(f"- {key}: {Path(path)}")


if __name__ == "__main__":
    main()

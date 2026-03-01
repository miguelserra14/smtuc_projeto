from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from smtuc_mvp.gtfs import load_gtfs
from smtuc_mvp.gtfs_probe import (
    _build_parser,
    casa,
    compare_nearest_network,
    find_direct_options,
    nearest_stop_for_dataset,
    nearest_stop_to_work,
    suggest_current_commute_options,
    suggest_random_commute_options,
    trabalho,
)

DATASETS = ["smtuc", "metrobus"]


def _dataset_dir(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / dataset


def _require_dataset(dataset: str) -> Path:
    d = _dataset_dir(dataset)
    required = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    if not d.exists() or any(not (d / f).exists() for f in required):
        pytest.skip(f"Dataset GTFS inválido/incompleto: {d}")
    return d


def _pick_valid_day(gtfs) -> date:
    if gtfs.calendar.empty or not {"start_date", "end_date"}.issubset(gtfs.calendar.columns):
        return date.today()

    for _, row in gtfs.calendar.iterrows():
        start = datetime.strptime(str(int(row["start_date"])), "%Y%m%d").date()
        end = datetime.strptime(str(int(row["end_date"])), "%Y%m%d").date()
        d = start
        while d <= min(end, start + timedelta(days=14)):
            if int(row.get(d.strftime("%A").lower(), 0)) == 1:
                return d
            d += timedelta(days=1)

    return date.today()


@pytest.mark.integration
@pytest.mark.parametrize("dataset", DATASETS)
def test_probe_nearest_stop_for_dataset_smoke(dataset: str) -> None:
    _require_dataset(dataset)
    result = nearest_stop_for_dataset(dataset, casa[0], casa[1])
    assert result.dataset == dataset
    assert isinstance(result.stop_id, str) and result.stop_id
    assert isinstance(result.stop_name, str) and result.stop_name
    assert result.distance_m >= 0


@pytest.mark.integration
def test_probe_compare_nearest_network_and_work_smoke() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    smtuc, metrobus, winner = compare_nearest_network(casa[0], casa[1])
    assert winner in {"smtuc", "metrobus"}
    assert smtuc.distance_m >= 0
    assert metrobus.distance_m >= 0

    work_best = nearest_stop_to_work()
    assert work_best.dataset in {"smtuc", "metrobus"}
    assert work_best.distance_m >= 0


@pytest.mark.integration
@pytest.mark.parametrize("dataset", DATASETS)
def test_probe_find_direct_options_smoke(dataset: str) -> None:
    gtfs = load_gtfs(str(_require_dataset(dataset)))

    trip_id = str(gtfs.trips.iloc[0]["trip_id"])
    st_trip = gtfs.stop_times[gtfs.stop_times["trip_id"].astype(str) == trip_id].sort_values("stop_sequence")
    assert len(st_trip) >= 2

    origin = str(st_trip.iloc[0]["stop_id"])
    dest = str(st_trip.iloc[1]["stop_id"])
    day_str = _pick_valid_day(gtfs).strftime("%Y-%m-%d")

    out = find_direct_options(dataset=dataset, origin_ref=origin, dest_ref=dest, day_str=day_str, time_str="00:00:00")

    assert isinstance(out, pd.DataFrame)
    if not out.empty:
        required_cols = {"route_id", "trip_id", "origin_dep_time", "dest_arr_time", "duration_min"}
        assert required_cols.issubset(set(out.columns))


@pytest.mark.integration
def test_probe_commute_functions_smoke() -> None:
    _require_dataset("smtuc")
    _require_dataset("metrobus")

    out_now = suggest_current_commute_options(casa[0], casa[1], trabalho[0], trabalho[1], limit=2)
    assert isinstance(out_now, pd.DataFrame)
    assert len(out_now) <= 2, f"commute-now devolveu {len(out_now)} linhas (limit=2)"

    out_random = suggest_random_commute_options(casa[0], casa[1], trabalho[0], trabalho[1], limit=2, tries=3)
    assert isinstance(out_random, pd.DataFrame)
    assert len(out_random) <= 2, f"commute-random devolveu {len(out_random)} linhas (limit=2)"


@pytest.mark.integration
def test_probe_parser_aliases() -> None:
    parser = _build_parser()

    args_random = parser.parse_args(["random", "--tries", "1"])
    assert args_random.cmd in {"commute-random", "random"}
    assert hasattr(args_random, "handler")

    args_now = parser.parse_args(["now", "--limit", "1"])
    assert args_now.cmd in {"commute-now", "now"}
    assert hasattr(args_now, "handler")

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import pytest

from gtfs_processing.gtfs import extract_or_copy_gtfs, load_gtfs
DATASETS = ["smtuc", "metrobus"]

def _dataset_dir(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    # tenta primeiro data/gtfs/<dataset>, depois data/<dataset>
    p1 = root / "data" / dataset
    p2 = root / "data" / dataset
    if p1.exists():
        return p1
    return p2


def _require_dataset(dataset: str) -> Path:
    d = _dataset_dir(dataset)
    required = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    if not d.exists() or any(not (d / f).exists() for f in required):
        pytest.skip(f"Dataset GTFS inválido/incompleto: {d}")
    return d


def _hash_txt(folder: Path) -> dict[str, str]:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.glob("*.txt"))}


def _to_seconds(hhmmss: str) -> int:
    h, m, s = map(int, hhmmss.split(":"))
    return h * 3600 + m * 60 + s


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


def _active_services(gtfs, day: date) -> set[str]:
    if gtfs.calendar.empty or "service_id" not in gtfs.calendar.columns:
        return set(gtfs.trips["service_id"].dropna().astype(str).unique())
    cal = gtfs.calendar.copy()
    ymd = int(day.strftime("%Y%m%d"))
    wd = day.strftime("%A").lower()
    base = cal[(cal.get(wd, 0) == 1) & (cal["start_date"] <= ymd) & (cal["end_date"] >= ymd)]["service_id"]
    return set(base.astype(str).tolist())


def _direct_options(gtfs, origin: str, dest: str, day: date, after: str) -> pd.DataFrame:
    trips = gtfs.trips.copy()
    sids = _active_services(gtfs, day)
    if sids and "service_id" in trips.columns:
        trips = trips[trips["service_id"].astype(str).isin(sids)]

    st = gtfs.stop_times.copy()
    o = st.rename(columns={"stop_id": "origin_stop_id", "stop_sequence": "origin_seq", "departure_seconds": "origin_dep_s"})
    d = st.rename(columns={"stop_id": "dest_stop_id", "stop_sequence": "dest_seq", "arrival_seconds": "dest_arr_s"})
    m = o.merge(d[["trip_id", "dest_stop_id", "dest_seq", "dest_arr_s"]], on="trip_id").merge(trips[["trip_id", "route_id"]], on="trip_id")

    return m[
        (m["origin_stop_id"].astype(str) == origin)
        & (m["dest_stop_id"].astype(str) == dest)
        & (m["origin_seq"] < m["dest_seq"])
        & (m["origin_dep_s"] >= _to_seconds(after))
    ]


def _stop_name_map(gtfs) -> dict[str, str]:
    if gtfs.stops.empty or "stop_id" not in gtfs.stops.columns:
        return {}
    name_col = "stop_name" if "stop_name" in gtfs.stops.columns else "stop_id"
    return (
        gtfs.stops[["stop_id", name_col]]
        .dropna()
        .assign(stop_id=lambda d: d["stop_id"].astype(str))
        .drop_duplicates("stop_id")
        .set_index("stop_id")[name_col]
        .astype(str)
        .to_dict()
    )


def _candidate_days(gtfs) -> list[date]:
    days: list[date] = [_pick_valid_day(gtfs)]

    if gtfs.calendar.empty or not {"start_date", "end_date"}.issubset(gtfs.calendar.columns):
        return days

    for _, row in gtfs.calendar.head(30).iterrows():
        start = datetime.strptime(str(int(row["start_date"])), "%Y%m%d").date()
        end = datetime.strptime(str(int(row["end_date"])), "%Y%m%d").date()
        d = start
        while d <= min(end, start + timedelta(days=14)):
            if int(row.get(d.strftime("%A").lower(), 0)) == 1:
                days.append(d)
                break
            d += timedelta(days=1)

    uniq: list[date] = []
    seen: set[date] = set()
    for d in days:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _find_case_with_direct_options(gtfs, after: str = "00:00:00") -> tuple[date, str, str, pd.DataFrame] | None:
    candidate_days = _candidate_days(gtfs)

    for day in candidate_days:
        active_sids = _active_services(gtfs, day)
        trips = gtfs.trips.copy()
        if active_sids and "service_id" in trips.columns:
            trips = trips[trips["service_id"].astype(str).isin(active_sids)]

        for trip_id in trips["trip_id"].astype(str).head(500):
            st_trip = gtfs.stop_times[gtfs.stop_times["trip_id"].astype(str) == trip_id].sort_values("stop_sequence")
            if len(st_trip) < 2:
                continue

            origin = str(st_trip.iloc[0]["stop_id"])
            dest = str(st_trip.iloc[1]["stop_id"])
            options = _direct_options(gtfs, origin, dest, day, after)
            if not options.empty:
                return day, origin, dest, options

    return None

@pytest.mark.integration
@pytest.mark.parametrize("dataset", DATASETS)
def test_01_extract_readonly(tmp_path: Path, dataset: str) -> None:
    src = _require_dataset(dataset)
    before = _hash_txt(src)
    dst = tmp_path / f"gtfs_copy_{dataset}"
    extract_or_copy_gtfs(src, dst)
    assert (dst / "routes.txt").exists()
    assert (dst / "trips.txt").exists()
    assert (dst / "stops.txt").exists()
    assert (dst / "stop_times.txt").exists()
    assert before == _hash_txt(src)


@pytest.mark.integration
@pytest.mark.parametrize("dataset", DATASETS)
def test_02_display_por_linha(dataset: str) -> None:
    gtfs = load_gtfs(str(_require_dataset(dataset)))
    by_line = gtfs.trips.groupby("route_id", as_index=False).agg(total_trips=("trip_id", "nunique"))
    print(f"\n[{dataset}] Top linhas:\n", by_line.sort_values("total_trips", ascending=False).head(5).to_string(index=False))
    assert not by_line.empty


@pytest.mark.integration
@pytest.mark.parametrize("dataset", DATASETS)
def test_03_display_por_paragem(dataset: str) -> None:
    gtfs = load_gtfs(str(_require_dataset(dataset)))
    by_stop = gtfs.stop_times.groupby("stop_id", as_index=False).agg(total_passagens=("trip_id", "count"))

    stop_names = _stop_name_map(gtfs)
    by_stop_print = by_stop.copy()
    by_stop_print["paragem"] = by_stop_print["stop_id"].astype(str).map(stop_names).fillna(by_stop_print["stop_id"].astype(str))

    print(f"\n[{dataset}] Top paragens:")
    print(
        by_stop_print[["paragem", "total_passagens"]]
        .sort_values("total_passagens", ascending=False)
        .head(5)
        .to_string(index=False)
    )

    assert not by_stop.empty


@pytest.mark.integration
@pytest.mark.parametrize("dataset", DATASETS)
def test_04_rota_a_b_por_hora_dia(dataset: str) -> None:
    gtfs = load_gtfs(str(_require_dataset(dataset)))
    case = _find_case_with_direct_options(gtfs, after="00:00:00")
    if case is None:
        pytest.skip(f"Sem caso A->B direto com serviço ativo para dataset: {dataset}")

    day, origin, dest, options = case

    stop_names = _stop_name_map(gtfs)
    origin_name = stop_names.get(origin, origin)
    dest_name = stop_names.get(dest, dest)

    options_print = options.copy()
    if "origin_stop_id" in options_print.columns:
        options_print["origem"] = options_print["origin_stop_id"].astype(str).map(stop_names).fillna(options_print["origin_stop_id"].astype(str))
    if "dest_stop_id" in options_print.columns:
        options_print["destino"] = options_print["dest_stop_id"].astype(str).map(stop_names).fillna(options_print["dest_stop_id"].astype(str))

    print(f"\n[{dataset}] A->B: {origin_name} -> {dest_name} | {day}")
    cols = [c for c in ["route_id", "trip_id", "origem", "destino", "origin_dep_s", "dest_arr_s"] if c in options_print.columns]
    print(options_print[cols].head(5).to_string(index=False))

    assert not options.empty, f"Sem opções diretas para dataset={dataset} no caso selecionado"
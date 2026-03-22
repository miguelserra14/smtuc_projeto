from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Fonte única para o projeto inteiro
REQUIRED_GTFS = ("routes", "trips", "stop_times", "stops")
OPTIONAL_GTFS = ("shapes", "calendar", "calendar_dates")

REQUIRED_FILES = {f"{name}.txt" for name in REQUIRED_GTFS}
OPTIONAL_FILES = {f"{name}.txt" for name in OPTIONAL_GTFS}
_MISSING_OPTIONAL_WARNED: set[tuple[str, str]] = set()


@dataclass
class GTFSData:
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    stops: pd.DataFrame
    shapes: pd.DataFrame
    calendar: pd.DataFrame
    calendar_dates: pd.DataFrame


def _read_csv(source: Path, stem: str) -> pd.DataFrame:
    file_path = source / f"{stem}.txt"
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _to_seconds(hhmmss: str) -> int:
    if not isinstance(hhmmss, str) or ":" not in hhmmss:
        return 0
    parts = hhmmss.split(":")
    if len(parts) != 3:
        return 0
    h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
    return h * 3600 + m * 60 + s


def _looks_like_gtfs_folder(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and all((folder / f).exists() for f in REQUIRED_FILES)


def _find_gtfs_dir(dataset: str, root: Path) -> Path:
    candidates = [
        root / "data" / dataset,  # ex.: data/metrobus
    ]

    for base in candidates:
        if _looks_like_gtfs_folder(base):
            return base
        if base.exists() and base.is_dir():
            for sub in sorted(p for p in base.rglob("*") if p.is_dir()):
                if _looks_like_gtfs_folder(sub):
                    return sub

    raise FileNotFoundError(f"Dataset GTFS não encontrado/incompleto: {dataset}")


def _resolve_source_dir(source_dir: Optional[str], dataset: str) -> Path:
    root = Path(__file__).resolve().parents[2]  # raiz do projeto

    if source_dir:
        p = Path(source_dir)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not _looks_like_gtfs_folder(p):
            raise FileNotFoundError(f"Diretório GTFS inválido: {p}")
        return p

    return _find_gtfs_dir(dataset, root)


def load_gtfs(source_dir: Optional[str] = None, dataset: str = "nd") -> GTFSData:
    source = _resolve_source_dir(source_dir=source_dir, dataset=dataset)

    frames: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_GTFS + OPTIONAL_GTFS:
        frames[name] = _read_csv(source, name)

    missing = [name for name in REQUIRED_GTFS if frames[name].empty]
    if missing:
        missing_list = ", ".join(f"{name}.txt" for name in missing)
        raise ValueError(f"Arquivos GTFS obrigatórios em falta: {missing_list}")

    missing = [name for name in OPTIONAL_GTFS if frames[name].empty]
    if missing:
        missing_list = ", ".join(f"{name}.txt" for name in missing)
        warning_key = (str(source), missing_list)
        if warning_key not in _MISSING_OPTIONAL_WARNED:
            print(f"Arquivos GTFS opcionais em falta: {missing_list}")
            _MISSING_OPTIONAL_WARNED.add(warning_key)

    st = frames["stop_times"]
    for col in ("arrival_time", "departure_time"):
        if col not in st.columns:
            raise ValueError(f"stop_times.txt sem coluna obrigatória: {col}")

    st["arrival_seconds"] = st["arrival_time"].apply(_to_seconds)
    st["departure_seconds"] = st["departure_time"].apply(_to_seconds)

    return GTFSData(
        routes=frames["routes"],
        trips=frames["trips"],
        stop_times=st,
        stops=frames["stops"],
        shapes=frames["shapes"],
        calendar=frames["calendar"],
        calendar_dates=frames["calendar_dates"],
    )


def extract_or_copy_gtfs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            by_name = {Path(m).name: m for m in zf.namelist()}
            missing = REQUIRED_FILES - set(by_name.keys())
            if missing:
                raise FileNotFoundError(f"GTFS incompleto. Faltam: {', '.join(sorted(missing))}")

            for name in sorted(REQUIRED_FILES | OPTIONAL_FILES):
                if name in by_name:
                    with zf.open(by_name[name]) as src_f, (dst / name).open("wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)

    elif src.is_dir():
        files = {p.name for p in src.glob("*.txt")}
        missing = REQUIRED_FILES - files
        if missing:
            raise FileNotFoundError(f"GTFS incompleto. Faltam: {', '.join(sorted(missing))}")

        for name in sorted(REQUIRED_FILES | OPTIONAL_FILES):
            fp = src / name
            if fp.exists():
                shutil.copy2(fp, dst / name)
    else:
        raise ValueError("Origem inválida. Usa uma pasta GTFS ou um ficheiro .zip")


def _main() -> None:
    parser = argparse.ArgumentParser(description="GTFS utilitário")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_int = sub.add_parser("integrate", help="Integrar GTFS (.zip/pasta) para data/")
    p_int.add_argument("--source", required=True, help="Caminho para GTFS (.zip ou pasta)")
    p_int.add_argument("--target", default=r"data", help="Pasta destino")

    p_ins = sub.add_parser("inspect", help="Carregar GTFS já existente em data/")
    p_ins.add_argument("--dataset", default="smtuc", help="smtuc | metrobus")
    p_ins.add_argument("--source-dir", default=None, help="Override de diretório GTFS")

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    if args.cmd == "integrate":
        source = Path(args.source).expanduser().resolve()
        target = (root / args.target).resolve()
        extract_or_copy_gtfs(source, target)
        print(f"GTFS integrado em: {target}")
        return

    gtfs = load_gtfs(source_dir=args.source_dir, dataset=args.dataset)
    print(f"dataset={args.dataset}")
    print(f"routes={len(gtfs.routes)} trips={len(gtfs.trips)} stops={len(gtfs.stops)} stop_times={len(gtfs.stop_times)}")

if __name__ == "__main__":
    _main()

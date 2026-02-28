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


def load_gtfs(source_dir: Optional[str] = r"data\gtfs\smtuc") -> GTFSData:
    if not source_dir:
        raise ValueError("source_dir é obrigatório para carregar GTFS real.")

    source = Path(source_dir)
    if not source.is_absolute():
        source = Path.cwd() / source
    source = source.resolve()

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Diretório GTFS inválido: {source}")

    frames: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_GTFS + OPTIONAL_GTFS:
        frames[name] = _read_csv(source, name)

    missing = [name for name in REQUIRED_GTFS if frames[name].empty]
    if missing:
        missing_list = ", ".join(f"{name}.txt" for name in missing)
        raise ValueError(f"Arquivos GTFS obrigatórios em falta: {missing_list}")

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
    parser = argparse.ArgumentParser(description="Integrar GTFS SMTUC no projeto")
    parser.add_argument("--source", required=True, help="Caminho para GTFS (.zip ou pasta)")
    parser.add_argument("--target", default=r"data\gtfs\smtuc", help="Pasta destino")
    args = parser.parse_args()

    # src/smtuc_mvp/gtfs.py -> raiz = parents[2]
    root = Path(__file__).resolve().parents[2]
    source = Path(args.source).expanduser().resolve()
    target = (root / args.target).resolve()

    extract_or_copy_gtfs(source, target)
    print(f"GTFS integrado em: {target}")


if __name__ == "__main__":
    _main()

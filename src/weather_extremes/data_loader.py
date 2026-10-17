"""NOAA GHCN-Daily download and parsing utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"

ELEMENT_UNITS = {
    "TMAX": "celsius_tenths",
    "TMIN": "celsius_tenths",
    "TAVG": "celsius_tenths",
    "TOBS": "celsius_tenths",
    "PRCP": "mm_tenths",
    "SNOW": "mm",
    "SNWD": "mm",
}


@dataclass
class StationInfo:
    station_id: str
    name: str
    state: str
    latitude: float
    longitude: float
    elevation: float


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _download(url: str, dest: str, force: bool = False, timeout: int = 120) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
        return dest
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            if chunk:
                handle.write(chunk)
    os.replace(tmp, dest)
    return dest


def download_station_metadata(raw_dir: str, base_url: str = DEFAULT_BASE_URL, force: bool = False) -> str:
    return _download(f"{base_url}/ghcnd-stations.txt", os.path.join(raw_dir, "ghcnd-stations.txt"), force)


def download_station_dly(station_id: str, raw_dir: str, base_url: str = DEFAULT_BASE_URL, force: bool = False) -> str:
    return _download(f"{base_url}/all/{station_id}.dly", os.path.join(raw_dir, f"{station_id}.dly"), force)


def parse_station_metadata(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            station_id = line[0:11].strip()
            if not station_id:
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "latitude": _to_float(line[12:20]),
                    "longitude": _to_float(line[21:30]),
                    "elevation": _to_float(line[31:37]),
                    "state": line[38:40].strip(),
                    "name": line[41:71].strip(),
                    "gsn": line[72:75].strip(),
                    "wmo_id": line[76:81].strip(),
                }
            )
    return pd.DataFrame(rows)


def get_station_info(station_id: str, metadata: pd.DataFrame) -> StationInfo:
    match = metadata[metadata["station_id"] == station_id]
    if match.empty:
        raise KeyError(f"station {station_id!r} not found in GHCN metadata")
    row = match.iloc[0]
    return StationInfo(
        station_id=station_id,
        name=str(row["name"]),
        state=str(row["state"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        elevation=float(row["elevation"]),
    )


def parse_dly(path: str, station_id: str | None = None) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if len(line) < 21:
                continue
            current_id = line[0:11].strip()
            if station_id and current_id != station_id:
                continue
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip()
            for day in range(1, 32):
                base = 21 + (day - 1) * 8
                raw_value = line[base : base + 5].strip()
                if not raw_value or raw_value == "-9999":
                    continue
                try:
                    value = int(raw_value)
                except ValueError:
                    continue
                try:
                    date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                records.append(
                    (
                        date,
                        element,
                        value,
                        line[base + 5 : base + 6].strip() or None,
                        line[base + 6 : base + 7].strip() or None,
                        line[base + 7 : base + 8].strip() or None,
                    )
                )
    return pd.DataFrame(records, columns=["date", "element", "value", "mflag", "qflag", "sflag"])


def values_to_wide(dly: pd.DataFrame, elements: list[str], units: str = "metric") -> pd.DataFrame:
    frame = dly[dly["element"].isin(elements)].copy()
    frame = frame[frame["qflag"].isna()]
    if frame.empty:
        return pd.DataFrame(columns=elements, dtype="float64")
    wide = frame.pivot_table(index="date", columns="element", values="value", aggfunc="mean")
    wide = wide.reindex(columns=elements)
    wide = wide.astype("float64")
    for element in elements:
        kind = ELEMENT_UNITS.get(element)
        if kind == "celsius_tenths":
            wide[element] = wide[element] / 10.0
        elif kind == "mm_tenths":
            wide[element] = wide[element] / 10.0
    if units == "imperial":
        for element in elements:
            if element in ("TMAX", "TMIN", "TAVG", "TOBS"):
                wide[element] = wide[element] * 9.0 / 5.0 + 32.0
            elif element in ("PRCP", "SNOW", "SNWD"):
                wide[element] = wide[element] / 25.4
    wide.index.name = "date"
    return wide


def load_station_data(
    station_id: str,
    start_date: str,
    end_date: str,
    elements: list[str],
    raw_dir: str,
    base_url: str = DEFAULT_BASE_URL,
    units: str = "metric",
    download: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    dly_path = os.path.join(raw_dir, f"{station_id}.dly")
    if download:
        download_station_dly(station_id, raw_dir, base_url=base_url, force=force_download)
    if not os.path.exists(dly_path):
        raise FileNotFoundError(f"GHCN-Daily file not found: {dly_path}")
    dly = parse_dly(dly_path, station_id=station_id)
    wide = values_to_wide(dly, elements, units=units)
    if wide.empty:
        return wide
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    wide = wide.reindex(full_index)
    wide.index.name = "date"
    wide = wide.rename_axis("date")
    wide = wide.rename(columns={c: c for c in wide.columns})
    return wide

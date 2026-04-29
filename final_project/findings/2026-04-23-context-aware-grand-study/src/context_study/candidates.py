from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

CANDIDATE_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "candidate_universes"

CANDIDATE_SET_FILES = {
    "sp100": "sp100_wikipedia.csv",
    "sp500": "sp500_wikipedia.csv",
}

NAMED_CANDIDATE_SETS = {
    "liquid_us_equity_100": ("sp100", 100),
    "liquid_us_equity_250": ("sp500", 250),
    "liquid_us_equity_500": ("sp500", 500),
}


def load_candidate_table(name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a candidate symbol table from a local CSV bundle."""
    if name not in CANDIDATE_SET_FILES:
        supported = sorted(CANDIDATE_SET_FILES)
        raise ValueError(f"Unknown candidate table {name!r}; supported={supported}")

    base_dir = Path(data_dir) if data_dir is not None else CANDIDATE_DATA_DIR
    path = base_dir / CANDIDATE_SET_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Candidate table not found: {path}")

    table = pd.read_csv(path)
    required = {"symbol", "name"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Candidate table missing required columns: {sorted(missing)}")

    table = table[["symbol", "name"]].copy()
    table["symbol"] = table["symbol"].astype(str).str.strip().str.upper()
    table = table.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return table


def load_candidate_symbols(name: str, data_dir: str | Path | None = None) -> list[str]:
    """Backward-compatible symbol-only loader for a stored candidate table."""
    return load_candidate_table(name, data_dir=data_dir)["symbol"].tolist()


def available_candidate_sets() -> list[str]:
    """Return the available named candidate workflows."""
    return sorted(NAMED_CANDIDATE_SETS)


def get_candidate_symbols(
    universe_name: str,
    data_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[str]:
    """Return candidate symbols for a named large-universe workflow."""
    if universe_name not in NAMED_CANDIDATE_SETS:
        supported = sorted(NAMED_CANDIDATE_SETS)
        raise ValueError(f"Unknown universe_name {universe_name!r}; supported={supported}")

    table_name, default_limit = NAMED_CANDIDATE_SETS[universe_name]
    table = load_candidate_table(table_name, data_dir=data_dir)
    effective_limit = default_limit if limit is None else min(limit, len(table))
    return table["symbol"].head(effective_limit).tolist()


def write_candidate_symbols(
    symbols: Iterable[str],
    path: str | Path,
) -> Path:
    """Persist a simple one-column candidate symbol file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = pd.DataFrame({"symbol": [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]})
    cleaned = cleaned.drop_duplicates(subset=["symbol"])
    cleaned.to_csv(output_path, index=False)
    return output_path

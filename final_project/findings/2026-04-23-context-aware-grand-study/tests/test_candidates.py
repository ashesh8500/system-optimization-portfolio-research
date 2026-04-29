from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.candidates import (
    CANDIDATE_SET_FILES,
    NAMED_CANDIDATE_SETS,
    get_candidate_symbols,
    load_candidate_table,
    write_candidate_symbols,
)


def test_load_candidate_table_has_expected_columns_and_rows() -> None:
    table = load_candidate_table("sp100")
    assert list(table.columns) == ["symbol", "name"]
    assert len(table) >= 100
    assert table["symbol"].is_unique


def test_named_candidate_sets_return_expected_sizes() -> None:
    assert len(get_candidate_symbols("liquid_us_equity_100")) == 100
    assert len(get_candidate_symbols("liquid_us_equity_250")) == 250
    assert len(get_candidate_symbols("liquid_us_equity_500")) == 500


def test_write_candidate_symbols_normalizes_and_deduplicates(tmp_path: Path) -> None:
    output = write_candidate_symbols(["aapl", " AAPL ", "msft", ""], tmp_path / "candidates.csv")
    saved = pd.read_csv(output)
    assert saved["symbol"].tolist() == ["AAPL", "MSFT"]


def test_unknown_candidate_name_raises() -> None:
    with pytest.raises(ValueError):
        load_candidate_table("unknown")
    with pytest.raises(ValueError):
        get_candidate_symbols("unknown")

import json

import pandas as pd
import pytest

from scouting.cli import main
from scouting.pipeline import refresh

TODAY = pd.Timestamp("2026-10-30")


@pytest.fixture(scope="module")
def data_csv(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    path, _ = refresh("synthetic", out_dir=out, today=TODAY)
    return str(path)


def _a_player(data_csv, pos="MID"):
    df = pd.read_csv(data_csv)
    return df[df.position_group == pos].sort_values("minutes").iloc[-1]["name"]


def test_recommend_prints_banner_query_and_rows(data_csv, capsys):
    assert main(["recommend", _a_player(data_csv), "--data", data_csv, "-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DEMO DATA" in out and "Query:" in out and "difficulty to buy" in out
    assert "similarity" in out and out.count("Moderate") + out.count("Easy") + out.count("Hard") >= 5


def test_recommend_csv_and_group_by(data_csv, capsys):
    name = _a_player(data_csv)
    assert main(["recommend", name, "--data", data_csv, "-k", "4", "--format", "csv"]) == 0
    csv_lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith(("1,", "2,", "3,", "4,"))]
    assert len(csv_lines) == 4
    assert main(["recommend", name, "--data", data_csv, "--group-by", "price_band", "--per-group", "2"]) == 0
    assert "--- price_band:" in capsys.readouterr().out


def test_recommend_filters_via_cli(data_csv, capsys):
    name = _a_player(data_csv, "FWD")
    rc = main(["recommend", name, "--data", data_csv, "-k", "20", "--max-age", "25",
               "--price-band", "Budget", "--price-band", "Mid-range", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    rows = json.loads(out[out.index("[\n"):])          # skip the banner/query lines before the JSON
    assert rows, "expected at least one filtered recommendation"
    assert all(r["age"] <= 25 and r["price_band"] in ("Budget", "Mid-range") for r in rows)


def test_unknown_player_exits_with_a_helpful_message(data_csv, capsys):
    assert main(["recommend", "Totally Unknown", "--data", data_csv]) == 2
    assert "No player matching" in capsys.readouterr().err


def test_failed_fetch_is_a_clean_error_and_exit_code_3(monkeypatch, capsys):
    from scouting import cli
    from scouting.sources._http import HttpError

    def boom(*a, **k):
        raise HttpError(0, "no route", "https://example.test")

    monkeypatch.setattr(cli, "refresh", boom)
    assert main(["refresh", "--source", "fpl"]) == 3
    err = capsys.readouterr().err
    assert "could not fetch data" in err and "left untouched" in err


def test_search_and_profile(data_csv, capsys):
    name = _a_player(data_csv)
    assert main(["search", name.split()[-1], "--data", data_csv]) == 0
    assert name in capsys.readouterr().out
    assert main(["profile", name, "--data", data_csv]) == 0
    assert "percentile" in capsys.readouterr().out

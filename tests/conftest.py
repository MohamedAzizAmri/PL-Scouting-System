import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scouting.engine import ScoutingEngine  # noqa: E402
from scouting.sources import synthetic  # noqa: E402

TODAY = pd.Timestamp("2026-10-30")


@pytest.fixture(scope="session")
def synth():
    return synthetic.generate(7, today=TODAY)


@pytest.fixture(scope="session")
def engine(synth):
    return ScoutingEngine(synth.players.drop(columns=["archetype"]), today=TODAY)


@pytest.fixture(scope="session")
def archetypes(synth):
    return synth.players.set_index("player_id")["archetype"]


@pytest.fixture(autouse=True)
def _no_real_role_model(tmp_path, monkeypatch):
    """Pipeline tests must never pick up a model trained on real data that sits in models/."""
    from scouting import config
    monkeypatch.setattr(config, "ROLE_MODEL", tmp_path / "no-model.joblib")

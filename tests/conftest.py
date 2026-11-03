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

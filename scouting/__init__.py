"""Premier League player scouting: find players with a similar profile."""
from .engine import AmbiguousPlayer, InsufficientData, PlayerNotFound, ScoutingEngine

__all__ = ["ScoutingEngine", "PlayerNotFound", "AmbiguousPlayer", "InsufficientData"]
__version__ = "0.1.0"

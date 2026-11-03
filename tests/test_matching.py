import pandas as pd

from scouting.matching import club_similarity, match_players, normalize_name


def test_normalize_strips_accents_and_punctuation():
    assert normalize_name("Martin Ødegaard") == "martin odegaard"
    assert normalize_name("N'Golo  Kanté") == "n golo kante"
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""


def test_club_aliases():
    assert club_similarity("Man Utd", "Manchester United Football Club") == 1.0
    assert club_similarity("Spurs", "Tottenham Hotspur FC") == 1.0
    assert club_similarity("Man Utd", "Manchester City FC") < 0.6


def _left(rows):
    return pd.DataFrame(rows, columns=["name", "web_name", "club"])


def _right(rows):
    return pd.DataFrame(rows, columns=["name", "club"])


def test_exact_match_ignores_accents():
    res = match_players(_left([("Martin Odegaard", "Odegaard", "Arsenal")]),
                        _right([("Martin Ødegaard", "Arsenal FC")]), right_club_col="club")
    assert len(res.pairs) == 1 and res.rate == 1.0


def test_token_subset_needs_matching_club():
    left = _left([("Bruno Fernandes", "B.Fernandes", "Man Utd")])
    right = _right([("Bruno Miguel Borges Fernandes", "Manchester United Football Club"),
                    ("Bruno Fernandes Jr", "Manchester City FC")])
    res = match_players(left, right, right_club_col="club")
    assert res.pairs.iloc[0]["right_index"] == 0          # same club wins, namesake ignored
    assert res.pairs.iloc[0]["method"] == "token-subset"


def test_single_token_name_never_fuzzy_matches():
    left = _left([("Gabriel", "Gabriel", "Arsenal")])
    right = _right([("Gabriel Jesus", "Arsenal FC"), ("Gabriel Martinelli", "Arsenal FC")])
    res = match_players(left, right, right_club_col="club")
    assert res.pairs.empty and res.unmatched_left == [0]


def test_duplicate_exact_names_broken_by_club():
    left = _left([("Ben Johnson", "Johnson", "West Ham")])
    right = _right([("Ben Johnson", "Ipswich Town FC"), ("Ben Johnson", "West Ham United FC")])
    res = match_players(left, right, right_club_col="club")
    assert res.pairs.iloc[0]["right_index"] == 1


def test_unresolvable_duplicates_are_left_unmatched_not_guessed():
    left = _left([("Ben Johnson", "Johnson", "Chelsea")])
    right = _right([("Ben Johnson", "Ipswich Town FC"), ("Ben Johnson", "Brentford FC")])
    res = match_players(left, right, right_club_col="club")
    assert res.pairs.empty


def test_right_row_is_never_used_twice():
    left = _left([("Sam Lee", "Lee", "Leeds"), ("Sam Lee", "Lee", "Leeds")])
    right = _right([("Sam Lee", "Leeds United FC")])
    res = match_players(left, right, right_club_col="club")
    assert len(res.pairs) == 1 and len(res.unmatched_left) == 1


# --- abbreviated names (API-Football style: "W. Fish") ---------------------------------------
def test_abbreviated_name_matches_initial_plus_surname_same_club():
    left = _left([("William Fish", "Fish", "Man Utd"), ("Bruno Miguel Borges Fernandes", "B.Fernandes", "Man Utd"),
                  ("Nathan O'Reilly", "O'Reilly", "Man City"), ("Eli Junior Kroupi", "Kroupi", "Bournemouth")])
    right = _right([("W. Fish", "Manchester United"), ("B. Fernandes", "Manchester United"),
                    ("N. O'Reilly", "Manchester City"), ("E. J. Kroupi", "AFC Bournemouth")])
    res = match_players(left, right, right_club_col="club")
    assert len(res.pairs) == 4 and set(res.pairs["method"]) == {"initial+surname"}


def test_abbreviated_name_with_wrong_initial_or_club_is_rejected():
    left = _left([("Zoe Fish", "Fish", "Man Utd"), ("William Fish", "Fish", "Chelsea")])
    right = _right([("W. Fish", "Manchester United")])
    res = match_players(left, right, right_club_col="club")
    assert res.pairs.empty                       # Zoe: wrong initial; William: different club


def test_abbreviation_shared_by_two_players_matches_neither():
    left = _left([("Ben Fish", "Fish", "Man Utd"), ("Billy Fish", "B.Fish", "Man Utd")])
    right = _right([("B. Fish", "Manchester United")])
    assert match_players(left, right, right_club_col="club").pairs.empty


def test_two_candidates_for_one_player_is_ambiguous():
    left = _left([("John Smith", "Smith", "Leeds")])
    right = _right([("J. Smith", "Leeds United"), ("J. Smith", "Leeds United")])
    assert match_players(left, right, right_club_col="club").pairs.empty


def test_cross_club_fallback_handles_out_of_date_squads():
    left = _left([("William Fish", "Fish", "Chelsea")])          # moved since the squad snapshot
    right = _right([("W. Fish", "Manchester United")])
    assert match_players(left, right, right_club_col="club").pairs.empty
    res = match_players(left, right, right_club_col="club", cross_club_fallback=True)
    assert res.pairs["method"].tolist() == ["initial+surname (any club)"]


def test_surname_only_web_name_never_matches_a_different_initial():
    """Regression: web_name "O'Reilly" used to match "N. O'Reilly" without checking the initial."""
    left = _left([("Ben O'Reilly", "O'Reilly", "Man City")])
    right = _right([("N. O'Reilly", "Manchester City")])
    assert match_players(left, right, right_club_col="club").pairs.empty

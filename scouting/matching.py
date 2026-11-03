"""Joining players across sources by name.

The sources share no common player id, so we match on names. Names differ by
accents ("Ødegaard"), length ("Bruno Fernandes" vs "Bruno Miguel Borges
Fernandes"), abbreviation ("W. Fish" vs "William Fish") and club spelling
("Man Utd" vs "Manchester United Football Club"). The matcher is tiered,
conservative (ambiguous -> no match) and reports its match rate so a bad join is
visible instead of silent.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

_TRANSLATE = str.maketrans({
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ß": "ss",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i", "þ": "th",
})
_CLUB_STOPWORDS = {"fc", "afc", "football", "club", "the", "and", "cf", "sc"}
_CLUB_ALIASES = {
    "man utd": "manchester united", "man city": "manchester city", "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur", "wolves": "wolverhampton wanderers",
    "nott m forest": "nottingham forest", "newcastle": "newcastle united",
    "west ham": "west ham united", "brighton": "brighton hove albion",
    "leeds": "leeds united", "leicester": "leicester city", "norwich": "norwich city",
}


def normalize_name(value: object) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).translate(_TRANSLATE)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_club(value: object) -> str:
    text = normalize_name(value)
    text = _CLUB_ALIASES.get(text, text)
    tokens = ["united" if t == "utd" else t for t in text.split() if t not in _CLUB_STOPWORDS]
    return _CLUB_ALIASES.get(" ".join(tokens), " ".join(tokens))


def club_similarity(a: object, b: object) -> float:
    """1.0 same club, 0.9 one name is a token-subset of the other, else 0.0.

    Deliberately NOT a fuzzy string ratio: "Manchester United" vs "Manchester City"
    scores ~0.8 on difflib, which would silently merge two different clubs.
    """
    na, nb = normalize_club(a), normalize_club(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    return 0.9 if ta <= tb or tb <= ta else 0.0


def split_abbreviated(norm: str) -> tuple[list[str], list[str]] | None:
    """'w fish' -> (['w'], ['fish']); 'e j kroupi' -> (['e','j'], ['kroupi']); full names -> None."""
    toks = norm.split()
    i = 0
    while i < len(toks) - 1 and len(toks[i]) == 1:
        i += 1
    return (toks[:i], toks[i:]) if i > 0 else None


def abbreviation_matches(full_tokens: list[str], initials: list[str], surname: list[str]) -> bool:
    """Do `full_tokens` (e.g. william fish / b fernandes) fit initials + surname?"""
    n = len(surname)
    if len(full_tokens) <= n or full_tokens[-n:] != surname:
        return False
    given = full_tokens[:-n]
    if not given[0].startswith(initials[0]):
        return False
    return not (len(initials) > 1 and len(given) > 1 and not given[1].startswith(initials[1]))


@dataclass
class MatchResult:
    pairs: pd.DataFrame   # columns: left_index, right_index, method
    unmatched_left: list
    unmatched_right_count: int

    @property
    def rate(self) -> float:
        total = len(self.pairs) + len(self.unmatched_left)
        return len(self.pairs) / total if total else 0.0


def match_players(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name_cols: tuple[str, ...] = ("name", "web_name"),
    left_club_col: str | None = "club",
    right_name_col: str = "name",
    right_club_col: str | None = None,
    club_threshold: float = 0.6,
    cross_club_fallback: bool = False,
) -> MatchResult:
    """Match each row of `left` to at most one row of `right`.

    Tiers (first hit wins, anything ambiguous is left unmatched):
      1. exact normalised name; duplicates are broken by club similarity
      2. token-subset on the same club: every token of the shorter name appears in
         the longer one, and the shorter name has >=2 tokens (so a bare "Gabriel"
         never matches "Gabriel Jesus")
      3. abbreviated name ("W. Fish") = first initial(s) + surname, same club
      4. like 3 but ignoring the club (only if cross_club_fallback): for squads that
         are out of date, e.g. a player who moved clubs since the snapshot
    Tiers 3-4 only accept a pair when it is unique in BOTH directions: the player
    has one candidate and that candidate fits no other player. A right-hand row is
    never used twice.
    """
    right = right.reset_index(drop=True)
    r_norm = right[right_name_col].map(normalize_name)
    r_tokens = r_norm.map(lambda s: set(s.split()))
    r_abbrev = {i: split_abbreviated(nm) for i, nm in r_norm.items()}
    # Abbreviated right-hand names ("W. Fish") never take part in tiers 1-2: those tiers
    # do not check initials or two-way uniqueness. Tier 3 does.
    by_name: dict[str, list[int]] = {}
    for idx, nm in r_norm.items():
        if nm and r_abbrev[idx] is None:
            by_name.setdefault(nm, []).append(idx)
    r_club = right[right_club_col] if right_club_col else None

    used: set[int] = set()
    pairs: list[tuple[int, int, str]] = []
    unresolved: list = []

    def left_club(row):
        return row[left_club_col] if left_club_col and left_club_col in left.columns else None

    def club_ok(l_club, r_idx) -> bool:
        if r_club is None or l_club is None:
            return True
        return club_similarity(l_club, r_club.iloc[r_idx]) >= club_threshold

    def club_score(l_club, r_idx) -> float:
        if r_club is None or l_club is None:
            return 0.0
        return club_similarity(l_club, r_club.iloc[r_idx])

    def names_of(row) -> list[str]:
        return [n for n in (normalize_name(row[c]) for c in left_name_cols if c in left.columns) if n]

    # ---- tiers 1-2 --------------------------------------------------------------
    for l_idx, row in left.iterrows():
        l_club, names = left_club(row), names_of(row)
        hit: tuple[int, str] | None = None
        for nm in names:                                           # tier 1
            cands = [i for i in by_name.get(nm, []) if i not in used]
            if len(cands) == 1:
                hit = (cands[0], "exact")
            elif len(cands) > 1 and r_club is not None and l_club is not None:
                ranked = sorted(cands, key=lambda i: -club_score(l_club, i))
                best, second = club_score(l_club, ranked[0]), club_score(l_club, ranked[1])
                if best >= club_threshold and best > second:  # winner must really be the same club
                    hit = (ranked[0], "exact+club")
            if hit:
                break
        if hit is None:                                            # tier 2
            for nm in names:
                l_tok = set(nm.split())
                cands = []
                for i, r_tok in r_tokens.items():
                    if i in used or not r_tok or r_abbrev[i] is not None:
                        continue
                    short, long_ = (l_tok, r_tok) if len(l_tok) <= len(r_tok) else (r_tok, l_tok)
                    if len(short) >= 2 and short <= long_ and club_ok(l_club, i):
                        cands.append(i)
                if len(cands) == 1:
                    hit = (cands[0], "token-subset")
                    break
        if hit is None:
            unresolved.append(l_idx)
        else:
            used.add(hit[0])
            pairs.append((l_idx, hit[0], hit[1]))

    # ---- tiers 3-4: abbreviated names, resolved with two-way uniqueness ------------
    def abbreviation_pass(todo: list, require_club: bool, method: str) -> list:
        cand_map: dict = {}
        for l_idx in todo:
            row = left.loc[l_idx]
            l_club = left_club(row)
            found: set[int] = set()
            for nm in names_of(row):
                toks = nm.split()
                for i, ab in r_abbrev.items():
                    if ab is None or i in used:
                        continue
                    if require_club and not club_ok(l_club, i):
                        continue
                    if abbreviation_matches(toks, *ab):
                        found.add(i)
            cand_map[l_idx] = found
        claimed: dict[int, int] = {}
        for found in cand_map.values():
            for i in found:
                claimed[i] = claimed.get(i, 0) + 1
        remaining = []
        for l_idx in todo:
            found = cand_map[l_idx]
            if len(found) == 1 and claimed[next(iter(found))] == 1:
                i = next(iter(found))
                used.add(i)
                pairs.append((l_idx, i, method))
            else:
                remaining.append(l_idx)
        return remaining

    unresolved = abbreviation_pass(unresolved, True, "initial+surname")
    if cross_club_fallback:
        unresolved = abbreviation_pass(unresolved, False, "initial+surname (any club)")

    pairs_df = pd.DataFrame(pairs, columns=["left_index", "right_index", "method"])
    return MatchResult(pairs_df, unresolved, len(right) - len(used))


def match_by_birth_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name_cols: tuple[str, ...] = ("name", "web_name"),
    right_name_col: str = "name",
    dob_col: str = "birth_date",
    exclude_right: set[int] | frozenset[int] = frozenset(),
    min_token_len: int = 3,
) -> pd.DataFrame:
    """Match players whose club may differ between sources (transfers, stale snapshots).

    A pair needs the SAME date of birth and at least one shared name token of
    `min_token_len`+ letters ("Ben" vs "Benjamin White" shares "white"). A birth date
    plus a shared name token is far stronger evidence than a name alone, so the club is
    ignored. The pair must still be unique in both directions. Returns columns
    left_index, right_index, method; `right` rows are addressed by their positional
    index after reset_index(drop=True).
    """
    empty = pd.DataFrame(columns=["left_index", "right_index", "method"])
    if dob_col not in left.columns or dob_col not in right.columns:
        return empty
    right = right.reset_index(drop=True)
    by_dob: dict = {}
    for i, d in right[dob_col].items():
        if pd.notna(d) and i not in exclude_right:
            by_dob.setdefault(pd.Timestamp(d).normalize(), []).append(i)
    r_tokens = {i: {t for t in normalize_name(n).split() if len(t) >= min_token_len}
                for i, n in right[right_name_col].items()}

    cand_map: dict = {}
    for l_idx, row in left.iterrows():
        d = row[dob_col]
        if pd.isna(d):
            continue
        l_tok = {t for c in left_name_cols if c in left.columns
                 for t in normalize_name(row[c]).split() if len(t) >= min_token_len}
        found = {i for i in by_dob.get(pd.Timestamp(d).normalize(), []) if l_tok & r_tokens[i]}
        if found:
            cand_map[l_idx] = found
    claimed: dict[int, int] = {}
    for found in cand_map.values():
        for i in found:
            claimed[i] = claimed.get(i, 0) + 1
    rows = [(l, next(iter(f)), "name+birth_date") for l, f in cand_map.items()
            if len(f) == 1 and claimed[next(iter(f))] == 1]
    return pd.DataFrame(rows, columns=["left_index", "right_index", "method"])

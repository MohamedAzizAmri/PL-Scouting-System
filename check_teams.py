from scouting.sources.apifootball import ApiFootballClient, parse_team_search

c = ApiFootballClient()
for term in ["liverpool", "bournemouth", "coventry", "ipswich"]:
    print("==", term)
    for t in parse_team_search(c._get("teams", {"search": term})):
        print("  ", t)
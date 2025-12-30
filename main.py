import hashlib
import os

import elote as elo
import pandas as pd

debaters = pd.DataFrame()
glicko_competitors = {}


def generate_player_id(institution: str, full_name: str) -> str:
    """Generates a unique player ID by hashing the debater's institution and full name together."""
    combined = f"{institution}{full_name}"
    return hashlib.sha256(combined.encode()).hexdigest()


def create_player_hashes(tournament: str) -> pd.DataFrame:
    """Creates unique hashes for each player based on their institution and entry"""

    entries_file = f"./tournaments/{tournament}/entries.csv"
    teams = pd.read_csv(entries_file, delimiter=",", header=0)
    teams.columns = teams.columns.str.strip()

    teams["hash"] = teams.apply(
        lambda row: generate_player_id(row["Institution"], row["Entry"]), axis=1
    )

    return teams


def parse_debaters_from_tournament(tournament: str) -> None:
    """Adds tournament entries to global debaters DataFrame"""
    global debaters
    global glicko_competitors

    teams = create_player_hashes(tournament)

    file = f"./tournaments/{tournament}/entries.csv"

    teams.to_csv(file, index=False)

    for _, team_row in teams.iterrows():
        hash = team_row["hash"]

        if debaters.empty:
            is_already_in_debaters = False
        else:
            is_already_in_debaters = hash in debaters["hash"].values

        if not is_already_in_debaters:
            debaters = pd.concat([debaters, team_row.to_frame().T], ignore_index=True)
            glicko_competitors[hash] = elo.GlickoCompetitor()


def run_round(tournament: str, round: str, is_prelim: bool) -> None:
    """Updates elos with wins and losses from a round"""

    global glicko_competitors

    if is_prelim:
        file = f"./tournaments/{tournament}/prelims/{round}.csv"
    else:
        file = f"./tournaments/{tournament}/elims/{round}.csv"

    round_data = pd.read_csv(file)

    round_data = replace_codes_with_hashes(round_data, tournament)

    for _, round_row in round_data.iterrows():
        aff_hash = str(round_row["Aff"])
        neg_hash = str(round_row["Neg"])
        winner = str(round_row["Win"]).lower()

        aff_competitor = glicko_competitors[aff_hash]
        neg_competitor = glicko_competitors[neg_hash]

        if "aff" in winner:
            aff_competitor.beat(neg_competitor)

        if "neg" in winner:
            neg_competitor.beat(aff_competitor)


def create_code_to_hash_dict(tournament: str) -> dict:
    """Creates a dictionary that maps entry codes to hashes"""

    teams = create_player_hashes(tournament)

    code_to_hash = {}
    for _, entry_row in teams.iterrows():
        code = entry_row["Code"]
        hash = entry_row["hash"]
        code_to_hash[code] = hash

    return code_to_hash


def replace_codes_with_hashes(
    round_data: pd.DataFrame, tournament: str
) -> pd.DataFrame:
    """Replaces entry codes for a round with hashes, returning that as a DataFrame"""

    code_to_hash = create_code_to_hash_dict(tournament)

    round_data["Aff"] = round_data["Aff"].map(code_to_hash)

    return round_data


def update_from_tournament(tournament: str) -> None:
    """Updates debaters with all prelim and elim rounds from a tournament"""

    parse_debaters_from_tournament(tournament)

    code_to_hash = create_code_to_hash_dict(tournament)

    prelims_folder = f"./tournaments/{tournament}/prelims/"
    if os.path.exists(prelims_folder):
        prelim_files = [f for f in os.listdir(prelims_folder) if f.endswith(".csv")]
        for prelim_file in prelim_files:
            round_name = prelim_file.replace(".csv", "")
            run_round(tournament, round_name, is_prelim=True)

    elims_folder = f"./tournaments/{tournament}/elims/"
    if os.path.exists(elims_folder):
        elim_files = [f for f in os.listdir(elims_folder) if f.endswith(".csv")]
        for elim_file in elim_files:
            round_name = elim_file.replace(".csv", "")
            run_round(tournament, round_name, is_prelim=False)


def main():
    update_from_tournament("greenhill")
    update_from_tournament("college-prep")

    print(debaters.to_string())


main()

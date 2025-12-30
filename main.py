import hashlib

import elote as elo
import pandas as pd

debaters = pd.DataFrame()


def generate_player_id(institution: str, full_name: str) -> str:
    """Generates a unique player ID by hashing the debater's institution and full name together."""
    combined = f"{institution}{full_name}"
    return hashlib.sha256(combined.encode()).hexdigest()


def parse_debaters_from_tournament(tournament):
    """Adds tournament entries to global debaters DataFrame"""
    global debaters

    file = "./tournaments/" + tournament + "/entries.csv"
    teams = pd.read_csv(file, delimiter=",", header=0, usecols=[0, 2])

    # Strip whitespace from column names
    teams.columns = teams.columns.str.strip()

    # Generate unique player IDs for each debater
    teams["hash"] = teams.apply(
        lambda row: generate_player_id(row["Institution"], row["Entry"]), axis=1
    )

    for _, team_row in teams.iterrows():
        hash = team_row["hash"]

        if debaters.empty:
            is_already_in_debaters = False
        else:
            is_already_in_debaters = hash in debaters["hash"].values

        if not is_already_in_debaters:
            debaters = pd.concat([debaters, team_row.to_frame().T], ignore_index=True)


def main():
    parse_debaters_from_tournament("greenhill")
    parse_debaters_from_tournament("college-prep")
    print(debaters.to_string())


main()

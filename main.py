import hashlib
import os

import pandas as pd
from skelo.model.glicko2 import Glicko2Model

debaters = pd.DataFrame()
glicko_model = Glicko2Model()
match_counter = 0


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
    global glicko_model

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
            glicko_model.add(hash)


def run_round(tournament: str, round: str, weight: int = 1) -> None:
    """Updates elos with wins and losses from a round

    Args:
        tournament: tournament name
        round: round name
        weight: how many times to process each match (higher = more impact on ratings)
    """

    global glicko_model
    global match_counter

    file = f"./tournaments/{tournament}/{round}.csv"

    round_data = pd.read_csv(file)

    round_data = replace_codes_with_hashes(round_data, tournament)

    for _, round_row in round_data.iterrows():
        aff_hash = str(round_row["Aff"])
        neg_hash = str(round_row["Neg"])
        winner = str(round_row["Win"]).lower()

        # Skip bye rounds or missing data
        if (
            "nan" in aff_hash
            or "nan" in neg_hash
            or "bye" in aff_hash.lower()
            or "bye" in neg_hash.lower()
        ):
            continue

        # Process the match 'weight' times to give it more impact
        for _ in range(weight):
            if "aff" in winner:
                glicko_model.update(aff_hash, neg_hash, match_counter)

            if "neg" in winner:
                glicko_model.update(neg_hash, aff_hash, match_counter)

            match_counter += 1


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
    round_data["Neg"] = round_data["Neg"].map(code_to_hash)

    return round_data


def determine_weight(round_name: str) -> int:
    """Determines weight of round"""

    round_lower = round_name.lower()

    match round_lower:
        case "doubles":
            return 2
        case "octos":
            return 3
        case "quarters":
            return 4
        case "semis":
            return 5
        case "finals":
            return 6

    return 1


def update_from_tournament(tournament: str) -> None:
    """Updates debaters with all prelim and elim rounds from a tournament"""

    parse_debaters_from_tournament(tournament)

    tournament_folder = f"./tournaments/{tournament}/"

    files = [
        f
        for f in os.listdir(tournament_folder)
        if f.endswith(".csv") and not f.startswith("entries")
    ]

    for file in files:
        round_name = file.replace(".csv", "")

        weight = determine_weight(round_name)

        run_round(tournament, round_name, weight)


def main():
    update_from_tournament("greenhill")
    update_from_tournament("college-prep")

    # Create rankings data
    rankings_data = []
    for index, debater in debaters.iterrows():
        hash = debater["hash"]
        rating_data = glicko_model.get(hash)
        # rating_data["rating"] is a tuple of (mu, phi, sigma)
        # We only want mu (the actual rating value)
        rankings_data.append(
            {
                "School": debater["Institution"],
                "Name": debater["Entry"],
                "Rating": rating_data["rating"][0],
            }
        )

    # Sort by glicko rating (highest to lowest)
    rankings_df = pd.DataFrame(rankings_data)
    rankings_df = rankings_df.sort_values(by="Rating", ascending=False)

    # Add rank column
    rankings_df.insert(0, "Rank", range(1, len(rankings_df) + 1))

    # Output to CSV
    rankings_df.to_csv("rankings.csv", index=False)


main()

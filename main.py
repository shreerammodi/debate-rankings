import hashlib
import os

import pandas as pd
from skelo.model.glicko2 import Glicko2Model

debaters = pd.DataFrame()
glicko_model = Glicko2Model()
match_counter = 0

# Debaters who compete for multiple teams
MULTI_TEAM_DEBATERS = {
    "Tilak Datta Iyer",
}


def generate_player_id(institution: str, full_name: str) -> str:
    """Generates a unique player ID by hashing the debater's institution and full name together.

    For debaters who compete for multiple teams, only the name is used to ensure
    a unified ranking across all their appearances.
    """

    if full_name in MULTI_TEAM_DEBATERS:
        combined = full_name
    else:
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
        weight: how many times to process each match (1 for regular, 2 for majors)
    """

    global glicko_model
    global match_counter

    file = f"./tournaments/{tournament}/{round}.csv"

    round_data = pd.read_csv(file)

    round_data = replace_codes_with_hashes(round_data, tournament)

    # Use the same timestamp for all matches in this round to avoid recency bias
    round_timestamp = match_counter

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

        # Process match 'weight' times for tournament importance
        # (1x for regular tournaments, 2x for majors)
        for _ in range(weight):
            if "aff" in winner:
                glicko_model.update(aff_hash, neg_hash, round_timestamp)

            if "neg" in winner:
                glicko_model.update(neg_hash, aff_hash, round_timestamp)

    # Only increment counter once per round, not per match
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


def determine_weight(tournament_name: str, round_name: str) -> int:
    """Determines weight of round based on tournament importance

    - Regular tournaments: 1x (each match processed once)
    - Major tournaments: 2x (each match processed twice)
    """

    majors = ["heart-of-texas", "glenbrooks", "greenhill", "emory", "cal"]

    if tournament_name in majors:
        return 2
    else:
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

        weight = determine_weight(tournament, round_name)

        run_round(tournament, round_name, weight)


def main():
    print("Running tournaments...")
    update_from_tournament("loyola")
    update_from_tournament("ukso")
    update_from_tournament("grapevine")
    update_from_tournament("greenhill-rr")
    update_from_tournament("greenhill")
    update_from_tournament("jack-howe")
    update_from_tournament("valley")
    update_from_tournament("nano-nagle")
    update_from_tournament("heart-of-texas")
    update_from_tournament("nyc")
    update_from_tournament("fbk-rr")
    update_from_tournament("fbk")
    update_from_tournament("apple-valley")
    update_from_tournament("glenbrooks")
    update_from_tournament("dsds1")
    update_from_tournament("longhorn")
    update_from_tournament("strake")
    update_from_tournament("blake")
    update_from_tournament("college-prep")

    print("Creating Rankings...")

    # Create rankings data
    rankings_data = []
    for index, debater in debaters.iterrows():
        hash = debater["hash"]
        rating_data = glicko_model.get(hash)
        # rating_data["rating"] is a tuple of (mu, phi, sigma)
        mu = rating_data["rating"][0]  # Rating
        phi = rating_data["rating"][1]  # Rating deviation (uncertainty)
        sigma = rating_data["rating"][2]  # Volatility

        # Count how many matches this debater has played
        # This is a rough estimate based on rating history
        match_count = len(glicko_model.ratings.get(hash, [])) - 1

        # Adjusted rating: Rating - 2*Deviation
        # This penalizes debaters with high uncertainty (few matches)
        adjusted_rating = mu - 2 * phi

        rankings_data.append(
            {
                "School": debater["Institution"],
                "Name": debater["Entry"],
                "Adjusted Rating": adjusted_rating,
                "Deviation": phi,
                "Matches": match_count,
                "Rating": mu,
            }
        )

    rankings_df = pd.DataFrame(rankings_data)
    rankings_df = rankings_df.sort_values(by="Adjusted Rating", ascending=False)

    rankings_df.insert(0, "Rank", range(1, len(rankings_df) + 1))

    rankings_df.to_csv("full_rankings.csv", index=False)

    rankings_df = rankings_df.drop(columns=["Deviation", "Matches", "Rating"])

    rankings_df["Adjusted Rating"] = rankings_df["Adjusted Rating"].round(2)

    rankings_df.rename(columns={"Adjusted Rating": "Rating"}, inplace=True)

    rankings_df.to_csv("rankings.csv", index=False)


main()

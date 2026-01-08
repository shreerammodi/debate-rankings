import hashlib

import pandas as pd


def generate_player_id(
    institution: str, full_name: str, multi_team_debaters: list, format: str
) -> str:
    """Generates a unique player ID by hashing the debater's institution and full name together.

    For debaters who compete for multiple teams, only the name is used to ensure
    a unified ranking across all their appearances.
    """
    if format == "hsld":
        if full_name in multi_team_debaters:
            combined = full_name
        else:
            combined = f"{institution}{full_name}"
    else:
        name_arr = full_name.split("&")

        clean_name_arr = []

        for name in name_arr:
            clean_name_arr.append(name.strip())

        clean_name_arr.sort()

        combined = f"{institution}{''.join(clean_name_arr)}"

    return hashlib.sha256(combined.encode()).hexdigest()


def create_player_hashes(
    tournament: str, multi_team_debaters: list, format: str
) -> pd.DataFrame:
    """Creates unique hashes for each player based on their institution and entry"""

    entries_file = f"./tournaments/{tournament}/entries.csv"
    teams = pd.read_csv(entries_file, delimiter=",", header=0)
    teams.columns = teams.columns.str.strip()

    teams["hash"] = teams.apply(
        lambda row: generate_player_id(
            row["Institution"], row["Entry"], multi_team_debaters, format
        ),
        axis=1,
    )

    return teams


def parse_debaters_from_tournament(
    tournament: str,
    debaters: pd.DataFrame,
    glicko_model,
    multi_team_debaters: list,
    format: str,
) -> pd.DataFrame:
    """Adds tournament entries to debaters DataFrame and glicko model

    Args:
        tournament: Tournament name
        debaters: DataFrame of existing debaters
        glicko_model: Glicko2 model instance
        multi_team_debaters: List of debaters who compete for multiple teams

    Returns:
        Updated debaters DataFrame
    """
    teams = create_player_hashes(tournament, multi_team_debaters, format)

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

    return debaters

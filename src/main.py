import math
import os

import pandas as pd
from fisher_scoring import BradleyTerry, pairs_from_counts

from determine_tournament_weight import determine_weight
from load_config import load_config
from player_utils import create_player_hashes, parse_debaters_from_tournament

# Ridge penalty on team skill (logit scale). 1.0 is a N(0, 1) prior: teams with
# few ballots are pulled toward the field average and unbeaten teams stay finite.
SKILL_PRIOR_PRECISION = 1.0
# Report ratings on the familiar Elo scale: 400 points = 10:1 odds.
RATING_BASE = 1500
RATING_SCALE = 400 / math.log(10)


class RankingSystem:
    """Manages debate rankings with a Bradley-Terry model fit to every ballot"""

    # Rank by rating minus this many deviations.
    UNCERTAINTY_PENALTY = 2

    def __init__(
        self, config_path: str, format_dir: str = "", tournaments_root: str = "./tournaments"
    ):
        """Initialize the ranking system with a config file

        Args:
            config_path: Path to the JSON config file
            format_dir: Directory name for the format (e.g., "cpd", "hsld")
            tournaments_root: Folder holding the format directories
        """
        self.config = load_config(config_path)
        self.tournaments_root = tournaments_root
        # Raise on round entries missing from entries.csv (off for archived seasons).
        self.strict_entries = True
        self.format_dir = format_dir
        self.debaters = pd.DataFrame()
        self.comparisons = []
        self.aff_advantage = float("nan")
        self.win_statistics = {}

    @staticmethod
    def _is_elimination_round(round_name: str, columns) -> bool:
        elimination_terms = (
            "runoff",
            "double",
            "triple",
            "quad",
            "octo",
            "quarter",
            "semi",
            "final",
        )
        normalized_name = round_name.lower()
        return (
            ("Judges" in columns and "Votes" in columns)
            or any(term in normalized_name for term in elimination_terms)
        )

    def _record_result(
        self, player_hash: str, side: str, won: bool, is_elimination: bool
    ) -> None:
        statistics = self.win_statistics.setdefault(
            player_hash,
            {
                "aff_wins": 0,
                "aff_rounds": 0,
                "neg_wins": 0,
                "neg_rounds": 0,
                "aff_elim_wins": 0,
                "aff_elim_rounds": 0,
                "neg_elim_wins": 0,
                "neg_elim_rounds": 0,
            },
        )
        statistics[f"{side}_rounds"] += 1
        statistics[f"{side}_wins"] += int(won)
        if is_elimination:
            statistics[f"{side}_elim_rounds"] += 1
            statistics[f"{side}_elim_wins"] += int(won)

    @staticmethod
    def _win_rate(wins: int, rounds: int) -> float:
        if rounds == 0:
            return float("nan")
        return round(100 * wins / rounds, 2)

    @staticmethod
    def _count_ballots(round_row, aff_won: bool) -> tuple[float, float]:
        """Returns (aff weight, neg weight) summing to one debate.

        A panel counts as one debate, not one per judge: half goes to the
        winner outright and half is split by ballot share, so a 5-0 is
        1/0 and a 3-2 is 0.8/0.2.
        """
        votes = round_row.get("Votes")
        if isinstance(votes, str):
            tokens = votes.lower().split()
            aff_ballots = sum(token in ("aff", "pro", "affirmative") for token in tokens)
            neg_ballots = sum(token in ("neg", "con", "negative") for token in tokens)
            panel = aff_ballots + neg_ballots
            if panel > 0:
                aff_share = 0.5 * aff_won + 0.5 * aff_ballots / panel
                return aff_share, 1 - aff_share
        return (1, 0) if aff_won else (0, 1)

    def run_round(self, tournament: str, round: str, weight: int = 1) -> None:
        """Records ballots and win statistics from a round

        Args:
            tournament: tournament name
            round: round name
            weight: likelihood weight for this round's ballots (2 for majors)
        """
        tournament_path = (
            f"{self.format_dir}/{tournament}" if self.format_dir else tournament
        )
        file = f"{self.tournaments_root}/{tournament_path}/{round}.csv"
        round_data = pd.read_csv(file)
        # Normalize "Pro"/"Con" (PF) and "Affirmative"/"Negative" column names
        round_data = round_data.rename(
            columns={
                "Pro": "Aff",
                "Con": "Neg",
                "Affirmative": "Aff",
                "Negative": "Neg",
            }
        )
        is_elimination = self._is_elimination_round(round, round_data.columns)
        round_data = self.replace_codes_with_hashes(round_data, tournament)

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
            aff_won = "aff" in winner or "pro" in winner
            neg_won = "neg" in winner or "con" in winner
            if not aff_won and not neg_won:
                continue

            self._record_result(aff_hash, "aff", aff_won, is_elimination)
            self._record_result(neg_hash, "neg", neg_won, is_elimination)
            aff_ballots, neg_ballots = self._count_ballots(round_row, aff_won)
            # Tournament weight affects ratings, not the raw win-rate sample.
            self.comparisons.append(
                {
                    "aff": aff_hash,
                    "neg": neg_hash,
                    "aff_ballots": aff_ballots * weight,
                    "neg_ballots": neg_ballots * weight,
                }
            )

    def _fit_skills(self) -> pd.DataFrame:
        """Fits Bradley-Terry skills (logit scale) with an aff-side term to all ballots

        Returns:
            DataFrame indexed by team hash with "ability" and "se" columns
        """
        if not self.comparisons:
            return pd.DataFrame(columns=["ability", "se"])

        comparisons = pd.DataFrame(self.comparisons)
        X, y, sample_weight = pairs_from_counts(
            comparisons, "aff", "neg", "aff_ballots", "neg_ballots"
        )
        model = BradleyTerry(use_bias=True, l2=SKILL_PRIOR_PRECISION)
        model.fit(X, y, sample_weight=sample_weight)
        self.aff_advantage = float(
            model.summary_frame().loc["bias (order effect)", "estimate"]
        )
        return model.rank()

    def create_code_to_hash_dict(self, tournament: str) -> dict:
        """Creates a dictionary that maps entry codes to hashes"""
        tournament_path = (
            f"{self.format_dir}/{tournament}" if self.format_dir else tournament
        )
        teams = create_player_hashes(
            tournament_path,
            self.config.get("multi_team_debaters", []),
            self.format_dir,
            self.tournaments_root,
        )

        code_to_hash = {}
        for _, entry_row in teams.iterrows():
            code = str(entry_row["Code"]).strip()
            hash = entry_row["hash"]
            code_to_hash[code] = hash

        return code_to_hash

    @staticmethod
    def _is_played_side(code) -> bool:
        if pd.isna(code):
            return False
        normalized_code = str(code).strip().lower()
        return normalized_code != "" and "bye" not in normalized_code

    def replace_codes_with_hashes(
        self, round_data: pd.DataFrame, tournament: str
    ) -> pd.DataFrame:
        """Replaces entry codes for a round with hashes, returning that as a DataFrame"""
        code_to_hash = self.create_code_to_hash_dict(tournament)

        for side in ("Aff", "Neg"):
            original_codes = round_data[side].copy()
            normalized_codes = original_codes.map(
                lambda code: str(code).strip() if self._is_played_side(code) else code
            )
            round_data[side] = normalized_codes.map(code_to_hash)
            unmapped_codes = sorted(
                {
                    str(code).strip()
                    for code, mapped in zip(original_codes, round_data[side])
                    if self._is_played_side(code) and pd.isna(mapped)
                }
            )
            if unmapped_codes and not self.strict_entries:
                # Archived seasons predate this check: skip those rounds.
                print(
                    f"Warning: {tournament} skips rounds with unmapped {side} entries: "
                    + ", ".join(unmapped_codes)
                )
            elif unmapped_codes:
                # Tabroom's entry export omits entries that dropped mid-tournament,
                # but their rounds remain in the round CSVs.
                raise ValueError(
                    f"{tournament} has unmapped {side} entries: "
                    + ", ".join(unmapped_codes)
                    + ". Likely dropped entries missing from Tabroom's entry export;"
                    " add Institution,Location,Entry,Code rows for them to "
                    f"{self.tournaments_root}/{self.format_dir + '/' if self.format_dir else ''}"
                    f"{tournament}/entries.csv"
                )

        return round_data

    def update_from_tournament(self, tournament: str) -> None:
        """Updates debaters with all prelim and elim rounds from a tournament"""
        tournament_path = (
            f"{self.format_dir}/{tournament}" if self.format_dir else tournament
        )

        self.debaters = parse_debaters_from_tournament(
            tournament_path,
            self.debaters,
            self.config.get("multi_team_debaters", []),
            self.format_dir,
            self.tournaments_root,
        )

        tournament_folder = f"{self.tournaments_root}/{tournament_path}/"

        files = [
            f
            for f in os.listdir(tournament_folder)
            if f.endswith(".csv") and not f.startswith("entries")
        ]

        files.sort()

        print(f"Processing {tournament}: {files}")

        for file in files:
            round_name = file.replace(".csv", "")
            weight = determine_weight(tournament, self.config.get("majors", []))
            self.run_round(tournament, round_name, weight)

    def generate_rankings(self, output_prefix: str = "") -> None:
        """Generate and save rankings to CSV files

        Args:
            output_prefix: Prefix for output filenames (e.g., "ld_" or "cpd_")
        """
        print("Creating Rankings...")

        skills = self._fit_skills()
        # Teams without a decided round sit at the prior: field average, prior sd.
        prior_se = SKILL_PRIOR_PRECISION**-0.5

        # Create rankings data
        rankings_data = []
        for index, debater in self.debaters.iterrows():
            hash = debater["hash"]
            if hash in skills.index:
                ability, se = skills.loc[hash, ["ability", "se"]]
            else:
                ability, se = 0.0, prior_se
            rating = RATING_BASE + RATING_SCALE * ability
            deviation = RATING_SCALE * se

            statistics = self.win_statistics.get(hash, {})
            aff_rounds = statistics.get("aff_rounds", 0)
            neg_rounds = statistics.get("neg_rounds", 0)
            aff_elim_rounds = statistics.get("aff_elim_rounds", 0)
            neg_elim_rounds = statistics.get("neg_elim_rounds", 0)
            match_count = aff_rounds + neg_rounds

            # Adjusted rating: Rating - UNCERTAINTY_PENALTY * Deviation
            # This penalizes debaters with high uncertainty (few matches)
            adjusted_rating = rating - self.UNCERTAINTY_PENALTY * deviation

            rankings_data.append(
                {
                    "School": debater["Institution"],
                    "Name": debater["Entry"],
                    "Adjusted Rating": adjusted_rating,
                    "Deviation": deviation,
                    "Matches": match_count,
                    "Rating": rating,
                    "Hash": debater["hash"],
                    "Aff Win Rate": self._win_rate(
                        statistics.get("aff_wins", 0), aff_rounds
                    ),
                    "Neg Win Rate": self._win_rate(
                        statistics.get("neg_wins", 0), neg_rounds
                    ),
                    "Aff Elim Win Rate": self._win_rate(
                        statistics.get("aff_elim_wins", 0), aff_elim_rounds
                    ),
                    "Neg Elim Win Rate": self._win_rate(
                        statistics.get("neg_elim_wins", 0), neg_elim_rounds
                    ),
                }
            )

        rankings_df = pd.DataFrame(rankings_data)
        rankings_df = rankings_df.sort_values(by="Adjusted Rating", ascending=False)

        rankings_df.insert(0, "Rank", range(1, len(rankings_df) + 1))

        rankings_df.to_csv(f"output/{output_prefix}full_rankings.csv", index=False)
        field_totals = {
            key: sum(statistics[key] for statistics in self.win_statistics.values())
            for key in (
                "aff_wins",
                "aff_rounds",
                "neg_wins",
                "neg_rounds",
                "aff_elim_wins",
                "aff_elim_rounds",
                "neg_elim_wins",
                "neg_elim_rounds",
            )
        }
        field_statistics = pd.DataFrame(
            [
                {
                    "Aff Win Rate": self._win_rate(
                        field_totals["aff_wins"], field_totals["aff_rounds"]
                    ),
                    "Neg Win Rate": self._win_rate(
                        field_totals["neg_wins"], field_totals["neg_rounds"]
                    ),
                    "Aff Elim Win Rate": self._win_rate(
                        field_totals["aff_elim_wins"],
                        field_totals["aff_elim_rounds"],
                    ),
                    "Neg Elim Win Rate": self._win_rate(
                        field_totals["neg_elim_wins"],
                        field_totals["neg_elim_rounds"],
                    ),
                    # Rating points the aff side is worth, from the model's side term.
                    "Aff Rating Advantage": round(
                        RATING_SCALE * self.aff_advantage, 2
                    ),
                }
            ]
        )
        field_statistics.to_csv(
            f"output/{output_prefix}field_statistics.csv", index=False
        )

        rankings_df.drop("Hash", inplace=True, axis=1)

        rankings_df = rankings_df.drop(columns=["Deviation", "Matches", "Rating"])

        rankings_df["Adjusted Rating"] = rankings_df["Adjusted Rating"].round(2)

        rankings_df.rename(columns={"Adjusted Rating": "Rating"}, inplace=True)

        rankings_df.to_csv(f"output/{output_prefix}rankings.csv", index=False)

        print(
            f"Rankings saved to output/{output_prefix}rankings.csv, "
            f"output/{output_prefix}full_rankings.csv, and "
            f"output/{output_prefix}field_statistics.csv"
        )

    def run(self, output_prefix: str = "", tournaments: list = None) -> None:
        """Run the ranking system for all (or a subset of) tournaments in the config

        Args:
            output_prefix: Prefix for output filenames (e.g., "ld_" or "cpd_")
            tournaments: Optional list of tournaments to process; defaults to all in config
        """
        if tournaments is None:
            tournaments = self.config.get("tournaments", [])

        for tournament in tournaments:
            print(f"\nProcessing tournament: {tournament}")
            self.update_from_tournament(tournament)

        self.generate_rankings(output_prefix)



def partition_tournaments_by_topic(
    tournaments: list, boundaries: dict
) -> tuple[list, list, list]:
    """Partition tournaments into Sept/Oct, Nov/Dec, and Jan/Feb topics."""
    partitions = []
    start = 0

    for boundary_name in ("sepoct_end", "novdec_end"):
        boundary = boundaries.get(boundary_name)
        try:
            end = tournaments.index(boundary, start) + 1
        except ValueError:
            end = len(tournaments)

        partitions.append(tournaments[start:end])
        start = end

    partitions.append(tournaments[start:])
    return tuple(partitions)


def main():
    from ttt_ranking import TTTRankingSystem, previous_season_for

    def run_format(format_dir: str) -> None:
        print(f"\nStarting {format_dir} ranking system")
        system = TTTRankingSystem(
            f"config/{format_dir}-config.json",
            format_dir,
            previous_season=previous_season_for(format_dir),
        )
        system.run(f"{format_dir}_")
        print(f"\n{format_dir} ranking generation complete!")

    ld_config_path = "config/hsld-config.json"
    ld_format_dir = "hsld"
    run_format(ld_format_dir)

    # Topic-specific LD rankings: each topic's rounds alone, on the
    # Bradley-Terry model (no drift or carried-over priors within a topic).
    ld_config = load_config(ld_config_path)
    all_tournaments = ld_config.get("tournaments", [])
    boundaries = ld_config.get("topic_boundaries", {})
    topic_tournaments = partition_tournaments_by_topic(all_tournaments, boundaries)

    for topic_name, topic_slug, tournaments in zip(
        ("Sept/Oct", "Nov/Dec", "Jan/Feb"),
        ("sepoct", "novdec", "janfeb"),
        topic_tournaments,
    ):
        if not tournaments:
            continue

        print(
            f"\nGenerating LD {topic_name} rankings "
            f"({tournaments[0]} → {tournaments[-1]})"
        )
        topic_system = RankingSystem(ld_config_path, ld_format_dir)
        topic_system.run(f"{ld_format_dir}_{topic_slug}_", tournaments=tournaments)

    for format_dir in ("hspf", "cpd", "hscx"):
        run_format(format_dir)


if __name__ == "__main__":
    main()

"""TrueSkill Through Time rankings.

Every debate from last season and this season goes into one history. Each
tournament is a time step, a debater's skill can drift between tournaments,
and the whole history is smoothed forwards and backwards, so later results
also revise earlier estimates. Last season is the starting point for this
season: a returning debater begins where they finished, with the uncertainty
widened by the time that has passed.
"""

import json
import math
import os
from datetime import date

import pandas as pd
import trueskillthroughtime as ttt

from load_config import load_config
from main import RankingSystem

# Model scale: one unit of skill is one unit of round-to-round performance
# noise (beta).
BETA = 1.0
# Prior for a debater seen for the first time.
PRIOR_MU = 0.0
PRIOR_SIGMA = 1.5
# Skill drift per day. Variance grows by GAMMA**2 per day between tournaments.
# Tuned with src/evaluate_models.py on LD (Blue Key through Damus held out):
# held-out log loss 0.576 at 0.02, 0.561 at 0.08, 0.557 at 0.12, 0.556 at
# 0.16. 0.12 keeps most of the gain without discounting older rounds as hard.
GAMMA = 0.12
# Days assumed between an undated archived season's last tournament and this
# season's first. Seasons with dates use the real gap instead.
OFFSEASON_DAYS = 100
# Archived configs list tournaments in order without dates; space them a week apart.
ARCHIVE_SPACING_DAYS = 7
# Hidden player on the aff team of every debate. Its skill is the aff side advantage.
AFF_PLAYER = "__AFF_SIDE__"
# Convert a skill difference on this model's probit scale to the logit scale
# RankingSystem reports on: P(win) = Phi(d / (sqrt(2) * beta)) ~ logistic(1.702 * d / (sqrt(2) * beta)).
LOGIT_PER_SKILL = 1.702 / (math.sqrt(2) * BETA)


def tournament_dates(config_path: str) -> dict:
    """Maps tournament slug -> date ordinal, or None for configs without dates."""
    with open(config_path) as config_file:
        tournaments = json.load(config_file).get("tournaments", {})
    if isinstance(tournaments, dict):
        return {slug: date.fromisoformat(day).toordinal() for slug, day in tournaments.items()}
    return None


def tournament_days(config_path: str) -> dict:
    """Maps tournament slug -> day number within its season (first tournament = 0).

    Configs with dates map slug -> ISO date. Archived configs that are a plain
    list in order get their tournaments spaced ARCHIVE_SPACING_DAYS apart.
    """
    dates = tournament_dates(config_path)
    if dates is not None:
        first = min(dates.values())
        return {slug: day - first for slug, day in dates.items()}
    with open(config_path) as config_file:
        tournaments = json.load(config_file).get("tournaments", [])
    return {slug: i * ARCHIVE_SPACING_DAYS for i, slug in enumerate(tournaments)}


class TTTRankingSystem(RankingSystem):
    """Rankings from TrueSkill Through Time, seeded with last season's results"""

    UNCERTAINTY_PENALTY = 1

    def __init__(
        self,
        config_path: str,
        format_dir: str = "",
        tournaments_root: str = "./tournaments",
        previous_season: tuple | None = None,
        gamma: float = GAMMA,
        prior_sigma: float = PRIOR_SIGMA,
    ):
        """
        Args:
            previous_season: optional (config_path, tournaments_root) for last
                season. Its games are added to the history before this season's.
            gamma: skill drift per day
            prior_sigma: uncertainty of a debater seen for the first time
        """
        super().__init__(config_path, format_dir, tournaments_root)
        self.config_path = config_path
        self.previous_season = previous_season
        self.gamma = gamma
        self.prior_sigma = prior_sigma
        self.history = None
        self._day = 0

    def update_from_tournament(self, tournament: str) -> None:
        start = len(self.comparisons)
        super().update_from_tournament(tournament)
        for comparison in self.comparisons[start:]:
            comparison["day"] = self._day
            comparison["tournament"] = tournament

    def _load_previous_season(self) -> list:
        """Returns last season's comparisons, with days ending before this season starts."""
        if self.previous_season is None:
            return []
        config_path, tournaments_root = self.previous_season
        previous = RankingSystem(config_path, self.format_dir, tournaments_root)
        previous.strict_entries = False
        # Match identities across seasons with this season's multi-team list.
        previous.config["multi_team_debaters"] = sorted(
            set(previous.config.get("multi_team_debaters", []))
            | set(self.config.get("multi_team_debaters", []))
        )
        days = tournament_days(config_path)
        previous_dates = tournament_dates(config_path)
        current_dates = tournament_dates(self.config_path)
        if previous_dates and current_dates:
            # Both seasons have dates: place last season on the real calendar.
            offset = min(previous_dates.values()) - min(current_dates.values())
        else:
            offset = -(max(days.values()) + OFFSEASON_DAYS) if days else 0
        comparisons = []
        for tournament in previous.config["tournaments"]:
            start = len(previous.comparisons)
            previous.update_from_tournament(tournament)
            for comparison in previous.comparisons[start:]:
                comparison["day"] = days[tournament] + offset
                comparison["tournament"] = f"previous:{tournament}"
            comparisons.extend(previous.comparisons[start:])
        return comparisons

    def run(self, output_prefix: str = "", tournaments: list = None) -> None:
        if tournaments is None:
            tournaments = self.config.get("tournaments", [])
        days = tournament_days(self.config_path)
        for tournament in tournaments:
            print(f"\nProcessing tournament: {tournament}")
            self._day = days[tournament]
            self.update_from_tournament(tournament)
        self.generate_rankings(output_prefix)

    def fit_history(self, comparisons: list):
        """Fits TrueSkill Through Time to comparisons (dicts with aff, neg, day, ballots)."""
        composition, results, times = [], [], []
        for comparison in comparisons:
            # A panel counts as a win for the side that took the decision.
            # TTT has no per-game weights, so split panels count as full wins.
            aff_won = comparison["aff_ballots"] > comparison["neg_ballots"]
            composition.append([[comparison["aff"], AFF_PLAYER], [comparison["neg"]]])
            results.append([1, 0] if aff_won else [0, 1])
            times.append(comparison["day"])
        priors = {
            # The side advantage is a constant: no performance noise, no drift.
            AFF_PLAYER: ttt.Player(ttt.Gaussian(0.0, 1.0), beta=0.0, gamma=0.0),
        }
        history = ttt.History(
            composition,
            results,
            times,
            priors=priors,
            mu=PRIOR_MU,
            sigma=self.prior_sigma,
            beta=BETA,
            gamma=self.gamma,
        )
        history.convergence(verbose=False)
        return history

    def _fit_skills(self) -> pd.DataFrame:
        comparisons = self._load_previous_season() + self.comparisons
        if not comparisons:
            return pd.DataFrame(columns=["ability", "se"])
        self.history = self.fit_history(comparisons)
        curves = self.history.learning_curves()
        aff = curves[AFF_PLAYER][-1][1]
        self.aff_advantage = LOGIT_PER_SKILL * aff.mu

        # Report each debater as of the last tournament in this season, so
        # someone who hasn't competed lately carries the extra drift uncertainty.
        season_end = max(c["day"] for c in self.comparisons) if self.comparisons else 0
        rows = {}
        for player, curve in curves.items():
            if player == AFF_PLAYER:
                continue
            last_day, posterior = curve[-1]
            elapsed = max(season_end - last_day, 0)
            sigma = math.sqrt(posterior.sigma**2 + elapsed * self.gamma**2)
            rows[player] = {
                "ability": LOGIT_PER_SKILL * posterior.mu,
                "se": LOGIT_PER_SKILL * sigma,
            }
        return pd.DataFrame.from_dict(rows, orient="index")


def previous_season_for(format_dir: str, season: str = "2025-26"):
    """(config_path, tournaments_root) for an archived season, if it exists."""
    config_path = f"archive/{season}/config/{format_dir}-config.json"
    tournaments_root = f"archive/{season}/tournaments"
    if os.path.exists(config_path) and os.path.isdir(f"{tournaments_root}/{format_dir}"):
        return config_path, tournaments_root
    return None

"""Backtest: fit each model on everything before a tournament, predict that tournament.

Usage: python src/evaluate_models.py [format_dir] [first held-out tournament]
Prints log loss, Brier score and accuracy on the held-out rounds (panels count
as one round, scored by who won the decision).
"""

import json
import math
import os
import sys

import pandas as pd

import main as rankings_main
import ttt_ranking
from main import RankingSystem
from ttt_ranking import AFF_PLAYER, BETA, TTTRankingSystem, previous_season_for, tournament_days


def load_season(format_dir: str, majors_weight: bool):
    """Returns (system, comparisons tagged with day and tournament, ordered tournaments)."""
    original = rankings_main.determine_weight
    if not majors_weight:
        rankings_main.determine_weight = lambda tournament, majors: 1
    try:
        config_path = f"config/{format_dir}-config.json"
        system = TTTRankingSystem(config_path, format_dir)
        days = tournament_days(config_path)
        for tournament in system.config["tournaments"]:
            system._day = days[tournament]
            system.update_from_tournament(tournament)
    finally:
        rankings_main.determine_weight = original
    return system, system.comparisons, system.config["tournaments"]


def score(predictions: list) -> dict:
    """predictions: list of (p_aff, aff_won)"""
    eps = 1e-6
    log_loss = -sum(
        math.log(max(p if won else 1 - p, eps)) for p, won in predictions
    ) / len(predictions)
    brier = sum((p - won) ** 2 for p, won in predictions) / len(predictions)
    accuracy = sum((p > 0.5) == won for p, won in predictions) / len(predictions)
    return {"rounds": len(predictions), "log_loss": log_loss, "brier": brier, "accuracy": accuracy}


def predict_bradley_terry(train: list, test: list) -> list:
    system = RankingSystem.__new__(RankingSystem)
    system.comparisons = train
    skills = system._fit_skills()
    out = []
    for c in test:
        a = skills["ability"].get(c["aff"], 0.0)
        n = skills["ability"].get(c["neg"], 0.0)
        p = 1 / (1 + math.exp(-(a - n + system.aff_advantage)))
        out.append((p, c["aff_ballots"] > c["neg_ballots"]))
    return out


def predict_ttt(train: list, test: list, gamma: float, prior_sigma: float) -> list:
    system = TTTRankingSystem.__new__(TTTRankingSystem)
    system.gamma, system.prior_sigma = gamma, prior_sigma
    history = system.fit_history(train)
    latest = {player: curve[-1] for player, curve in history.learning_curves().items()}
    aff_side = latest[AFF_PLAYER][1]

    def skill(player, day):
        if player not in latest:
            return ttt_ranking.PRIOR_MU, prior_sigma**2
        last_day, posterior = latest[player]
        return posterior.mu, posterior.sigma**2 + max(day - last_day, 0) * gamma**2

    out = []
    for c in test:
        mu_a, var_a = skill(c["aff"], c["day"])
        mu_n, var_n = skill(c["neg"], c["day"])
        sd = math.sqrt(2 * BETA**2 + var_a + var_n + aff_side.sigma**2)
        z = (mu_a + aff_side.mu - mu_n) / sd
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        out.append((p, c["aff_ballots"] > c["neg_ballots"]))
    return out


def main():
    format_dir = sys.argv[1] if len(sys.argv) > 1 else "hsld"
    first_holdout = sys.argv[2] if len(sys.argv) > 2 else None
    grid = [(g, s) for g in (0.005, 0.01, 0.02, 0.04) for s in (1.0, 1.5, 2.5)]
    if len(sys.argv) > 3:
        grid = [tuple(float(x) for x in pair.split(",")) for pair in sys.argv[3:]]

    system, weighted, order = load_season(format_dir, majors_weight=True)
    _, unweighted, _ = load_season(format_dir, majors_weight=False)
    previous = []
    if previous_season_for(format_dir):
        system.previous_season = previous_season_for(format_dir)
        previous = system._load_previous_season()

    holdouts = order[order.index(first_holdout):] if first_holdout else order[-4:]
    if os.environ.get("EVAL_ONLY"):
        holdouts = holdouts[:1]
    results = {}
    for tournament in holdouts:
        cutoff = min(c["day"] for c in unweighted if c["tournament"] == tournament)
        train_w = [c for c in weighted if c["day"] < cutoff]
        train_u = [c for c in unweighted if c["day"] < cutoff]
        test = [c for c in unweighted if c["tournament"] == tournament]
        if not test:
            continue
        print(f"Held out {tournament}: {len(test)} rounds", flush=True)
        runs = {
            "BT, majors 2x (current)": lambda: predict_bradley_terry(train_w, test),
            "BT, majors 1x": lambda: predict_bradley_terry(train_u, test),
        }
        for gamma, sigma in grid:
            runs[f"TTT g={gamma} s={sigma}"] = (
                lambda g=gamma, s=sigma: predict_ttt(previous + train_u, test, g, s)
            )
            runs[f"TTT g={gamma} s={sigma}, no last season"] = (
                lambda g=gamma, s=sigma: predict_ttt(train_u, test, g, s)
            )
        for name, run in runs.items():
            results.setdefault(name, []).extend(run())

    if os.environ.get("EVAL_SAVE"):
        # Save raw predictions so separate runs can be pooled.
        with open(os.environ["EVAL_SAVE"], "w") as out:
            json.dump(results, out)
    table = pd.DataFrame({name: score(p) for name, p in results.items()}).T
    print(table.sort_values("log_loss").round(4).to_string())


if __name__ == "__main__":
    main()

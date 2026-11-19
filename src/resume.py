"""Résumé score: what each debater has achieved this season.

Separate from the strength rating, which estimates how good a debater is.
Each tournament scores points for how deep a debater went in elims (or where
they finished in a round robin), multiplied by how strong that tournament's
field was. A debater's résumé is the sum of their best BEST_N tournaments.
Not clearing scores nothing.
"""

import math

import pandas as pd

from main import RATING_BASE, RATING_SCALE

# Points by result. Keys are rounds from the end: 0 = finals, 1 = semis, ...
CHAMPION_POINTS = 100
LOST_IN_POINTS = {0: 70, 1: 50, 2: 35, 3: 25, 4: 18, 5: 12}
DEEPER_POINTS = 8
# Finals not played or not recorded (e.g. a closeout): halfway between
# champion and finalist for both.
SHARED_FINAL_POINTS = (CHAMPION_POINTS + LOST_IN_POINTS[0]) / 2
# Round robins have no bracket: points by finishing place on wins.
ROUND_ROBIN_PLACES = [(1, CHAMPION_POINTS), (2, LOST_IN_POINTS[0]), (4, LOST_IN_POINTS[1]), (8, LOST_IN_POINTS[2])]
# Field strength: mean rating of the top FIELD_TOP entrants. A field
# FIELD_SCALE rating points above the season's median tournament is worth 2x;
# below the median it shrinks, down to MIN_MULTIPLIER.
FIELD_TOP = 32
FIELD_SCALE = 400
MIN_MULTIPLIER = 0.25
BEST_N = 5

LABELS = {0: "Finalist", 1: "Semis", 2: "Quarters", 3: "Octas", 4: "Doubles", 5: "Triples"}


def bracket_depths(rounds: list) -> list:
    """Rounds from the end (0 = finals) for each elim round of one tournament.

    File order isn't reliable, so the bracket is rebuilt from who played
    where. Round A feeds round B if more debaters who were in A were knocked
    out in B than the reverse; with no eliminations either way, if A's
    winners debate in B (file order breaks a two-way tie). A round's depth is
    the longest path from it to the last round. If the last recorded round
    has more than one debate (the final wasn't recorded), depths shift so the
    debate counts fit: 2 debates = semis.
    """
    n = len(rounds)
    entrants = [{e for row in r for e in row[:2] if e} for r in rounds]
    winners = [{row[2] for row in r if row[2]} for r in rounds]
    losers = [
        {e for row in r if row[2] for e in row[:2] if e and e != row[2]} for r in rounds
    ]

    def feeds(a, b):
        knocked_after = len(losers[b] & entrants[a])
        knocked_before = len(losers[a] & entrants[b])
        if knocked_after or knocked_before:
            return knocked_after > knocked_before
        forward = bool(winners[a] & entrants[b])
        backward = bool(winners[b] & entrants[a])
        return forward and (not backward or a < b)

    after = {a: [b for b in range(n) if b != a and feeds(a, b)] for a in range(n)}
    depth, visiting = {}, set()

    def longest(a):
        if a in depth:
            return depth[a]
        visiting.add(a)
        depth_a = max((longest(b) + 1 for b in after[a] if b not in visiting), default=0)
        visiting.discard(a)
        depth[a] = depth_a
        return depth_a

    from_end = [longest(a) for a in range(n)]
    levels = [math.ceil(math.log2(len(r))) if len(r) > 1 else 0 for r in rounds]
    shift = max(0, max(level - d for level, d in zip(levels, from_end)))
    return [d + shift for d in from_end]


def elim_results(rounds: list) -> dict:
    """Maps debater -> (points, label) for one tournament's elims."""
    depths = bracket_depths(rounds)
    best = {}
    for r, d in zip(rounds, depths):
        for aff, neg, winner in r:
            for entrant in (aff, neg):
                if entrant is None:
                    continue
                if winner == entrant:
                    if d == 0:
                        result = (CHAMPION_POINTS, "Champion")
                    elif d == 1 and not any(dd == 0 for dd in depths):
                        result = (SHARED_FINAL_POINTS, "Finals (not played)")
                    else:
                        continue  # advanced; scored by a later round
                elif winner is None and neg is None:
                    continue
                elif winner is None:
                    # Undecided (closeout or unknown): both reached this round.
                    if d == 0:
                        result = (SHARED_FINAL_POINTS, "Finals (closeout)")
                    else:
                        result = (LOST_IN_POINTS.get(d, DEEPER_POINTS), LABELS.get(d, "Cleared"))
                else:
                    result = (LOST_IN_POINTS.get(d, DEEPER_POINTS), LABELS.get(d, "Cleared"))
                if result[0] > best.get(entrant, (-1, ""))[0]:
                    best[entrant] = result
    # Debaters who won their last recorded round without a later one (e.g.
    # a bye into a round that wasn't recorded) still cleared.
    for r, d in sorted(zip(rounds, depths), key=lambda pair: pair[1]):
        for aff, neg, winner in r:
            if winner and winner not in best:
                best[winner] = (LOST_IN_POINTS.get(max(d - 1, 0), DEEPER_POINTS), LABELS.get(max(d - 1, 0), "Cleared"))
    return best


def round_robin_results(comparisons: list) -> dict:
    """Maps debater -> (points, label) by finishing place on wins (ties share)."""
    wins = {}
    for c in comparisons:
        aff_won = c["aff_ballots"] > c["neg_ballots"]
        wins[c["aff"]] = wins.get(c["aff"], 0) + aff_won
        wins[c["neg"]] = wins.get(c["neg"], 0) + (not aff_won)
    order = sorted(wins, key=lambda e: -wins[e])

    def place_points(place):
        for last_place, points in ROUND_ROBIN_PLACES:
            if place <= last_place:
                return points
        return 0

    results = {}
    i = 0
    while i < len(order):
        tied = [e for e in order if wins[e] == wins[order[i]]]
        places = range(i + 1, i + len(tied) + 1)
        points = sum(place_points(p) for p in places) / len(tied)
        label = f"RR {i + 1}" + ("T" if len(tied) > 1 else "") + f" ({wins[order[i]]} wins)"
        for e in tied:
            if points > 0:
                results[e] = (points, label)
        i += len(tied)
    return results


def field_strengths(comparisons: list, ratings: pd.Series) -> pd.DataFrame:
    entrants = {}
    for c in comparisons:
        entrants.setdefault(c["tournament"], set()).update((c["aff"], c["neg"]))
    rows = []
    for tournament, people in entrants.items():
        top = sorted((ratings.get(p, RATING_BASE) for p in people), reverse=True)[:FIELD_TOP]
        rows.append({"Tournament": tournament, "Entries": len(people), "Field Strength": sum(top) / len(top)})
    fields = pd.DataFrame(rows).set_index("Tournament")
    median = fields["Field Strength"].median()
    fields["Multiplier"] = (1 + (fields["Field Strength"] - median) / FIELD_SCALE).clip(lower=MIN_MULTIPLIER)
    return fields


def build_resume(system, output_prefix: str = "") -> pd.DataFrame:
    """Writes output/{prefix}resume.csv and output/{prefix}tournament_strength.csv."""
    ratings = RATING_BASE + RATING_SCALE * system.skills["ability"]
    fields = field_strengths(system.comparisons, ratings)

    scores = {}
    for tournament in fields.index:
        if tournament.endswith("-rr"):
            results = round_robin_results([c for c in system.comparisons if c["tournament"] == tournament])
        else:
            results = elim_results(system.elim_rounds.get(tournament, []))
        multiplier = fields.loc[tournament, "Multiplier"]
        for debater, (points, label) in results.items():
            scores.setdefault(debater, []).append((points * multiplier, tournament, label))

    info = system.debaters.set_index("hash")
    rows = []
    for debater, results in scores.items():
        results.sort(reverse=True)
        counted = results[:BEST_N]
        rows.append(
            {
                "School": info.loc[debater, "Institution"],
                "Name": info.loc[debater, "Entry"],
                "Resume Score": round(sum(points for points, _, _ in counted), 1),
                "Tournaments Counted": len(counted),
                "Results": "; ".join(f"{t}: {label} ({points:.0f})" for points, t, label in counted),
            }
        )
    resume = pd.DataFrame(rows).sort_values("Resume Score", ascending=False)
    resume.insert(0, "Rank", range(1, len(resume) + 1))
    resume.to_csv(f"output/{output_prefix}resume.csv", index=False)
    fields.sort_values("Field Strength", ascending=False).round(2).to_csv(
        f"output/{output_prefix}tournament_strength.csv"
    )
    print(f"Résumé saved to output/{output_prefix}resume.csv")
    return resume

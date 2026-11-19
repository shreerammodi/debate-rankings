import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resume import (
    CHAMPION_POINTS,
    LOST_IN_POINTS,
    SHARED_FINAL_POINTS,
    bracket_depths,
    elim_results,
    round_robin_results,
)

# An 8-entry bracket: quarters, semis, final, each as (aff, neg, winner).
QUARTERS = [("a", "b", "a"), ("c", "d", "c"), ("e", "f", "e"), ("g", "h", "g")]
SEMIS = [("a", "c", "a"), ("e", "g", "e")]
FINAL = [("a", "e", "e")]


class BracketTest(unittest.TestCase):
    def test_orders_rounds_regardless_of_file_order(self):
        self.assertEqual(bracket_depths([FINAL, QUARTERS, SEMIS]), [0, 2, 1])

    def test_partial_runoff_comes_before_a_larger_round(self):
        runoff = [("x", "y", "x")]
        quarters = [("x", "b", "b")] + QUARTERS[1:]
        self.assertEqual(bracket_depths([runoff, quarters, SEMIS, FINAL])[0], 3)

    def test_points_by_depth(self):
        results = elim_results([QUARTERS, SEMIS, FINAL])
        self.assertEqual(results["e"], (CHAMPION_POINTS, "Champion"))
        self.assertEqual(results["a"][0], LOST_IN_POINTS[0])
        self.assertEqual(results["c"][0], LOST_IN_POINTS[1])
        self.assertEqual(results["b"][0], LOST_IN_POINTS[2])

    def test_missing_final_shares_the_final(self):
        results = elim_results([QUARTERS, SEMIS])
        self.assertEqual(results["a"][0], SHARED_FINAL_POINTS)
        self.assertEqual(results["e"][0], SHARED_FINAL_POINTS)
        self.assertEqual(results["c"][0], LOST_IN_POINTS[1])

    def test_undecided_final_is_a_closeout(self):
        results = elim_results([QUARTERS, SEMIS, [("a", "e", None)]])
        self.assertEqual(results["a"], (SHARED_FINAL_POINTS, "Finals (closeout)"))


class RoundRobinTest(unittest.TestCase):
    def test_places_by_wins_with_shared_ties(self):
        games = [
            {"aff": "a", "neg": "b", "aff_ballots": 1, "neg_ballots": 0},
            {"aff": "a", "neg": "c", "aff_ballots": 1, "neg_ballots": 0},
            {"aff": "b", "neg": "c", "aff_ballots": 1, "neg_ballots": 0},
        ]
        results = round_robin_results(games)
        self.assertEqual(results["a"][0], CHAMPION_POINTS)
        self.assertEqual(results["b"][0], LOST_IN_POINTS[0])
        self.assertEqual(results["c"][0], LOST_IN_POINTS[1])


if __name__ == "__main__":
    unittest.main()

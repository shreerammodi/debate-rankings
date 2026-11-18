import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt_ranking import AFF_PLAYER, TTTRankingSystem


def write_tournament(root: Path, name: str, rounds: list) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Institution": "Alpha", "Entry": "Alice", "Code": "Alpha A"},
            {"Institution": "Beta", "Entry": "Bob", "Code": "Beta B"},
            {"Institution": "Gamma", "Entry": "Cara", "Code": "Gamma C"},
        ]
    ).to_csv(folder / "entries.csv", index=False)
    for i, (aff, neg, win) in enumerate(rounds):
        pd.DataFrame(
            [{"Aff": aff, "Neg": neg, "Judge": "J", "Win": win}]
        ).to_csv(folder / f"0{i}-round.csv", index=False)


class TTTRankingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cwd = os.getcwd()
        os.chdir(self.root)
        (self.root / "output").mkdir()

    def tearDown(self):
        os.chdir(self.cwd)
        self.directory.cleanup()

    def test_ranks_winner_first_and_hides_side_player(self):
        season = self.root / "tournaments" / "fmt"
        write_tournament(season, "one", [("Alpha A", "Beta B", "Aff"), ("Beta B", "Alpha A", "Neg")])
        write_tournament(season, "two", [("Alpha A", "Gamma C", "Aff"), ("Gamma C", "Beta B", "Neg")])
        (self.root / "config.json").write_text(
            json.dumps({"tournaments": {"one": "2026-09-01", "two": "2026-09-08"}})
        )

        system = TTTRankingSystem("config.json", "fmt")
        system.run("t_")

        rankings = pd.read_csv("output/t_full_rankings.csv")
        self.assertEqual(rankings.iloc[0]["Name"], "Alice")
        self.assertEqual(len(rankings), 3)
        self.assertNotIn(AFF_PLAYER, set(rankings["Hash"]))

    def test_last_season_is_the_starting_point(self):
        # Last season Bob beat Cara three times; this season they have no rounds
        # against each other, and both beat Alice once.
        previous = self.root / "archive" / "tournaments" / "fmt"
        write_tournament(
            previous,
            "old",
            [("Beta B", "Gamma C", "Aff"), ("Gamma C", "Beta B", "Neg"), ("Beta B", "Gamma C", "Aff")],
        )
        (self.root / "archive" / "config.json").write_text(json.dumps({"tournaments": ["old"]}))
        season = self.root / "tournaments" / "fmt"
        write_tournament(season, "new", [("Beta B", "Alpha A", "Aff"), ("Gamma C", "Alpha A", "Aff")])
        (self.root / "config.json").write_text(json.dumps({"tournaments": {"new": "2026-09-01"}}))

        seeded = TTTRankingSystem(
            "config.json", "fmt", previous_season=("archive/config.json", "archive/tournaments")
        )
        seeded.run("seeded_")
        rankings = pd.read_csv("output/seeded_full_rankings.csv").set_index("Name")
        self.assertGreater(rankings.loc["Bob", "Rating"], rankings.loc["Cara", "Rating"])
        # Win-rate statistics cover this season only.
        self.assertEqual(rankings.loc["Bob", "Matches"], 1)


if __name__ == "__main__":
    unittest.main()

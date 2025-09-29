from __future__ import annotations
import os
import json

from shared_types import (
    Player,
    ALL_POSITIONS_SHORT_LABELS,
    Optimization,
    PlayerStatisticName,
)
from database import request_players_from_database
from optimization import find_optimized_weights

PLAYERS_JSON_FILENAME = "players.json"
DATABASE_JSON_FILENAME = "ea_fc_players.json"


def get_players() -> list[Player]:
    if os.path.exists(DATABASE_JSON_FILENAME):
        with open(DATABASE_JSON_FILENAME, encoding="utf-8") as f:
            data = json.load(f)
            return data["items"]

    else:
        return request_players_from_database()


def main():
    # players = request_players_from_database()
    players = get_players()

    optimizations: list[Optimization] = []

    for position_short_label in ALL_POSITIONS_SHORT_LABELS:
        if position_short_label == "GK":
            varying_stat_names: list[PlayerStatisticName] = [
                "gkDiving",
                "gkHandling",
                "gkKicking",
                "gkPositioning",
                "gkReflexes",
            ]

        else:
            varying_stat_names: list[PlayerStatisticName] = [
                "pac",
                "sho",
                "pas",
                "dri",
                "def",
                "phy",
            ]

        optimization = find_optimized_weights(
            optimization_name=f"{position_short_label} - Overall Rating",
            players=players,
            position_short_label=position_short_label,
            varying_stats_names=varying_stat_names,
            target_stat_name="overallRating",
        )
        optimizations.append(optimization)

        with open(
            "optimizations.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(optimizations, f)


if __name__ == "__main__":
    main()

from __future__ import annotations
from typing import TypedDict, Literal, Dict, Union, List


PlayerStatisticName = Literal[
    # Pace
    "pac",
    "acceleration",
    "sprintSpeed",
    #
    # Shooting
    "sho",
    "positioning",
    "shotPower",
    "longShots",
    "volleys",
    "penalties",
    #
    # Pass
    "pas",
    "vision",
    "crossing",
    "freeKickAccuracy",
    "shortPassing",
    "longPassing",
    "curve",
    #
    # Dribling
    "dri",
    "agility",
    "balance",
    "reactions",
    "ballControl",
    "dribbling",
    "composure",
    #
    # Defending
    "def",
    "interceptions",
    "headingAccuracy",
    "defensiveAwareness",
    "standingTackle",
    "slidingTackle",
    #
    # Physical
    "phy",
    "jumping",
    "stamina",
    "strength",
    "agrression",
    #
    # Goalkeeper
    "gol",
    "gkDiving",
    "gkHandling",
    "gkKicking",
    "gkPositioning",
    "gkReflexes",
]


PlayerStatistic = TypedDict(
    "PlayerStatistic",
    {
        "value": float,  # Between 0 and 100
        "diff": float,
    },
)

PlayerPositionShortLabel = Literal[
    "GK",
    # Defense
    "CB",
    "LB",
    "RB",
    # Midfield
    "CDM",
    "CM",
    "CAM",
    "LW",
    "LM",
    "RW",
    "RM",
    # Attack
    "ST",
]


PlayerPosition = TypedDict(
    "PlayerPosition",
    {
        "id": str,
        "shortLabel": PlayerPositionShortLabel,
        "label": str,
    },
)


Player = TypedDict(
    "Player",
    {
        "overallRating": float,  # Between 0 and 100
        "stats": Dict[
            PlayerStatisticName,
            PlayerStatistic,
        ],
        "position": PlayerPosition,
    },
)


ALL_POSITIONS_SHORT_LABELS: list[PlayerPositionShortLabel] = [
    "GK",
    # Defense
    "CB",
    "LB",
    "RB",
    # Midfield
    "CDM",
    "CM",
    "CAM",
    "LW",
    "LM",
    "RW",
    "RM",
    # Attack
    "ST",
]


OptimizedWeights = Dict[
    PlayerStatisticName,
    float,
]

TargetStatisticName = Union[
    PlayerStatisticName,
    Literal["overallRating"],
]

Optimization = TypedDict(
    "Optimization",
    {
        "name": str,
        "position_short_label": PlayerPositionShortLabel,
        "varying_stats_names": List[PlayerStatisticName],
        "target_stat_name": TargetStatisticName,
        "optimized_weights": OptimizedWeights,
        "mean_error": float,
    },
)

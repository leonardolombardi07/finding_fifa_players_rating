from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from shared_types import (
    Player,
    PlayerPositionShortLabel,
    PlayerStatisticName,
    Optimization,
    TargetStatisticName,
    OptimizedWeights,
)


def find_optimized_weights(
    optimization_name: str,
    players: list[Player],
    position_short_label: PlayerPositionShortLabel,
    varying_stats_names: list[PlayerStatisticName],
    target_stat_name: TargetStatisticName,
) -> Optimization:
    """For players of given position_short_label, consider that:
    player[target_stat_name] = w1 x varying_stats_names[0] + w2 x varying_stats_names[1] + ...
    Subject to: wi >= 0 and sum(wi) = 1 (weighted average constraint).
    Optimizes weights by minimizing mean absolute relative error using scipy.
    """

    # --- Filter players for this position
    players_for_position = [
        p for p in players if p["position"]["shortLabel"] == position_short_label
    ]

    print(f"Position: {position_short_label}")
    print(f"Number of players: {len(players_for_position)}")
    print("\n")

    # If insufficient data, raise error
    n_features = len(varying_stats_names)
    if not players_for_position or n_features == 0:
        raise ValueError("Not players for position or features to analyze.")

    # --- Helper functions
    def _get_target_values(
        players_subset: list[Player],
    ) -> np.ndarray:
        """Extract target values for all players."""
        values = []
        for player in players_subset:
            if target_stat_name == "overallRating":
                values.append(float(player["overallRating"]))
            else:
                values.append(float(player["stats"][target_stat_name]["value"]))
        return np.array(values)

    def _get_feature_matrix(players_subset: list[Player]) -> np.ndarray:
        """Build feature matrix where each row is a player's stats vector."""
        matrix = []
        for player in players_subset:
            features = [
                float(player["stats"][name]["value"]) for name in varying_stats_names
            ]
            matrix.append(features)
        return np.array(matrix)

    # Prepare data
    y_actual = _get_target_values(players_for_position)
    X = _get_feature_matrix(players_for_position)

    # Filter out players with zero target values to avoid division by zero
    valid_indices = y_actual != 0
    y_actual_valid = y_actual[valid_indices]
    X_valid = X[valid_indices]

    if len(y_actual_valid) == 0:
        # If all target values are zero, return uniform weights
        uniform = 1.0 / n_features
        return {
            "name": optimization_name,
            "position_short_label": position_short_label,
            "varying_stats_names": varying_stats_names,
            "target_stat_name": target_stat_name,
            "optimized_weights": {name: uniform for name in varying_stats_names},
            "mean_error": 0.0,
        }

    def objective(weights: np.ndarray) -> float:
        """
        Objective function: Mean Absolute Relative Error (MARE)
        MARE = mean(|predicted - actual| / |actual|)
        """
        y_predicted = X_valid @ weights  # Matrix multiplication
        relative_errors = np.abs((y_predicted - y_actual_valid) / y_actual_valid)
        return np.mean(relative_errors)

    # --- Set up optimization constraints and bounds
    # Constraint: sum of weights = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: each weight must be >= 0
    bounds = [(0, None) for _ in range(n_features)]

    # Initial guess: uniform weights
    initial_weights = np.ones(n_features) / n_features

    # --- Optimize using scipy
    # Try multiple optimization methods for robustness
    methods = ["SLSQP", "trust-constr"]
    best_result = None
    best_error = float("inf")

    for method in methods:
        try:
            if method == "trust-constr":
                # trust-constr uses different constraint format
                from scipy.optimize import LinearConstraint

                # Sum constraint: 1^T * w = 1
                linear_constraint = LinearConstraint(np.ones((1, n_features)), 1.0, 1.0)
                result = minimize(
                    objective,
                    initial_weights,
                    method=method,
                    bounds=bounds,
                    constraints=[linear_constraint],
                    options={"maxiter": 1000},
                )
            else:
                result = minimize(
                    objective,
                    initial_weights,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )

            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun

        except Exception as e:
            print(f"Warning: Optimization with {method} failed: {e}")
            continue

    # If no optimization succeeded, fall back to uniform weights
    if best_result is None:
        print("Warning: All optimization methods failed. Using uniform weights.")
        optimized_weights_array = initial_weights
        mean_error = objective(initial_weights)
    else:
        optimized_weights_array = best_result.x
        mean_error = best_result.fun

        # Ensure weights sum to exactly 1 (numerical precision fix)
        optimized_weights_array = optimized_weights_array / np.sum(
            optimized_weights_array
        )

    # Convert to dictionary format
    optimized_weights_dict: OptimizedWeights = {
        name: float(weight)
        for name, weight in zip(varying_stats_names, optimized_weights_array)
    }

    # --- Optional: Print sample predictions for debugging
    print(f"Optimization completed. Mean relative error: {mean_error:.4f}")
    print(f"Optimized weights: {optimized_weights_dict}")

    # Show predictions for first few players
    sample_size = min(10, len(players_for_position))
    if sample_size > 0:
        print("\nSample predictions (first 10 players):")
        for i, player in enumerate(players_for_position[:sample_size]):
            x = np.array(
                [float(player["stats"][name]["value"]) for name in varying_stats_names]
            )
            y_actual_player = (
                float(player["overallRating"])
                if target_stat_name == "overallRating"
                else float(player["stats"][target_stat_name]["value"])
            )
            y_predicted = np.dot(optimized_weights_array, x)

            if y_actual_player != 0:
                error = abs((y_predicted - y_actual_player) / y_actual_player)
                print(
                    f"  Player {i + 1}: "
                    f"Actual={y_actual_player:.1f}, "
                    f"Predicted={y_predicted:.1f}, "
                    f"Error={error * 100:.2f}%"
                )

    print("-" * 50)

    return {
        "name": optimization_name,
        "position_short_label": position_short_label,
        "varying_stats_names": varying_stats_names,
        "target_stat_name": target_stat_name,
        "optimized_weights": optimized_weights_dict,
        "mean_error": float(mean_error),
    }

from __future__ import annotations

from typing import Any

from .types import AggregationOutput


def compute_instance_risks(
    base_votes: dict[str, int],
    probe_votes_by_family: dict[str, dict[str, int]],
) -> dict[str, float]:
    risks: dict[str, float] = {}
    for family, family_votes in probe_votes_by_family.items():
        common = sorted(set(base_votes) & set(family_votes))
        if not common:
            risks[family] = 0.0
            continue
        flips = sum(1 for sensor in common if base_votes[sensor] != family_votes[sensor])
        risks[family] = flips / float(len(common))
    return risks


def compute_sensor_disagreement(base_votes: dict[str, int]) -> float:
    if len(base_votes) < 2:
        return 0.0
    votes = list(base_votes.values())
    n_pos = sum(1 for v in votes if v == 1)
    n_neg = len(votes) - n_pos
    disagreement = 2.0 * min(n_pos, n_neg) / len(votes)
    return float(disagreement)


def _finalize_output(
    base_votes: dict[str, int],
    sensor_weights: dict[str, float],
    margin: float,
    instance_risks: dict[str, float] | None = None,
) -> AggregationOutput:
    weighted_sum = sum(sensor_weights.get(sensor, 0.0) * vote for sensor, vote in base_votes.items())
    total_weight = sum(sensor_weights.values())
    abstained = (total_weight == 0.0) or (abs(weighted_sum) < margin)
    prediction = 1 if weighted_sum >= 0 else -1
    confidence = abs(weighted_sum) / max(total_weight, 1e-8)
    return AggregationOutput(
        prediction=prediction,
        abstained=abstained,
        score=float(weighted_sum),
        confidence=float(confidence),
        sensor_weights=sensor_weights,
        sensor_votes=base_votes,
        instance_risks=instance_risks or {},
    )


def aggregate_single_sensor(
    base_votes: dict[str, int],
    sensor_name: str,
) -> AggregationOutput:
    sensor_weights = {sensor: 1.0 if sensor == sensor_name else 0.0 for sensor in base_votes}
    return _finalize_output(base_votes, sensor_weights, margin=0.0, instance_risks={})


def aggregate_majority(
    base_votes: dict[str, int],
) -> AggregationOutput:
    sensor_weights = {sensor: 1.0 for sensor in base_votes}
    return _finalize_output(base_votes, sensor_weights, margin=0.0, instance_risks={})


def aggregate_clean_weighted(
    base_votes: dict[str, int],
    profiles: dict[str, dict[str, Any]],
    margin: float,
) -> AggregationOutput:
    sensor_weights = {
        sensor: max(0.0, float(profiles[sensor]["base_weight"]))
        for sensor in base_votes
    }
    return _finalize_output(base_votes, sensor_weights, margin=margin, instance_risks={})


def aggregate_selectively(
    base_votes: dict[str, int],
    profiles: dict[str, dict[str, Any]],
    probe_votes_by_family: dict[str, dict[str, int]],
    lambda_attack: float,
    rho_dependence: float,
    margin: float,
) -> AggregationOutput:
    instance_risks = compute_instance_risks(base_votes, probe_votes_by_family)

    disagreement = compute_sensor_disagreement(base_votes)
    adjusted_margin = margin + 0.3 * disagreement

    sensor_weights: dict[str, float] = {}

    for sensor, vote in base_votes.items():
        profile = profiles[sensor]

        if instance_risks:
            family_penalties = [
                instance_risks.get(family, 0.0) * float(profile["vulnerabilities"].get(family, 0.0))
                for family in instance_risks
            ]
            worst_penalty = max(family_penalties) if family_penalties else 0.0
            avg_penalty = sum(family_penalties) / len(family_penalties) if family_penalties else 0.0
            penalty = 0.7 * worst_penalty + 0.3 * avg_penalty
        else:
            penalty = 0.0

        majority_direction = 1 if sum(base_votes.values()) >= 0 else -1
        outlier_penalty = 0.2 if vote != majority_direction else 0.0

        weight = (
            float(profile["base_weight"])
            - lambda_attack * penalty
            - rho_dependence * float(profile["dependence"])
            - outlier_penalty
        )
        sensor_weights[sensor] = max(0.0, weight)

    return _finalize_output(base_votes, sensor_weights, adjusted_margin, instance_risks=instance_risks)
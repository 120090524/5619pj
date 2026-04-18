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
    sensor_weights: dict[str, float] = {}

    dep_values = [float(profiles[s]["dependence"]) for s in base_votes if s in profiles]
    mean_dep = sum(dep_values) / len(dep_values) if dep_values else 0.0

    for sensor, vote in base_votes.items():
        profile = profiles[sensor]
        penalty = 0.0
        for family, risk in instance_risks.items():
            penalty += risk * float(profile["vulnerabilities"].get(family, 0.0))
        # 只惩罚比平均 dependence 高的 sensor，奖励比平均低的 sensor
        dep_delta = float(profile["dependence"]) - mean_dep
        weight = float(profile["base_weight"]) - lambda_attack * penalty - rho_dependence * dep_delta
        sensor_weights[sensor] = max(0.0, weight)

    return _finalize_output(base_votes, sensor_weights, margin=margin, instance_risks=instance_risks)
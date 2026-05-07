from __future__ import annotations

import numpy as np
from typing import Any

from .types import AggregationOutput


class MetaAggregator:
    """
    Learned meta-aggregator that replaces the hand-crafted SENTRY weight formula.

    Instead of:
        weight = base_weight - lambda * attack_penalty - rho * dependence_penalty

    We learn a small logistic regression on calibration data that takes each
    sensor's feature vector as input and outputs a reliability score used as weight.

    Features per sensor (per instance):
        - base_weight       : log-odds reliability from clean calibration
        - dependence        : average correlation with other sensors
        - instance_risk     : probe flip rate on this instance (max across attack families)
        - vulnerability     : sensor's historical sensitivity to the detected risk family
        - vote              : the sensor's actual vote (+1 / -1), encoded as 0/1
        - confidence_proxy  : |base_weight| normalized to [0, 1]

    Training target:
        For each (sensor, example) pair in calibration, label = 1 if the sensor
        was correct, 0 if wrong. The meta-aggregator learns to predict correctness
        from the feature vector, then uses predicted P(correct) as the weight.

    Abstention:
        Learned threshold tau on the weighted confidence score. Optimized on
        calibration set to maximize robust accuracy under a coverage constraint.
    """

    def __init__(self, coverage_floor: float = 0.80) -> None:
        self.coverage_floor = coverage_floor
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._tau: float = 0.0
        self._fitted = False

    # ── Feature extraction ────────────────────────────────────────────────────

    def _sensor_features(
        self,
        sensor: str,
        vote: int,
        profile: dict[str, Any],
        instance_risks: dict[str, float],
    ) -> np.ndarray:
        base_weight = float(profile["base_weight"])
        dependence = float(profile["dependence"])
        vulnerabilities = profile.get("vulnerabilities", {})

        # Worst-case instance risk weighted by sensor vulnerability
        worst_risk = 0.0
        for family, risk in instance_risks.items():
            vuln = float(vulnerabilities.get(family, 0.0))
            worst_risk = max(worst_risk, risk * vuln)

        avg_vulnerability = float(np.mean(list(vulnerabilities.values()))) if vulnerabilities else 0.0
        confidence_proxy = min(1.0, abs(base_weight) / 3.0)
        vote_encoded = 1.0 if vote == 1 else 0.0

        return np.array([
            base_weight,
            dependence,
            worst_risk,
            avg_vulnerability,
            confidence_proxy,
            vote_encoded,
        ], dtype=np.float32)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        calibration_rows: list[dict[str, Any]],
        profiles: dict[str, dict[str, Any]],
        probe_votes_by_example: dict[str, dict[str, dict[str, int]]],
    ) -> None:
        """
        Train the meta-aggregator on calibration data.

        calibration_rows: list of dicts with keys:
            example_id, sensor, prediction, label
        profiles: sensor profiles from compute_profiles()
        probe_votes_by_example: {example_id: {family: {sensor: vote}}}
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X_rows, y_rows = [], []

        for row in calibration_rows:
            sensor = row["sensor"]
            vote = int(row["prediction"])
            label = int(row["label"])
            example_id = row["example_id"]

            if sensor not in profiles:
                continue

            probe_votes_by_family = probe_votes_by_example.get(example_id, {})
            base_votes_for_example = {
                r["sensor"]: int(r["prediction"])
                for r in calibration_rows
                if r["example_id"] == example_id
            }
            instance_risks = _compute_instance_risks(base_votes_for_example, probe_votes_by_family)

            features = self._sensor_features(sensor, vote, profiles[sensor], instance_risks)
            correct = 1 if vote == label else 0

            X_rows.append(features)
            y_rows.append(correct)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
        clf.fit(X_scaled, y)

        self._coef = clf.coef_[0]
        self._intercept = float(clf.intercept_[0])
        self._fitted = True

        self._tau = self._find_tau(calibration_rows, profiles, probe_votes_by_example)

    def _predict_correctness_prob(self, features: np.ndarray) -> float:
        scaled = self._scaler.transform(features.reshape(1, -1))[0]
        logit = float(np.dot(self._coef, scaled) + self._intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _find_tau(
        self,
        calibration_rows: list[dict[str, Any]],
        profiles: dict[str, dict[str, Any]],
        probe_votes_by_example: dict[str, dict[str, dict[str, int]]],
    ) -> float:
        example_ids = list({r["example_id"] for r in calibration_rows})
        labels = {r["example_id"]: int(r["label"]) for r in calibration_rows}

        scores = []
        for example_id in example_ids:
            rows = [r for r in calibration_rows if r["example_id"] == example_id]
            base_votes = {r["sensor"]: int(r["prediction"]) for r in rows}
            probe_votes_by_family = probe_votes_by_example.get(example_id, {})
            instance_risks = _compute_instance_risks(base_votes, probe_votes_by_family)

            weighted_sum, total_weight = 0.0, 0.0
            for r in rows:
                sensor = r["sensor"]
                vote = int(r["prediction"])
                if sensor not in profiles:
                    continue
                features = self._sensor_features(sensor, vote, profiles[sensor], instance_risks)
                prob = self._predict_correctness_prob(features)
                weighted_sum += prob * vote
                total_weight += prob

            confidence = abs(weighted_sum) / max(total_weight, 1e-8)
            prediction = 1 if weighted_sum >= 0 else -1
            scores.append((confidence, prediction, labels[example_id]))

        scores.sort(key=lambda x: x[0])
        best_tau, best_robust_acc = 0.0, 0.0

        for i in range(len(scores)):
            tau_candidate = scores[i][0]
            covered = [(pred, lab) for conf, pred, lab in scores if conf >= tau_candidate]
            coverage = len(covered) / len(scores)
            if coverage < self.coverage_floor:
                break
            acc = sum(1 for pred, lab in covered if pred == lab) / max(len(covered), 1)
            if acc > best_robust_acc:
                best_robust_acc = acc
                best_tau = tau_candidate

        return best_tau

    # ── Inference ─────────────────────────────────────────────────────────────

    def aggregate(
        self,
        base_votes: dict[str, int],
        profiles: dict[str, dict[str, Any]],
        probe_votes_by_family: dict[str, dict[str, int]],
    ) -> AggregationOutput:
        if not self._fitted:
            raise RuntimeError("MetaAggregator must be fitted before calling aggregate().")

        instance_risks = _compute_instance_risks(base_votes, probe_votes_by_family)
        sensor_weights: dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for sensor, vote in base_votes.items():
            if sensor not in profiles:
                sensor_weights[sensor] = 0.0
                continue
            features = self._sensor_features(sensor, vote, profiles[sensor], instance_risks)
            prob = self._predict_correctness_prob(features)
            sensor_weights[sensor] = prob
            weighted_sum += prob * vote
            total_weight += prob

        confidence = abs(weighted_sum) / max(total_weight, 1e-8)
        prediction = 1 if weighted_sum >= 0 else -1
        abstained = (total_weight == 0.0) or (confidence < self._tau)

        return AggregationOutput(
            prediction=prediction,
            abstained=abstained,
            score=float(weighted_sum),
            confidence=float(confidence),
            sensor_weights=sensor_weights,
            sensor_votes=base_votes,
            instance_risks=instance_risks,
        )


# ── Helper (mirrors aggregator.py) ────────────────────────────────────────────

def _compute_instance_risks(
    base_votes: dict[str, int],
    probe_votes_by_family: dict[str, dict[str, int]],
) -> dict[str, float]:
    risks: dict[str, float] = {}
    for family, family_votes in probe_votes_by_family.items():
        common = sorted(set(base_votes) & set(family_votes))
        if not common:
            risks[family] = 0.0
            continue
        flips = sum(1 for s in common if base_votes[s] != family_votes[s])
        risks[family] = flips / float(len(common))
    return risks